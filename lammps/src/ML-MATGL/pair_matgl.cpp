/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   pair_matgl: TorchScript bridge to a MatGL TensorNet PES.

   Mirrors ACEsuit/lammps:src/ML-MACE/pair_mace.cpp's overall control flow,
   adapted to the LAMMPSMatGLModel forward signature defined in
   matgl/src/matgl/ext/_lammps.py:

     forward(positions, edge_index, unit_shifts, cell, atomic_numbers,
             local_or_ghost, compute_virials)
       -> {total_energy_local, node_energy, forces, virials}

   The model owns the autograd machinery; this pair style is a thin shim
   that translates LAMMPS data structures into the tensors that
   forward expects, then accumulates the returned forces and virials back
   into LAMMPS arrays.

   Required LAMMPS commands (also documented in lammps/README.md):
     atom_modify map yes
     newton on
     pair_style matgl
     pair_coeff * * <model.pt> <species1> <species2> ...
------------------------------------------------------------------------- */

#include "pair_matgl.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

using namespace LAMMPS_NS;

namespace {

constexpr const char *kPeriodicTable[] = {
    "X",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",   "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn",  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh",  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd",  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re",  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th",  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db",  "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
};
constexpr int kNumElements = sizeof(kPeriodicTable) / sizeof(kPeriodicTable[0]);

int symbol_to_z(const std::string &sym)
{
  for (int z = 1; z < kNumElements; ++z) {
    if (sym == kPeriodicTable[z]) return z;
  }
  return -1;
}

}  // namespace

/* ---------------------------------------------------------------------- */

PairMATGL::PairMATGL(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;        // no per-pair energy decomposition
  restartinfo = 0;          // model lives on disk, not in restart files
  one_coeff = 1;            // a single pair_coeff line covers all type pairs
  manybody_flag = 1;        // GNN: not pairwise additive
  centroidstressflag = CENTROID_NOTAVAIL;
  no_virial_fdotr_compute = 1;  // we set the virial directly from the model
  unit_convert_flag = 0;
}

/* ---------------------------------------------------------------------- */

PairMATGL::~PairMATGL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ----------------------------------------------------------------------
   global settings: optional `no_domain_decomposition` keyword
------------------------------------------------------------------------- */

void PairMATGL::settings(int narg, char **arg)
{
  no_domain_decomposition_ = false;
  for (int i = 0; i < narg; ++i) {
    if (std::strcmp(arg[i], "no_domain_decomposition") == 0) {
      no_domain_decomposition_ = true;
    } else {
      error->all(FLERR, "Illegal pair_style matgl option: {}", arg[i]);
    }
  }
}

/* ----------------------------------------------------------------------
   parse `pair_coeff * * <model.pt> S1 S2 ...`
------------------------------------------------------------------------- */

void PairMATGL::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes)
    error->all(FLERR,
               "pair_coeff matgl expects: * * <model.pt> {} species names "
               "(one per LAMMPS atom type, in order)",
               ntypes);

  if (std::strcmp(arg[0], "*") != 0 || std::strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff matgl requires both type indices to be '*'");

  std::string model_path = arg[2];
  {
    namespace fs = std::filesystem;
    fs::path p(model_path);
    fs::path dir = fs::is_directory(p) ? p : p.parent_path();
    fs::path native_config = dir / "model.json";
    fs::path native_state = dir / "state.pt";

    if (fs::exists(native_config) && fs::exists(native_state)) {
      // `dir` holds a matgl IOMixIn checkpoint (model.json + model.pt +
      // state.pt) rather than a LAMMPS-ready TorchScript module. Export
      // (and cache) a scripted model alongside it, then load that instead.
      fs::path cached = dir / "lammps_model.pt";

      bool need_export = !fs::exists(cached);
      if (!need_export) {
        auto cached_time = fs::last_write_time(cached);
        for (const auto &src : {native_config, native_state})
          if (fs::last_write_time(src) > cached_time) need_export = true;
      }

      if (need_export) {
        if (comm->me == 0) {
          utils::logmesg(lmp,
                         "pair_matgl: exporting matgl checkpoint '{}' to LAMMPS "
                         "TorchScript format (one-time, cached as '{}')...\n",
                         dir.string(), cached.string());
          static const std::string kPython = "/global/cfs/cdirs/m4845/repos/matgl/.venv/bin/python";
          static const std::string kExportScript =
              "/global/cfs/cdirs/m4845/repos/lammps/src/ML-MATGL/export_matgl_checkpoint.py";
          std::string cmd = kPython + " '" + kExportScript + "' '" + dir.string() + "' '" +
                             cached.string() + "'";
          int rc = std::system(cmd.c_str());
          if (rc != 0)
            error->one(FLERR, "pair_matgl: export of matgl checkpoint '{}' failed (command '{}' exited {})",
                       dir.string(), cmd, rc);
        }
        MPI_Barrier(world);
        if (!fs::exists(cached))
          error->all(FLERR, "pair_matgl: expected exported model '{}' not found after export",
                     cached.string());
      }

      model_path = cached.string();
    } else if (fs::exists(p) && fs::is_directory(p)) {
      model_path = (p / "model.pt").string();
    }
  }

  // Try-catch around torch::jit::load: if libtorch can't read the file or the
  // module's forward signature isn't ours, surface a useful error early.
  try {
    model_ = torch::jit::load(model_path);
  } catch (const std::exception &e) {
    error->all(FLERR, "Could not load TorchScript model '{}': {}", model_path, e.what());
  }
  model_.eval();

  // Pull r_max and dtype out of the scripted module's named buffers. The
  // Python wrapper guarantees these are present.
  bool found_r_max = false;
  bool found_dtype_probe = false;
  for (const auto &b : model_.named_buffers()) {
    if (b.name == "z_to_index") {
      // dtype probe: the model's compute dtype matches its non-integer
      // buffers (data_mean, data_std, element_refs). z_to_index is int64.
      // We pick element_refs below.
      continue;
    }
    if (b.name == "element_refs") {
      dtype_ = b.value.scalar_type();
      found_dtype_probe = true;
    }
  }
  // Read r_max from a Python attribute (compiled to an IValue).
  try {
    auto attr = model_.attr("r_max");
    if (attr.isDouble()) {
      r_max_ = attr.toDouble();
    } else if (attr.isInt()) {
      r_max_ = static_cast<double>(attr.toInt());
    }
    found_r_max = (r_max_ > 0.0);
  } catch (const std::exception &) {
    // fall through; we'll error below
  }

  if (!found_r_max)
    error->all(FLERR,
               "TorchScript model is missing the `r_max` attribute — was it produced "
               "by `mgl create-lammps-model`?");
  if (!found_dtype_probe)
    error->all(FLERR,
               "TorchScript model is missing the `element_refs` buffer — was it "
               "produced by `mgl create-lammps-model`?");

  r_max_squared_ = r_max_ * r_max_;

  // Map LAMMPS atom-type index (1-based) -> atomic number Z.
  type_to_z_.assign(ntypes + 1, 0);
  for (int t = 1; t <= ntypes; ++t) {
    const std::string sym = arg[2 + t];
    const int z = ::symbol_to_z(sym);
    if (z < 0) error->all(FLERR, "pair_matgl: unknown species symbol '{}'", sym);
    type_to_z_[t] = z;
  }

  for (int i = 1; i <= ntypes; ++i)
    for (int j = i; j <= ntypes; ++j) setflag[i][j] = 1;

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "pair_matgl: loaded {} (r_max={:.4f} Å, dtype={})\n",
                   model_path,
                   r_max_,
                   (dtype_ == torch::kFloat64 ? "float64" : "float32"));
}

/* ---------------------------------------------------------------------- */

void PairMATGL::allocate()
{
  allocated = 1;
  const int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; ++i)
    for (int j = i; j < n; ++j) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

double PairMATGL::init_one(int /*i*/, int /*j*/)
{
  // All type pairs share the model cutoff.
  return r_max_;
}

/* ---------------------------------------------------------------------- */

void PairMATGL::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "pair_style matgl requires atom-IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "pair_style matgl requires `newton on`");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,
               "pair_style matgl requires `atom_modify map yes` so neighbor "
               "lookups can resolve ghost atoms");

  // Full neighbor list with ghost atoms — the Python wrapper expects
  // edge_index to point at the same `positions` table used for both owned
  // and ghost atoms.
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
}

/* ----------------------------------------------------------------------
   the heart of the pair style: build edge tensors, run the model,
   accumulate forces and the virial
------------------------------------------------------------------------- */

void PairMATGL::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (eflag_atom)
    error->all(FLERR, "pair_matgl does not support per-atom energies");
  if (vflag_atom)
    error->all(FLERR, "pair_matgl does not support per-atom virials");

  const int inum = list->inum;
  const int *const ilist = list->ilist;
  const int *const numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const double *const *const x = atom->x;
  double *const *const f = atom->f;
  const int *const type = atom->type;

  // 1) Allocate Cartesian / atomic-number / mask buffers sized to nall.
  auto opts_real = torch::TensorOptions().dtype(dtype_);
  auto opts_long = torch::TensorOptions().dtype(torch::kInt64);
  auto opts_bool = torch::TensorOptions().dtype(torch::kBool);

  torch::Tensor positions = torch::empty({nall, 3}, opts_real);
  torch::Tensor atomic_numbers = torch::empty({nall}, opts_long);
  torch::Tensor local_or_ghost = torch::empty({nall}, opts_bool);

  // Fill them. Promote to the model's dtype on the fly.
  if (dtype_ == torch::kFloat64) {
    auto pos_a = positions.accessor<double, 2>();
    for (int i = 0; i < nall; ++i) {
      pos_a[i][0] = x[i][0];
      pos_a[i][1] = x[i][1];
      pos_a[i][2] = x[i][2];
    }
  } else {
    auto pos_a = positions.accessor<float, 2>();
    for (int i = 0; i < nall; ++i) {
      pos_a[i][0] = static_cast<float>(x[i][0]);
      pos_a[i][1] = static_cast<float>(x[i][1]);
      pos_a[i][2] = static_cast<float>(x[i][2]);
    }
  }
  {
    auto z_a = atomic_numbers.accessor<int64_t, 1>();
    auto m_a = local_or_ghost.accessor<bool, 1>();
    for (int i = 0; i < nall; ++i) {
      z_a[i] = type_to_z_[type[i]];
      m_a[i] = (i < nlocal);
    }
  }

  // 2) Walk the neighbor list, filter by r_max_squared, build edge_index +
  //    unit_shifts. Ghost positions are already wrapped+imaged by LAMMPS,
  //    so we recover the integer image triple from the positional offset
  //    relative to the owned image of the atom.
  std::vector<int64_t> edge_src;
  std::vector<int64_t> edge_dst;
  std::vector<int64_t> edge_shifts;  // flat (E*3,)
  edge_src.reserve(nall * 32);
  edge_dst.reserve(nall * 32);
  edge_shifts.reserve(nall * 32 * 3);

  // TensorNet's message-passing layers require every physical atom to be
  // addressed by ONE consistent row index across all its edges; periodicity
  // must be encoded as a position offset on that same row (unit_shifts @
  // cell), not as a second "ghost" row. A ghost row only ever appears as an
  // edge destination here (only owned atoms are loop centers below), so it
  // never propagates outgoing messages and the real atom never receives the
  // message that landed on its ghost -- silently dropping every
  // periodic-boundary bond's contribution. So: resolve every neighbor `j`
  // back to the local row of the atom it represents via its tag, and
  // recover the integer image vector from the (cancelling-boxlo) Cartesian
  // offset between the ghost and its owned copy.
  if (comm->nprocs > 1)
    error->all(FLERR,
               "pair_style matgl: multi-rank (MPI-decomposed) runs are not "
               "yet supported -- ghost atoms may be owned by a different "
               "rank, so there is no local row to fold periodic edges back "
               "onto. Run with a single MPI rank.");

  const double *const h_inv = domain->h_inv;

  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    const double xi = x[i][0];
    const double yi = x[i][1];
    const double zi = x[i][2];
    const int *const jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - xi;
      const double dy = x[j][1] - yi;
      const double dz = x[j][2] - zi;
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq > r_max_squared_) continue;

      // Fold `j` (local or ghost) back onto the local row of the atom it
      // represents. For a local atom this is `j` itself (shift 0,0,0); for
      // a ghost it's the owned copy with the same tag.
      const int j_local = atom->map(atom->tag[j]);
      if (j_local < 0 || j_local >= nlocal)
        error->one(FLERR,
                   "pair_style matgl: could not resolve neighbor atom "
                   "{} (tag {}) to a local row -- is `atom_modify map yes` "
                   "set, and is this a single-rank run?",
                   j, atom->tag[j]);

      // Integer image vector (nx,ny,nz) such that
      //   x[j] == x[j_local] + (nx,ny,nz) @ cell
      // boxlo cancels in the difference, so this is the same affine map
      // Domain::x2lamda uses, applied to (x[j] - x[j_local]) directly.
      const double ddx = x[j][0] - x[j_local][0];
      const double ddy = x[j][1] - x[j_local][1];
      const double ddz = x[j][2] - x[j_local][2];
      const double lz = h_inv[2] * ddz;
      const double ly = h_inv[1] * ddy + h_inv[3] * ddz;
      const double lx = h_inv[0] * ddx + h_inv[5] * ddy + h_inv[4] * ddz;
      const int64_t nx = static_cast<int64_t>(std::lround(lx));
      const int64_t ny = static_cast<int64_t>(std::lround(ly));
      const int64_t nz = static_cast<int64_t>(std::lround(lz));

      edge_src.push_back(static_cast<int64_t>(i));
      edge_dst.push_back(static_cast<int64_t>(j_local));
      edge_shifts.push_back(nx);
      edge_shifts.push_back(ny);
      edge_shifts.push_back(nz);
    }
  }

  const int64_t E = static_cast<int64_t>(edge_src.size());
  torch::Tensor edge_index =
      torch::empty({2, E}, opts_long);
  torch::Tensor unit_shifts = torch::empty({E, 3}, opts_long);
  if (E > 0) {
    auto ei_a = edge_index.accessor<int64_t, 2>();
    auto us_a = unit_shifts.accessor<int64_t, 2>();
    for (int64_t e = 0; e < E; ++e) {
      ei_a[0][e] = edge_src[e];
      ei_a[1][e] = edge_dst[e];
      us_a[e][0] = edge_shifts[3 * e + 0];
      us_a[e][1] = edge_shifts[3 * e + 1];
      us_a[e][2] = edge_shifts[3 * e + 2];
    }
  }

  // 3) Cell (row-vector basis). LAMMPS stores h_inv etc.; we build the cell
  //    from boxlo/boxhi/xy/xz/yz.
  torch::Tensor cell = torch::zeros({3, 3}, opts_real);
  {
    const double xprd = domain->xprd;
    const double yprd = domain->yprd;
    const double zprd = domain->zprd;
    const double xy = domain->xy;
    const double xz = domain->xz;
    const double yz = domain->yz;
    if (dtype_ == torch::kFloat64) {
      auto c = cell.accessor<double, 2>();
      c[0][0] = xprd;
      c[1][0] = xy;   c[1][1] = yprd;
      c[2][0] = xz;   c[2][1] = yz;   c[2][2] = zprd;
    } else {
      auto c = cell.accessor<float, 2>();
      c[0][0] = static_cast<float>(xprd);
      c[1][0] = static_cast<float>(xy);   c[1][1] = static_cast<float>(yprd);
      c[2][0] = static_cast<float>(xz);   c[2][1] = static_cast<float>(yz);
      c[2][2] = static_cast<float>(zprd);
    }
  }

  // 4) Run the scripted forward.
  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(7);
  inputs.emplace_back(positions);
  inputs.emplace_back(edge_index);
  inputs.emplace_back(unit_shifts);
  inputs.emplace_back(cell);
  inputs.emplace_back(atomic_numbers);
  inputs.emplace_back(local_or_ghost);
  inputs.emplace_back(static_cast<bool>(vflag_global));

  torch::IValue result;
  try {
    result = model_.forward(inputs);
  } catch (const std::exception &e) {
    error->all(FLERR, "pair_matgl: model forward failed: {}", e.what());
  }
  auto out = result.toGenericDict();

  // 5) Read scalars + forces back into LAMMPS arrays.
  auto total_energy = out.at("total_energy_local").toTensor().to(torch::kFloat64).item<double>();
  if (eflag_global) eng_vdwl += total_energy;

  torch::Tensor forces_t = out.at("forces").toTensor().to(torch::kFloat64);
  auto fa = forces_t.accessor<double, 2>();
  for (int i = 0; i < nall; ++i) {
    f[i][0] += fa[i][0];
    f[i][1] += fa[i][1];
    f[i][2] += fa[i][2];
  }

  // 6) Virial — the model returns dE/dstrain. LAMMPS uses
  //    W = sum_i r_i (x) f_i = -dE/dstrain. Store the six Voigt components
  //    in the order xx, yy, zz, xy, xz, yz.
  if (vflag_global) {
    auto vir_t = out.at("virials").toTensor().to(torch::kFloat64);
    auto va = vir_t.accessor<double, 2>();
    virial[0] -= va[0][0];
    virial[1] -= va[1][1];
    virial[2] -= va[2][2];
    virial[3] -= 0.5 * (va[0][1] + va[1][0]);
    virial[4] -= 0.5 * (va[0][2] + va[2][0]);
    virial[5] -= 0.5 * (va[1][2] + va[2][1]);
  }
}
