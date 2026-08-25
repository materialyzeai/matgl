/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   pair_matgl/kk: Kokkos variant of pair_matgl. See pair_matgl_kokkos.h.

   Notes for readers familiar with pair_mace_kokkos.cpp: the overall control
   flow is identical (count edges, scan, fill, hand off to libtorch, scatter
   forces). The only matgl-specific parts are:

     * the model forward signature (we pass `compute_virials: bool`),
     * the output dict keys (`total_energy_local`, `forces`, `virials`),
     * every edge is folded back onto a *local* row (via `local_row_of_`)
       with a recovered integer `unit_shifts`, rather than pointing at the
       ghost row directly -- TensorNet's message-passing layers need one
       consistent row per physical atom, and a ghost row never propagates
       outgoing messages back to the atom it duplicates (see pair_matgl.cpp
       for the full explanation and how the shift is recovered).

   Caveats on multi-rank Kokkos with libtorch (mirroring MACE):
     * Single-GPU runs are the supported configuration.
     * `mpirun -n 1 lmp -k on g 1 -sf kk` works.
     * Multi-rank Kokkos with libtorch is unreliable (see MACE issues #1294,
       #322); this pair style does not attempt to fix that.
------------------------------------------------------------------------- */

#include "pair_matgl_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"

#include <algorithm>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairMATGLKokkos<DeviceType>::PairMATGLKokkos(LAMMPS *lmp) : PairMATGL(lmp)
{
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairMATGLKokkos<DeviceType>::~PairMATGLKokkos() = default;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairMATGLKokkos<DeviceType>::init_style()
{
  PairMATGL::init_style();

  // Replace the host neighbor request with a Kokkos one.
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType, LMPHostType>::value &&
                          !std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);

  // Pick the matching libtorch device.
  if (std::is_same<DeviceType, Kokkos::Cuda>::value) {
    int gpu = 0;
    if (lmp->kokkos->ngpus > 0) cudaGetDevice(&gpu);
    torch_device_ = torch::Device(torch::kCUDA, gpu);
  } else {
    torch_device_ = torch::kCPU;
  }
  // Move the model to the matching device. ``torch::jit::Module::to`` is
  // safe to call repeatedly.
  model_.to(torch_device_);
  if (comm->me == 0)
    utils::logmesg(lmp, "pair_matgl/kk: model on {}\n",
                   torch_device_.is_cuda() ? "cuda" : "cpu");

  // Materialize the type->Z table on device.
  const int ntypes = atom->ntypes;
  d_type_to_z_ = Kokkos::View<int64_t *, DeviceType>("matgl:type_to_z", ntypes + 1);
  auto h_type_to_z = Kokkos::create_mirror_view(d_type_to_z_);
  for (int t = 0; t <= ntypes; ++t) h_type_to_z(t) = (t == 0) ? 0 : type_to_z_[t];
  Kokkos::deep_copy(d_type_to_z_, h_type_to_z);
}

/* ----------------------------------------------------------------------
   helper: a torch::Tensor view of a Kokkos device buffer (no copy).
------------------------------------------------------------------------- */

namespace {

template <class View>
torch::Tensor blob_from_view(const View &v, torch::TensorOptions opts)
{
  std::vector<int64_t> shape;
  shape.reserve(View::rank);
  for (size_t r = 0; r < View::rank; ++r)
    shape.push_back(static_cast<int64_t>(v.extent(r)));
  return torch::from_blob(v.data(), shape, opts);
}

}  // namespace

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairMATGLKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (eflag_atom)
    error->all(FLERR, "pair_matgl/kk does not support per-atom energies");
  if (vflag_atom)
    error->all(FLERR, "pair_matgl/kk does not support per-atom virials");

  atomKK->sync(execution_space, datamask_read);
  atomKK->modified(execution_space, datamask_modify);

  auto x = atomKK->k_x.template view<DeviceType>();
  auto f = atomKK->k_f.template view<DeviceType>();
  auto type = atomKK->k_type.template view<DeviceType>();

  if (comm->nprocs > 1)
    error->all(FLERR,
               "pair_style matgl/kk: multi-rank (MPI-decomposed) runs are "
               "not yet supported -- ghost atoms may be owned by a "
               "different rank, so there is no local row to fold periodic "
               "edges back onto. Run with a single MPI rank (`-k on g 1` "
               "single-GPU is the supported configuration).");

  const int inum = list->inum;
  const int nall = atom->nlocal + atom->nghost;
  const int nlocal = atom->nlocal;

  auto k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_ilist = k_list->d_ilist;
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;

  // 1) Resize per-atom buffers.
  if (nall > atom_capacity_) {
    atom_capacity_ = nall;
    d_atomic_numbers_ = Kokkos::View<int64_t *, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:Z"), atom_capacity_);
    d_local_or_ghost_ = Kokkos::View<bool *, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:mask"),
        atom_capacity_);
    d_local_row_of_ = Kokkos::View<int *, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:localrow"),
        atom_capacity_);
    d_numneigh_short_ = Kokkos::View<int *, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:nshort"),
        atom_capacity_);
    d_first_edge_ = Kokkos::View<int *, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:first"),
        atom_capacity_ + 1);
  }

  // Fill Z + mask from atom type.
  const auto type_to_z = d_type_to_z_;
  const auto d_atomic_numbers = d_atomic_numbers_;
  const auto d_local_or_ghost = d_local_or_ghost_;
  Kokkos::parallel_for(
      "matgl_kk:fill_atoms",
      Kokkos::RangePolicy<DeviceType>(0, nall),
      KOKKOS_LAMBDA(const int i) {
        const int t = type(i);
        d_atomic_numbers(i) = type_to_z(t);
        d_local_or_ghost(i) = (i < nlocal);
      });

  // local_row_of_(j) = the owned row representing the same physical atom as
  // row j (identity for local rows; the owned copy with the same tag for
  // ghosts). atom->map() is a host-only lookup, and `nall` is small (a few
  // thousand atoms) next to the GPU model forward pass below, so we just
  // build this on the host once per compute() call. See pair_matgl.cpp for
  // why every edge must resolve to one consistent row per atom.
  atomKK->sync(Host, TAG_MASK);
  auto h_local_row_of = Kokkos::create_mirror_view(d_local_row_of_);
  for (int j = 0; j < nall; ++j) {
    const int j_local = atom->map(atom->tag[j]);
    if (j_local < 0 || j_local >= nlocal)
      error->one(FLERR,
                 "pair_style matgl/kk: could not resolve atom {} (tag {}) "
                 "to a local row -- is `atom_modify map yes` set?",
                 j, atom->tag[j]);
    h_local_row_of(j) = j_local;
  }
  Kokkos::deep_copy(d_local_row_of_, h_local_row_of);

  // 2) Count short-cutoff neighbors per i (only inum atoms are listed,
  //    so initialize numneigh_short_ for ghost atoms to zero).
  Kokkos::deep_copy(d_numneigh_short_, 0);
  const double r_max_sq = r_max_squared_;
  const auto d_numneigh_short = d_numneigh_short_;

  Kokkos::parallel_for(
      "matgl_kk:count_neigh",
      Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        const double xi = x(i, 0);
        const double yi = x(i, 1);
        const double zi = x(i, 2);
        const int jnum = d_numneigh(i);
        int nshort = 0;
        for (int jj = 0; jj < jnum; ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          const double dx = x(j, 0) - xi;
          const double dy = x(j, 1) - yi;
          const double dz = x(j, 2) - zi;
          const double rsq = dx * dx + dy * dy + dz * dz;
          if (rsq <= r_max_sq) ++nshort;
        }
        d_numneigh_short(i) = nshort;
      });

  // 3) Exclusive prefix-sum into d_first_edge_ (length nall+1).
  const auto d_first_edge = d_first_edge_;
  Kokkos::parallel_scan(
      "matgl_kk:scan_edges",
      Kokkos::RangePolicy<DeviceType>(0, nall + 1),
      KOKKOS_LAMBDA(const int i, int &update, const bool final) {
        const int v = (i < nall) ? d_numneigh_short(i) : 0;
        if (final) d_first_edge(i) = update;
        update += v;
      });

  // Fetch total edge count.
  int total_edges = 0;
  Kokkos::deep_copy(total_edges, Kokkos::subview(d_first_edge_, nall));

  if (total_edges > edge_capacity_) {
    edge_capacity_ = total_edges;
    d_edge_index_ = Kokkos::View<int64_t **, Kokkos::LayoutRight, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:edges"), 2,
        edge_capacity_);
    d_unit_shifts_ = Kokkos::View<int64_t **, Kokkos::LayoutRight, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "matgl:shifts"),
        edge_capacity_, 3);
  }

  // 4) Fill edge_index + unit_shifts. Every edge must resolve to the local
  //    row of the atom it represents -- TensorNet's message-passing layers
  //    need one consistent row per physical atom (periodicity goes through
  //    unit_shifts, not through ghost-row duplication; see pair_matgl.cpp).
  const auto d_local_row_of = d_local_row_of_;
  const auto d_edge_index = d_edge_index_;
  const auto d_unit_shifts = d_unit_shifts_;
  const double *const h_inv_host = domain->h_inv;
  const double hinv0 = h_inv_host[0], hinv1 = h_inv_host[1], hinv2 = h_inv_host[2];
  const double hinv3 = h_inv_host[3], hinv4 = h_inv_host[4], hinv5 = h_inv_host[5];
  Kokkos::parallel_for(
      "matgl_kk:fill_edges",
      Kokkos::RangePolicy<DeviceType>(0, inum),
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        const double xi = x(i, 0);
        const double yi = x(i, 1);
        const double zi = x(i, 2);
        const int jnum = d_numneigh(i);
        int e = d_first_edge(i);
        for (int jj = 0; jj < jnum; ++jj) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          const double dx = x(j, 0) - xi;
          const double dy = x(j, 1) - yi;
          const double dz = x(j, 2) - zi;
          const double rsq = dx * dx + dy * dy + dz * dz;
          if (rsq > r_max_sq) continue;

          const int j_local = d_local_row_of(j);
          // Integer image vector (nx,ny,nz) such that
          //   x[j] == x[j_local] + (nx,ny,nz) @ cell
          // (boxlo cancels in the difference -- same affine map as
          // Domain::x2lamda, applied directly to x[j] - x[j_local]).
          const double ddx = x(j, 0) - x(j_local, 0);
          const double ddy = x(j, 1) - x(j_local, 1);
          const double ddz = x(j, 2) - x(j_local, 2);
          const double lz = hinv2 * ddz;
          const double ly = hinv1 * ddy + hinv3 * ddz;
          const double lx = hinv0 * ddx + hinv5 * ddy + hinv4 * ddz;

          d_edge_index(0, e) = i;
          d_edge_index(1, e) = j_local;
          d_unit_shifts(e, 0) = static_cast<int64_t>(Kokkos::round(lx));
          d_unit_shifts(e, 1) = static_cast<int64_t>(Kokkos::round(ly));
          d_unit_shifts(e, 2) = static_cast<int64_t>(Kokkos::round(lz));
          ++e;
        }
      });

  // 5) Build positions tensor on the same device as the model.
  //    LAMMPS' k_x is (nall,3) double; if the model is float32 we cast.
  const auto torch_real_opts = torch::TensorOptions().dtype(dtype_).device(torch_device_);
  const auto torch_long_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch_device_);
  const auto torch_bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch_device_);

  // x is (nall,3) double already on `DeviceType`. We make a libtorch view
  // through from_blob and cast to the model's dtype if needed.
  torch::Tensor positions_d = torch::from_blob(
      x.data(), {nall, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch_device_));
  torch::Tensor positions = (dtype_ == torch::kFloat64)
                                ? positions_d.clone()
                                : positions_d.to(dtype_);

  torch::Tensor edge_index = blob_from_view(d_edge_index_, torch_long_opts);
  // The Kokkos View is 2 x edge_capacity_ but we only filled 0..total_edges.
  edge_index = edge_index.narrow(/*dim=*/1, /*start=*/0, /*length=*/total_edges);

  torch::Tensor unit_shifts = blob_from_view(d_unit_shifts_, torch_long_opts);
  unit_shifts = unit_shifts.narrow(0, 0, total_edges);

  torch::Tensor atomic_numbers = blob_from_view(d_atomic_numbers_, torch_long_opts);
  atomic_numbers = atomic_numbers.narrow(0, 0, nall);
  torch::Tensor local_or_ghost = blob_from_view(d_local_or_ghost_, torch_bool_opts);
  local_or_ghost = local_or_ghost.narrow(0, 0, nall);

  // 6) Cell.
  torch::Tensor cell = torch::zeros({3, 3}, torch_real_opts);
  {
    auto host_opts = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);
    auto cell_h = torch::zeros({3, 3}, host_opts);
    if (dtype_ == torch::kFloat64) {
      auto c = cell_h.accessor<double, 2>();
      c[0][0] = domain->xprd;
      c[1][0] = domain->xy;   c[1][1] = domain->yprd;
      c[2][0] = domain->xz;   c[2][1] = domain->yz;   c[2][2] = domain->zprd;
    } else {
      auto c = cell_h.accessor<float, 2>();
      c[0][0] = static_cast<float>(domain->xprd);
      c[1][0] = static_cast<float>(domain->xy);
      c[1][1] = static_cast<float>(domain->yprd);
      c[2][0] = static_cast<float>(domain->xz);
      c[2][1] = static_cast<float>(domain->yz);
      c[2][2] = static_cast<float>(domain->zprd);
    }
    cell.copy_(cell_h, /*non_blocking=*/false);
  }

  // 7) Forward.
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
    error->all(FLERR, "pair_matgl/kk: model forward failed: {}", e.what());
  }
  auto out = result.toGenericDict();

  // 8) Energy + force scatter. LAMMPS f is double on device; the model may
  //    return float32 — promote on the fly.
  double total_energy = out.at("total_energy_local").toTensor().to(torch::kFloat64).item<double>();
  if (eflag_global) eng_vdwl += total_energy;

  torch::Tensor forces_t =
      out.at("forces").toTensor().to(torch::kFloat64).contiguous();

  // Wrap the force tensor as a Kokkos device-side unmanaged view and add
  // into LAMMPS' f.
  using UnmanagedF = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  UnmanagedF d_force_in(forces_t.data_ptr<double>(), nall, 3);

  Kokkos::parallel_for(
      "matgl_kk:add_forces",
      Kokkos::RangePolicy<DeviceType>(0, nall),
      KOKKOS_LAMBDA(const int i) {
        f(i, 0) += d_force_in(i, 0);
        f(i, 1) += d_force_in(i, 1);
        f(i, 2) += d_force_in(i, 2);
      });

  // 9) Virial — the model returns a small 3x3 tensor; pull to host.
  //    The model returns virials = dE/dstrain (strain_grad), but LAMMPS'
  //    virial[] convention is W = sum_i r_i (x) f_i = -dE/dstrain (matches
  //    pair_gnnp.cpp: virial -= volume * stress, with ASE stress = dE/dstrain / V).
  if (vflag_global) {
    auto vir_t = out.at("virials").toTensor().to(torch::kFloat64).cpu();
    auto va = vir_t.accessor<double, 2>();
    virial[0] -= va[0][0];
    virial[1] -= va[1][1];
    virial[2] -= va[2][2];
    virial[3] -= 0.5 * (va[0][1] + va[1][0]);
    virial[4] -= 0.5 * (va[0][2] + va[2][0]);
    virial[5] -= 0.5 * (va[1][2] + va[2][1]);
  }
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairMATGLKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMATGLKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS
