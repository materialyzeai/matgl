/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   pair_matgl: serial / OpenMP pair style backed by a TorchScripted MatGL
   ``LAMMPSMatGLModel`` (see matgl/src/matgl/ext/_lammps.py). The Python
   side ships a ``mgl create-lammps-model`` CLI that produces the .pt file
   this pair style consumes.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(matgl, PairMATGL);
// clang-format on
#else

#ifndef LMP_PAIR_MATGL_H
#define LMP_PAIR_MATGL_H

#include "pair.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairMATGL : public Pair {
 public:
  PairMATGL(class LAMMPS *);
  ~PairMATGL() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  void allocate();

  // The TorchScript model produced by `mgl create-lammps-model`.
  torch::jit::Module model_;

  // Cutoff radius baked into the model (read from the `r_max` buffer).
  double r_max_ = 0.0;
  double r_max_squared_ = 0.0;

  // Tensor dtype the model was exported with: torch::kFloat32 or torch::kFloat64.
  torch::Dtype dtype_ = torch::kFloat32;

  // Atomic numbers per LAMMPS atom-type (1-based; index 0 unused). Built from
  // the species names in the pair_coeff line.
  std::vector<int64_t> type_to_z_;

  // Whether the user disabled MPI domain decomposition with the optional
  // ``no_domain_decomposition`` keyword on the pair_style line. Tracks the
  // MACE-LAMMPS flag of the same name; reserved for future use.
  bool no_domain_decomposition_ = false;
};

}  // namespace LAMMPS_NS

#endif
#endif
