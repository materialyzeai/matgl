/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   pair_matgl/kk: Kokkos variant of pair_matgl.

   The model body still runs inside LibTorch (CUDA/HIP) — Kokkos here is the
   glue that builds the edge / position / type tensors on the GPU and
   accumulates forces back into LAMMPS' Kokkos-managed views without going
   through host memory.

   See ACEsuit/lammps:src/KOKKOS/pair_mace_kokkos.{cpp,h} for the reference
   implementation we mirror.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(matgl/kk, PairMATGLKokkos<LMPDeviceType>);
PairStyle(matgl/kk/device, PairMATGLKokkos<LMPDeviceType>);
PairStyle(matgl/kk/host, PairMATGLKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_MATGL_KOKKOS_H
#define LMP_PAIR_MATGL_KOKKOS_H

#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "pair_kokkos.h"
#include "pair_matgl.h"

#include <torch/script.h>
#include <torch/torch.h>

namespace LAMMPS_NS {

template <class DeviceType>
class PairMATGLKokkos : public PairMATGL {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;

  PairMATGLKokkos(class LAMMPS *);
  ~PairMATGLKokkos() override;

  void compute(int, int) override;
  void init_style() override;

 protected:
  // Pinned to the CUDA device the libtorch model is on. Picked once at
  // init_style() from `lmp -k on g 1`.
  torch::Device torch_device_ = torch::kCPU;

  // Persistent device-side buffers — re-allocated on size change.
  // NOTE: explicit LayoutRight is required here. Without it, DeviceType's
  // default array_layout (LayoutLeft on Cuda, LayoutRight on host) leaves
  // the on-device memory column-major, while blob_from_view()/from_blob()
  // below always reinterpret the raw buffer as row-major. On the CUDA
  // backend that mismatch scrambled edge_index into a bogus graph topology
  // (wrong energy, zero/NaN forces); the host backend happened to match by
  // coincidence since its default layout is already LayoutRight.
  Kokkos::View<int64_t **, Kokkos::LayoutRight, DeviceType> d_edge_index_;        // (2, E)
  Kokkos::View<int64_t **, Kokkos::LayoutRight, DeviceType> d_unit_shifts_;       // (E, 3)
  Kokkos::View<int64_t *, DeviceType> d_atomic_numbers_;     // (N,)
  Kokkos::View<bool *, DeviceType> d_local_or_ghost_;        // (N,)
  // local_row_of_(j) = the local (owned, <nlocal) row representing the same
  // physical atom as row j. Identity for local rows; for ghosts, the owned
  // copy with the same tag. Built host-side once per compute() (cheap vs.
  // the model forward pass) and uploaded -- see pair_matgl.cpp for why this
  // exists (TensorNet needs one consistent row per atom; periodicity goes
  // through unit_shifts, not through ghost-row duplication).
  Kokkos::View<int *, DeviceType> d_local_row_of_;           // (N,)

  // Edge-counting scratch.
  Kokkos::View<int *, DeviceType> d_numneigh_short_;
  Kokkos::View<int *, DeviceType> d_first_edge_;

  // For converting LAMMPS atom-type (1..ntypes) -> Z on device.
  Kokkos::View<int64_t *, DeviceType> d_type_to_z_;

  // Capacity tracking so we only resize on growth.
  int64_t edge_capacity_ = 0;
  int64_t atom_capacity_ = 0;
};

}  // namespace LAMMPS_NS

#endif
#endif
