# ML-MATGL Kokkos variant -- documentation / warning fragment.
#
# The Kokkos sources are picked up by LAMMPS' own KOKKOS package machinery
# (Packages/KOKKOS.cmake scans <lammps>/src/KOKKOS for *_kokkos.* styles and
# registers them via RegisterStylesExt), so this snippet adds NO sources:
#
#   cp /path/to/matgl/lammps/src/KOKKOS/pair_matgl_kokkos.* <lammps>/src/KOKKOS/
#
# and configure with -D PKG_ML-MATGL=ON -D PKG_KOKKOS=ON (plus the usual
# Kokkos CUDA flags). libtorch linkage comes from ML-MATGL.cmake.

if(NOT PKG_ML-MATGL OR NOT PKG_KOKKOS)
    return()
endif()

# Single-GPU only: warn loudly. MACE upstream issues #1294 and #322 cover
# the multi-rank-with-libtorch breakage we inherit.
message(STATUS
    "ML-MATGL-KOKKOS: enabled. Single-GPU runs only -- multi-rank Kokkos with "
    "libtorch is unreliable (see MACE issues #1294, #322).")
