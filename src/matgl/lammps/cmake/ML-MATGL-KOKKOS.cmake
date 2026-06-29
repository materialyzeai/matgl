# ML-MATGL Kokkos variant — drop-in CMake snippet.
#
# Layered on top of ML-MATGL.cmake: include() this *after* the base snippet,
# OR set PKG_ML-MATGL=ON and PKG_KOKKOS=ON together.
#
# Usage (from a stock LAMMPS source tree):
#   cmake -B build \
#       -D PKG_ML-MATGL=ON -D PKG_KOKKOS=ON \
#       -D Kokkos_ENABLE_CUDA=ON \
#       -D Kokkos_ARCH_AMPERE80=ON \
#       -D CMAKE_PREFIX_PATH=/path/to/libtorch \
#       -D CMAKE_CXX_COMPILER=$LAMMPS/lib/kokkos/bin/nvcc_wrapper \
#       <other flags>
#
# The `pair_matgl/kk` style is registered via the standard LAMMPS Kokkos
# pair-style macro so users invoke it with `pair_style matgl/kk` or by
# launching LAMMPS with `-sf kk -k on g 1`.

if(NOT PKG_ML-MATGL OR NOT PKG_KOKKOS)
    return()
endif()

if(NOT DEFINED ML_MATGL_KOKKOS_DIR)
    get_filename_component(ML_MATGL_KOKKOS_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../src/KOKKOS" ABSOLUTE)
endif()

if(NOT EXISTS "${ML_MATGL_KOKKOS_DIR}/pair_matgl_kokkos.cpp")
    message(FATAL_ERROR
        "ML-MATGL-KOKKOS source not found at ${ML_MATGL_KOKKOS_DIR}. "
        "Set -DML_MATGL_KOKKOS_DIR=<path/to/lammps/src/KOKKOS>.")
endif()

file(GLOB ML_MATGL_KOKKOS_SOURCES "${ML_MATGL_KOKKOS_DIR}/*.cpp")

target_sources(lammps PRIVATE ${ML_MATGL_KOKKOS_SOURCES})
target_include_directories(lammps PRIVATE ${ML_MATGL_KOKKOS_DIR})

# Single-GPU only: warn loudly. MACE upstream issues #1294 and #322 cover
# the multi-rank-with-libtorch breakage we inherit.
message(STATUS
    "ML-MATGL-KOKKOS: enabled. Single-GPU runs only — multi-rank Kokkos with "
    "libtorch is unreliable (see MACE issues #1294, #322).")
