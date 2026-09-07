# ML-MATGL Kokkos variant — drop-in CMake snippet.
#
# Layered on top of ML-MATGL.cmake: include() this *after* the base snippet,
# and set PKG_ML-MATGL=ON and PKG_KOKKOS=ON together.
#
# Like ML-MATGL.cmake, this must be included from <lammps>/cmake/CMakeLists.txt
# *before* the `GenerateStyleHeaders(${LAMMPS_STYLE_HEADERS_DIR})` call (right
# after the accelerator-package `foreach(PKG_WITH_INCL ... KOKKOS ...)` loop).
# It calls RegisterStyles(); appended at the end of the file it runs too late
# and `pair_style matgl/kk` is never registered.
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
#
# CMake variables consumed:
#   ML_MATGL_KOKKOS_DIR    - override path to lammps/src/KOKKOS (defaults to
#                            ${CMAKE_CURRENT_LIST_DIR}/../src/KOKKOS).
#   CMAKE_PREFIX_PATH      - must point at a libtorch install (CXX11 ABI build).

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

# Pull in libtorch ourselves rather than relying on ML-MATGL.cmake having
# already left TORCH_LIBRARIES / TORCH_INCLUDE_DIRS behind as a side effect.
find_package(Torch REQUIRED)
if(NOT TORCH_LIBRARIES)
    message(FATAL_ERROR
        "find_package(Torch) succeeded but TORCH_LIBRARIES is empty. "
        "Did you set CMAKE_PREFIX_PATH to a libtorch install?")
endif()

# Only our own source; a bare *.cpp glob would also sweep up anything else
# placed in this directory.
file(GLOB ML_MATGL_KOKKOS_SOURCES "${ML_MATGL_KOKKOS_DIR}/pair_matgl_kokkos.cpp")

# Register PairStyle(matgl/kk, ...) & friends from pair_matgl_kokkos.h with
# LAMMPS' style factory. Must run before GenerateStyleHeaders().
RegisterStyles(${ML_MATGL_KOKKOS_DIR})

target_sources(lammps PRIVATE ${ML_MATGL_KOKKOS_SOURCES})
target_include_directories(lammps PRIVATE ${ML_MATGL_KOKKOS_DIR})
target_include_directories(lammps PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_features(lammps PRIVATE cxx_std_17)
target_link_libraries(lammps PRIVATE ${TORCH_LIBRARIES})
if(DEFINED TORCH_CXX_FLAGS)
    set_property(TARGET lammps APPEND_STRING PROPERTY COMPILE_FLAGS " ${TORCH_CXX_FLAGS}")
endif()

# Single-GPU only: warn loudly. MACE upstream issues #1294 and #322 cover
# the multi-rank-with-libtorch breakage we inherit.
message(STATUS
    "ML-MATGL-KOKKOS: enabled, sources from ${ML_MATGL_KOKKOS_DIR}. "
    "Single-GPU runs only — multi-rank Kokkos with libtorch is unreliable "
    "(see MACE issues #1294, #322).")
