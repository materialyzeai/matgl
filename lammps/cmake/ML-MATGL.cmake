# ML-MATGL package -- libtorch link fragment for a stock LAMMPS source tree.
#
# This snippet ONLY locates libtorch and links it into the `lammps` target.
# It must NOT add the pair-style sources: LAMMPS registers styles by scanning
# package directories and generating style_pair.h roughly two thirds of the
# way through cmake/CMakeLists.txt, so sources added from an appended
# include() are compiled but never registered -- and adding them here while
# the package loop also adds them compiles every file twice and breaks the
# link. See lammps/README.md, "Build LAMMPS with the package":
#
#   1. Copy or symlink lammps/src/ML-MATGL  ->  <lammps>/src/ML-MATGL
#   2. Add ML-MATGL to set(STANDARD_PACKAGES ...) in
#      <lammps>/cmake/CMakeLists.txt, so the per-package loop does
#      RegisterStyles + target_sources at the right point.
#   3. Append to <lammps>/cmake/CMakeLists.txt:
#        include(/path/to/matgl/lammps/cmake/ML-MATGL.cmake)
#   4. Configure with -D PKG_ML-MATGL=ON and CMAKE_PREFIX_PATH at libtorch.

option(PKG_ML-MATGL "Build the matgl pair_style backed by libtorch" OFF)

if(NOT PKG_ML-MATGL)
    return()
endif()

# Pull in libtorch.
find_package(Torch REQUIRED)
if(NOT TORCH_LIBRARIES)
    message(FATAL_ERROR
        "find_package(Torch) succeeded but TORCH_LIBRARIES is empty. "
        "Did you set CMAKE_PREFIX_PATH to a libtorch install?")
endif()

target_compile_features(lammps PRIVATE cxx_std_17)
target_link_libraries(lammps PRIVATE ${TORCH_LIBRARIES})

# Make sure libtorch's headers come ahead of any system Eigen/torch shims.
target_include_directories(lammps PRIVATE ${TORCH_INCLUDE_DIRS})

# LibTorch ships with -D_GLIBCXX_USE_CXX11_ABI=...; propagate it so every
# consumer (including the KOKKOS sources) sees the same ABI.
if(DEFINED TORCH_CXX_FLAGS)
    set_property(TARGET lammps APPEND_STRING PROPERTY COMPILE_FLAGS " ${TORCH_CXX_FLAGS}")
endif()

message(STATUS "ML-MATGL: libtorch linked, TORCH_LIBRARIES=${TORCH_LIBRARIES}")
