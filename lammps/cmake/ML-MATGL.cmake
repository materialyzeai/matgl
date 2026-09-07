# ML-MATGL package — drop-in CMake snippet for a stock LAMMPS source tree.
#
# Usage (from a stock LAMMPS source tree):
#   1. Copy or symlink lammps/src/ML-MATGL  →  <lammps>/src/ML-MATGL
#   2. Add to <lammps>/cmake/CMakeLists.txt, *before* the
#      `GenerateStyleHeaders(${LAMMPS_STYLE_HEADERS_DIR})` call (right after
#      the accelerator-package `foreach(PKG_WITH_INCL ... KOKKOS OPT INTEL GPU)`
#      loop is a good spot). Do NOT append it at the end of the file: this
#      snippet calls RegisterStyles(), which must run before the style
#      headers are generated or `pair_style matgl` is never registered.
#        include(/path/to/matgl/lammps/cmake/ML-MATGL.cmake)
#   3. Configure with:
#        cmake -B build \
#            -D PKG_ML-MATGL=ON \
#            -D CMAKE_PREFIX_PATH=/path/to/libtorch \
#            -D CMAKE_BUILD_TYPE=Release \
#            <other flags>
#
# CMake variables consumed:
#   PKG_ML-MATGL           - turn the package on/off (default OFF).
#   CMAKE_PREFIX_PATH      - must point at a libtorch install (CXX11 ABI build).
#   ML_MATGL_DIR           - override path to lammps/src/ML-MATGL (defaults to
#                            ${CMAKE_CURRENT_LIST_DIR}/../src/ML-MATGL).
#   MATGL_PYTHON_EXECUTABLE - interpreter pair_matgl shells out to when it has
#                            to export a raw matgl checkpoint to TorchScript
#                            (default "python3", i.e. whatever is on $PATH at
#                            run time). Overridable at run time without a
#                            rebuild via the MATGL_PYTHON environment variable;
#                            MATGL_EXPORT_SCRIPT likewise overrides the script.

option(PKG_ML-MATGL "Build the matgl pair_style backed by libtorch" OFF)

if(NOT PKG_ML-MATGL)
    return()
endif()

# Locate the source directory.
if(NOT DEFINED ML_MATGL_DIR)
    get_filename_component(ML_MATGL_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../src/ML-MATGL" ABSOLUTE)
endif()

if(NOT EXISTS "${ML_MATGL_DIR}/pair_matgl.cpp")
    message(FATAL_ERROR
        "ML-MATGL source not found at ${ML_MATGL_DIR}. "
        "Set -DML_MATGL_DIR=<path/to/lammps/src/ML-MATGL>.")
endif()

# Pull in libtorch.
find_package(Torch REQUIRED)
if(NOT TORCH_LIBRARIES)
    message(FATAL_ERROR
        "find_package(Torch) succeeded but TORCH_LIBRARIES is empty. "
        "Did you set CMAKE_PREFIX_PATH to a libtorch install?")
endif()

# Compose the source list.
file(GLOB ML_MATGL_SOURCES "${ML_MATGL_DIR}/*.cpp")

# Register the PairStyle(...) macro in pair_matgl.h with LAMMPS' style
# factory. LAMMPS only scans src/ and the *standard* package directories on
# its own, so a package living outside the LAMMPS tree has to do this
# explicitly. Must run before GenerateStyleHeaders() in CMakeLists.txt.
RegisterStyles(${ML_MATGL_DIR})

# Tell pair_matgl.cpp where to find the interpreter + export script it uses to
# convert a matgl IOMixIn checkpoint into TorchScript on first use.
if(NOT DEFINED MATGL_PYTHON_EXECUTABLE)
    set(MATGL_PYTHON_EXECUTABLE "python3")
endif()
target_compile_definitions(lammps PRIVATE
    MATGL_PYTHON_EXECUTABLE="${MATGL_PYTHON_EXECUTABLE}"
    MATGL_EXPORT_SCRIPT="${ML_MATGL_DIR}/export_matgl_checkpoint.py")

# Hook into the LAMMPS build. This file is included from
# <lammps>/cmake/CMakeLists.txt; the `lammps` target already exists by then.
target_sources(lammps PRIVATE ${ML_MATGL_SOURCES})
target_include_directories(lammps PRIVATE ${ML_MATGL_DIR})
target_compile_features(lammps PRIVATE cxx_std_17)
target_link_libraries(lammps PRIVATE ${TORCH_LIBRARIES})

# Make sure libtorch's headers come ahead of any system Eigen/torch shims.
target_include_directories(lammps PRIVATE ${TORCH_INCLUDE_DIRS})

# LibTorch ships with -D_GLIBCXX_USE_CXX11_ABI=…; propagate it so consumers
# (e.g. KOKKOS in Phase 3) see the same ABI.
if(DEFINED TORCH_CXX_FLAGS)
    set_property(TARGET lammps APPEND_STRING PROPERTY COMPILE_FLAGS " ${TORCH_CXX_FLAGS}")
endif()

message(STATUS "ML-MATGL: enabled, sources from ${ML_MATGL_DIR}")
message(STATUS "ML-MATGL: checkpoint export uses ${MATGL_PYTHON_EXECUTABLE} (override with MATGL_PYTHON at run time)")
message(STATUS "ML-MATGL: linking against TORCH_LIBRARIES=${TORCH_LIBRARIES}")
