#!/usr/bin/env python3
"""Wire the ML-MATGL package into a clean LAMMPS source tree.

Usage:
    patch_lammps.py /path/to/lammps /path/to/matgl

Run once, right before `cmake -B build -S /path/to/lammps/cmake ...`. It
inserts

    include(<matgl>/lammps/cmake/ML-MATGL.cmake)
    include(<matgl>/lammps/cmake/ML-MATGL-KOKKOS.cmake)

into <lammps>/cmake/CMakeLists.txt immediately after the accelerator-package
loop (`foreach(PKG_WITH_INCL ... KOKKOS OPT INTEL GPU) ... endforeach()`),
i.e. before `GenerateStyleHeaders(...)`. The snippets call RegisterStyles(),
which must run before LAMMPS generates style_pair.h or `pair_style matgl` /
`matgl/kk` are never registered ("Unrecognized pair style" at run time).

The LAMMPS tree must be unmodified: the script refuses to run if
CMakeLists.txt already mentions ML-MATGL. Nothing is copied into
<lammps>/src; the snippets compile the sources out of the matgl checkout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ACCEL_LOOP = re.compile(r"^foreach\(PKG_WITH_INCL\b.*\bKOKKOS\b")


def main() -> None:
    """Patch <lammps>/cmake/CMakeLists.txt or exit with an error."""
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    lammps, matgl = (Path(p).expanduser().resolve() for p in sys.argv[1:])

    cmakelists = lammps / "cmake" / "CMakeLists.txt"
    if not cmakelists.is_file():
        sys.exit(f"error: {cmakelists} not found; is {lammps} a LAMMPS source tree?")
    snippets = [matgl / "lammps" / "cmake" / f for f in ("ML-MATGL.cmake", "ML-MATGL-KOKKOS.cmake")]
    for snippet in snippets:
        if not snippet.is_file():
            sys.exit(f"error: {snippet} not found; is {matgl} a matgl checkout?")

    lines = cmakelists.read_text().splitlines(keepends=True)
    if any("ML-MATGL" in line for line in lines):
        sys.exit(
            f"error: {cmakelists} already mentions ML-MATGL. This script expects a clean LAMMPS "
            "source tree; restore the original CMakeLists.txt (e.g. `git checkout cmake/CMakeLists.txt`) "
            "and rerun."
        )

    start = next((i for i, line in enumerate(lines) if ACCEL_LOOP.match(line)), None)
    if start is None:
        sys.exit(f"error: accelerator-package foreach(PKG_WITH_INCL ...) loop not found in {cmakelists}")
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("endforeach()"))
    if not any(line.startswith("GenerateStyleHeaders(") for line in lines[end:]):
        sys.exit(f"error: GenerateStyleHeaders() not found after the accelerator loop in {cmakelists}")

    block = ["\n", "# ML-MATGL (added by patch_lammps.py; must precede GenerateStyleHeaders)\n"]
    block += [f"include({snippet.as_posix()})\n" for snippet in snippets]
    lines[end + 1 : end + 1] = block
    cmakelists.write_text("".join(lines))
    print(f"patched {cmakelists}")


if __name__ == "__main__":
    main()
