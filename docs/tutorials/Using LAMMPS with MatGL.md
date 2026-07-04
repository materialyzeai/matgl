---
layout: default
title: Using LAMMPS with MatGL.md
nav_exclude: true
---

# Introduction

This notebook demonstrates the use of the LAMMPS interface to MatGL developed by AdvancedSoft. To install, please clone LAMMPS from AdvancedSoft's Github repo. It is recommended that you use cmake, especially if you are on Apple Silicon Macs. Below are a sequence of instructions that worked for me. Modify as necessary.

```
git clone https://github.com/advancesoftcorp/lammps.git
cd lammps
git checkout based-on-lammps_2Aug2023
mkdir build
cd build
cmake -C ../cmake/presets/basic.cmake -D BUILD_SHARED_LIBS=on -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on -D PKG_ML-GNNP=on -D PKG_EXTRA-COMPUTE=on -D PKG_EXTRA-FIX=on -D PKG_MANYBODY=on -D PKG_EXTRA-DUMP=on -D PKG_MOLECULE=on ../cmake
cmake --build .
make install
```

After installation, your lmp executable should be in your `$HOME/.local/bin` directory. You will need to add that to your PATH if it is not already there. You may also need to adjust your library paths if needed.

Upon running this notebook, if you encounter an error about DFTD3 and you do not need dispersion corrections, I recommend you simply comment out the `from dftd3.ase import DFTD3` line in your `$HOME/.local/share/lammps/src/ML-GNNP/gnnp_driver.py` file. DFTD3 is a pain to install.


```python
from __future__ import annotations

import glob
import os
import subprocess  # noqa: F401  # used inside the %timeit magic line below
from datetime import timedelta

import pandas as pd
import seaborn as sns

n_cpus = 1

HOME_DIR = os.environ["HOME"]
```

We will first create our test structure - a simple MgO with 8 atoms.


```python
mgo_data = """

       8  atoms
       2  atom types

 0.00  4.19  xlo xhi
 0.00  4.19  ylo yhi
 0.00  4.19  zlo zhi

 Atoms

       1        1  0.0 0.0 0.0
       2        1  0.0 0.5 0.5
       3        1  0.5 0.0 0.5
       4        1  0.5 0.5 0.0
       5        1  0.0 0.0 0.5
       6        1  0.0 0.5 0.0
       7        1  0.5 0.0 0.0
       8        1  0.5 0.5 0.5

 Masses

       1  24.305 # Mg
       2  15.999 # O

"""

with open("dat.lammps", "w") as f:
    f.write(mgo_data)
```


```python
run_stats = []

for x in [2, 4, 8, 16, 32, 64, 128]:
    modified_script = f"""
units         metal
boundary      p p p
atom_style    atomic

# if you want to use CUDA GPU, please switch m3gnet to m3gnet/gpu in pair_style
pair_style    gnnp {HOME_DIR}/.local/share/lammps/src/ML-GNNP

read_data     ./dat.lammps
replicate     {x} 1 1
pair_coeff    * *  matgl M3GNet-PES-MatPES-PBE-2025.2  Mg O  # MatGL will be called

dump          myDump all custom 10 xyz.lammpstrj id element x y z
dump_modify   myDump sort id element Mg O

thermo_style  custom step time cpu pe ke etotal temp press vol density
thermo        10

velocity      all create 300.0 12345
fix           myEnse all npt temp 300.0 300.0 0.1 aniso 1.0 1.0 1.0
timestep      1.0e-3
run           100
"""

    outfile = f"MgO_{x}.out"
    # Write the modified script to a temporary file
    with open("lammps.in", "w") as f:
        f.write(modified_script)
    os.environ["OMP_NUM_THREADS"] = f"{n_cpus}"
    lammps_command = f"{HOME_DIR}/.local/bin/lmp < lammps.in > {outfile}"
    r = %timeit -n 1 -r 1 -o subprocess.run(lammps_command, shell=True)
    with open(outfile) as f:
        for line in f:
            if "Total wall time" in line:
                _, hours, minutes, seconds = line.split(":")
                walltime = timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds)).total_seconds()
    run_stats.append((x * 8, r.average / 100, walltime / 100))
```


```python
results = pd.DataFrame(run_stats, columns=["natoms", "run_time", "wall_time"])
```


```python
ax = sns.relplot(kind="scatter", data=results, x="natoms", y="wall_time", height=5, aspect=1.5)
_ = ax.set_xlabels("No. of Atoms")
_ = ax.set_ylabels("Wall time (s) per MD step.")
```

    /Users/shyue/miniconda3/envs/mavrl/lib/python3.9/site-packages/seaborn/axisgrid.py:118: UserWarning: The figure layout has changed to tight
      self._figure.tight_layout(*args, **kwargs)




![png](assets/Using%20LAMMPS%20with%20MatGL_6_1.png)




```python
# Perform some cleanup

os.remove("lammps.in")
for fn in glob.glob("MgO*.out") + glob.glob("*.lammps"):
    os.remove(fn)
```