# Description

This model is a M3GNet universal potential for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials. This is essentially a retrained version of the M3GNet universal potential originally
implemented in tensorflow. It should be noted that the structures with high forces (> 100 eV/A) and no bond are removed
in the dataset.

# Training dataset

MP-2021.2.8: Materials Project structure relaxations as of 2021.2.8.
- Training set size: 167378
- Validation set size: 9294

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 18.697 meV/atom, 0.063 eV/A, 0.259 GPa
- Validation: 21.099 meV/atom, 0.072 eV/A, 0.380 GPa
## Cubic crystals benchmark

This is the same benchmark used for the TF implementation. The error characteristics are consistent between this
new implementation and the old version.

| Material    | Crystal structure   |   a (Å) |   MP a (Å) |   Predicted a (Å) | % error vs Expt   | % error vs MP   |
|:------------|:--------------------|--------:|-----------:|------------------:|:------------------|:----------------|
| Th          | FCC                 | 5.08    |    5.04725 |           5.04712 | -0.647%           | -0.003%         |
| GaAs        | Zinc blende (FCC)   | 5.653   |    5.75018 |           5.75044 | 1.724%            | 0.004%          |
| BN          | Zinc blende (FCC)   | 3.615   |    3.626   |           3.62638 | 0.315%            | 0.010%          |
| Au          | FCC                 | 4.065   |    4.17129 |           4.17198 | 2.632%            | 0.017%          |
| SrVO3       | Cubic perovskite    | 3.838   |    3.90089 |           3.90005 | 1.617%            | -0.022%         |
| InSb        | Zinc blende (FCC)   | 6.479   |    6.63322 |           6.63599 | 2.423%            | 0.042%          |
| CaVO3       | Cubic perovskite    | 3.767   |    3.83041 |           3.83438 | 1.789%            | 0.104%          |
| NbN         | Halite              | 4.392   |    4.45247 |           4.44784 | 1.271%            | -0.104%         |
| Mo          | BCC                 | 3.142   |    3.16762 |           3.16398 | 0.700%            | -0.115%         |
| VN          | Halite              | 4.136   |    4.12493 |           4.11997 | -0.387%           | -0.120%         |
| Na          | BCC                 | 4.23    |    4.20805 |           4.21325 | -0.396%           | 0.124%          |
| Nb          | BCC                 | 3.3008  |    3.31763 |           3.32285 | 0.668%            | 0.157%          |
| CsI         | Caesium chloride    | 4.567   |    4.66521 |           4.6727  | 2.314%            | 0.160%          |
| VC0.97      | Halite              | 4.166   |    4.16195 |           4.15511 | -0.261%           | -0.164%         |
| Sr          | FCC                 | 6.08    |    6.06721 |           6.07763 | -0.039%           | 0.172%          |
| Li          | BCC                 | 3.49    |    3.43931 |           3.43232 | -1.653%           | -0.203%         |
| Ca          | FCC                 | 5.58    |    5.57682 |           5.56547 | -0.260%           | -0.203%         |
| HfN         | Halite              | 4.392   |    4.51172 |           4.5209  | 2.935%            | 0.204%          |
| Si          | Diamond (FCC)       | 5.43102 |    5.4437  |           5.45671 | 0.473%            | 0.239%          |
| V           | BCC                 | 3.0399  |    2.9824  |           2.9904  | -1.628%           | 0.268%          |
| HfC0.99     | Halite              | 4.64    |    4.63428 |           4.64672 | 0.145%            | 0.269%          |
| ZnO         | Halite (FCC)        | 4.58    |    4.33888 |           4.35142 | -4.991%           | 0.289%          |
| Al          | FCC                 | 4.046   |    4.03893 |           4.05227 | 0.155%            | 0.330%          |
| ScN         | Halite              | 4.52    |    4.51094 |           4.52712 | 0.158%            | 0.359%          |
| Ta          | BCC                 | 3.3058  |    3.30986 |           3.32178 | 0.483%            | 0.360%          |
| TiN         | Halite              | 4.249   |    4.24125 |           4.2567  | 0.181%            | 0.364%          |
| W           | BCC                 | 3.155   |    3.17032 |           3.18242 | 0.869%            | 0.382%          |
| ZrC0.97     | Halite              | 4.698   |    4.71287 |           4.73135 | 0.710%            | 0.392%          |
| Fe          | BCC                 | 2.856   |    2.86304 |           2.85153 | -0.157%           | -0.402%         |
| TaC0.99     | Halite              | 4.456   |    4.4678  |           4.48713 | 0.699%            | 0.433%          |
| C (diamond) | Diamond (FCC)       | 3.567   |    3.56075 |           3.57685 | 0.276%            | 0.452%          |
| Ac          | FCC                 | 5.31    |    5.69621 |           5.66835 | 6.749%            | -0.489%         |
| TiC         | Halite              | 4.328   |    4.33144 |           4.35304 | 0.579%            | 0.499%          |
| ZrN         | Halite              | 4.577   |    4.58853 |           4.61466 | 0.823%            | 0.570%          |
| PbTe        | Halite (FCC)        | 6.462   |    6.54179 |           6.57932 | 1.816%            | 0.574%          |
| BP          | Zinc blende (FCC)   | 4.538   |    4.53214 |           4.56013 | 0.488%            | 0.618%          |
| EuTiO3      | Cubic perovskite    | 7.81    |    3.90388 |           3.93011 | -49.678%          | 0.672%          |
| Ir          | FCC                 | 3.84    |    3.85393 |           3.88262 | 1.110%            | 0.744%          |
| Pt          | FCC                 | 3.912   |    3.94315 |           3.97353 | 1.573%            | 0.771%          |
| AlP         | Zinc blende (FCC)   | 5.451   |    5.47297 |           5.51519 | 1.178%            | 0.771%          |
| KTaO3       | Cubic perovskite    | 3.9885  |    3.99488 |           4.02688 | 0.962%            | 0.801%          |
| SrTiO3      | Cubic perovskite    | 3.98805 |    3.9127  |           3.94503 | -1.079%           | 0.826%          |
| Ni          | FCC                 | 3.499   |    3.47515 |           3.50407 | 0.145%            | 0.832%          |
| AlAs        | Zinc blende (FCC)   | 5.6605  |    5.6758  |           5.72403 | 1.122%            | 0.850%          |
| KBr         | Halite              | 6.6     |    6.58908 |           6.64632 | 0.702%            | 0.869%          |
| InP         | Zinc blende (FCC)   | 5.869   |    5.90395 |           5.95562 | 1.476%            | 0.875%          |
| LiI         | Halite              | 6.01    |    5.96835 |           6.02271 | 0.211%            | 0.911%          |
| CdS         | Zinc blende (FCC)   | 5.832   |    5.88591 |           5.94074 | 1.864%            | 0.932%          |
| Ce          | FCC                 | 5.16    |    4.67243 |           4.72008 | -8.526%           | 1.020%          |
| RbI         | Halite              | 7.35    |    7.38233 |           7.45884 | 1.481%            | 1.036%          |
| GaP         | Zinc blende (FCC)   | 5.4505  |    5.45162 |           5.50968 | 1.086%            | 1.065%          |
| CdTe        | Zinc blende (FCC)   | 6.482   |    6.56423 |           6.63419 | 2.348%            | 1.066%          |
| AlSb        | Zinc blende (FCC)   | 6.1355  |    6.18504 |           6.25173 | 1.894%            | 1.078%          |
| Pd          | FCC                 | 3.859   |    3.9173  |           3.96018 | 2.622%            | 1.095%          |
| Cu          | FCC                 | 3.597   |    3.57743 |           3.61748 | 0.569%            | 1.119%          |
| Pb          | FCC                 | 4.92    |    4.98951 |           5.04687 | 2.579%            | 1.150%          |
| CdSe        | Zinc blende (FCC)   | 6.05    |    6.14054 |           6.2112  | 2.664%            | 1.151%          |
| CrN         | Halite              | 4.149   |    4.19086 |           4.14236 | -0.160%           | -1.157%         |
| ZnS         | Zinc blende (FCC)   | 5.42    |    5.38737 |           5.45094 | 0.571%            | 1.180%          |
| Rh          | FCC                 | 3.8     |    3.80597 |           3.85381 | 1.416%            | 1.257%          |
| Ge          | Diamond (FCC)       | 5.658   |    5.67485 |           5.74924 | 1.613%            | 1.311%          |
| KI          | Halite              | 7.07    |    7.08487 |           7.17857 | 1.536%            | 1.322%          |
| Yb          | FCC                 | 5.49    |    5.38726 |           5.45903 | -0.564%           | 1.332%          |
| NaBr        | Halite              | 5.97    |    5.92308 |           6.00257 | 0.545%            | 1.342%          |
| Ag          | FCC                 | 4.079   |    4.10436 |           4.15989 | 1.983%            | 1.353%          |
| CsCl        | Caesium chloride    | 4.123   |    4.1437  |           4.20103 | 1.893%            | 1.384%          |
| NaI         | Halite              | 6.47    |    6.43731 |           6.53014 | 0.929%            | 1.442%          |
| K           | BCC                 | 5.23    |    5.39512 |           5.31521 | 1.629%            | -1.481%         |
| KCl         | Halite              | 6.29    |    6.28374 |           6.38015 | 1.433%            | 1.534%          |
| GaSb        | Zinc blende (FCC)   | 6.0959  |    6.13721 |           6.23157 | 2.226%            | 1.538%          |
| InAs        | Zinc blende (FCC)   | 6.0583  |    6.10712 |           6.20284 | 2.386%            | 1.567%          |
| MgO         | Halite (FCC)        | 4.212   |    4.194   |           4.26133 | 1.171%            | 1.605%          |
| RbCl        | Halite              | 6.59    |    6.61741 |           6.72733 | 2.084%            | 1.661%          |
| RbF         | Halite              | 5.65    |    5.63228 |           5.72925 | 1.403%            | 1.722%          |
| Eu          | BCC                 | 4.61    |    4.4876  |           4.56876 | -0.895%           | 1.808%          |
| NaCl        | Halite              | 5.64    |    5.58813 |           5.69132 | 0.910%            | 1.847%          |
| Ne          | FCC                 | 4.43    |    4.19502 |           4.2771  | -3.451%           | 1.957%          |
| CsF         | Halite              | 6.02    |    6.01228 |           6.13042 | 1.834%            | 1.965%          |
| KF          | Halite              | 5.34    |    5.30883 |           5.41705 | 1.443%            | 2.038%          |
| RbBr        | Halite              | 6.89    |    6.9119  |           7.05604 | 2.410%            | 2.085%          |
| NaF         | Halite              | 4.63    |    4.57179 |           4.6819  | 1.121%            | 2.408%          |
| Cr          | BCC                 | 2.88    |    2.9689  |           2.84808 | -1.108%           | -4.070%         |
| Cs          | BCC                 | 6.05    |    6.25693 |           5.98735 | -1.035%           | -4.308%         |
| Ar          | FCC                 | 5.26    |    5.36316 |           5.66902 | 7.776%            | 5.703%          |

# References

```txt
Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.
```
