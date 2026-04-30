# Description

This model is a M3GNet universal potential trained for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials. This is essentially a retrained version of the M3GNet universal potential originally
implemented in tensorflow. It should be noted that this is a preliminary version, which require further optimizations of
hyper-parameters for better accuracy. The accuracy is expected to be worse than the MP-2021.2.8 version since the structures
and corresponding energies, forces and stresses are more diverse. However, we expected that the extrapolation should be more reliable.

# Training dataset

MP-2021.2.8-DIRECT: Materials Project structure relaxations as of 2021.2.8 combined with DIRECT sampling.
- Training set size: 167192
- Validation set size: 9284

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 21.848 meV/atom, 0.049 eV/A, 0.257 GPa
- Validation: 33.547 meV/atom, 0.087 eV/A, 0.593 GPa

## Cubic crystals benchmark

This is the same benchmark used for the TF implementation. The error characteristics are consistent between this
new implementation and the old version. It should be noted that the large deviation of lattice constant in Li-BCC
is attributed to the poor initialization. If the initial lattice constant is set to less than 4.3 Å, it gets down
to 1 %

| Material    | Crystal structure   |   a (Å) |   MP a (Å) |   Predicted a (Å) | % error vs Expt   | % error vs MP   |
|:------------|:--------------------|--------:|-----------:|------------------:|:------------------|:----------------|
| Mo          | BCC                 | 3.142   |    3.16762 |           3.1674  | 0.808%            | -0.007%         |
| CdS         | Zinc blende (FCC)   | 5.832   |    5.88591 |           5.88501 | 0.909%            | -0.015%         |
| Au          | FCC                 | 4.065   |    4.17129 |           4.17062 | 2.598%            | -0.016%         |
| Th          | FCC                 | 5.08    |    5.04725 |           5.04823 | -0.625%           | 0.019%          |
| Al          | FCC                 | 4.046   |    4.03893 |           4.03703 | -0.222%           | -0.047%         |
| NbN         | Halite              | 4.392   |    4.45247 |           4.45549 | 1.446%            | 0.068%          |
| GaAs        | Zinc blende (FCC)   | 5.653   |    5.75018 |           5.75423 | 1.791%            | 0.070%          |
| Nb          | BCC                 | 3.3008  |    3.31763 |           3.32034 | 0.592%            | 0.081%          |
| RbI         | Halite              | 7.35    |    7.38233 |           7.38836 | 0.522%            | 0.082%          |
| TiC         | Halite              | 4.328   |    4.33144 |           4.33513 | 0.165%            | 0.085%          |
| BN          | Zinc blende (FCC)   | 3.615   |    3.626   |           3.6299  | 0.412%            | 0.108%          |
| VC0.97      | Halite              | 4.166   |    4.16195 |           4.15722 | -0.211%           | -0.113%         |
| Li          | BCC                 | 3.49    |    3.43931 |           3.44346 | -1.334%           | 0.120%          |
| ZrC0.97     | Halite              | 4.698   |    4.71287 |           4.71858 | 0.438%            | 0.121%          |
| KTaO3       | Cubic perovskite    | 3.9885  |    3.99488 |           4.00024 | 0.294%            | 0.134%          |
| RbCl        | Halite              | 6.59    |    6.61741 |           6.60661 | 0.252%            | -0.163%         |
| InSb        | Zinc blende (FCC)   | 6.479   |    6.63322 |           6.6465  | 2.585%            | 0.200%          |
| BP          | Zinc blende (FCC)   | 4.538   |    4.53214 |           4.5426  | 0.101%            | 0.231%          |
| ScN         | Halite              | 4.52    |    4.51094 |           4.5233  | 0.073%            | 0.274%          |
| Si          | Diamond (FCC)       | 5.43102 |    5.4437  |           5.45948 | 0.524%            | 0.290%          |
| TaC0.99     | Halite              | 4.456   |    4.4678  |           4.48137 | 0.569%            | 0.304%          |
| C (diamond) | Diamond (FCC)       | 3.567   |    3.56075 |           3.57236 | 0.150%            | 0.326%          |
| PbTe        | Halite (FCC)        | 6.462   |    6.54179 |           6.567   | 1.625%            | 0.385%          |
| TiN         | Halite              | 4.249   |    4.24125 |           4.25808 | 0.214%            | 0.397%          |
| Ta          | BCC                 | 3.3058  |    3.30986 |           3.32345 | 0.534%            | 0.411%          |
| Sr          | FCC                 | 6.08    |    6.06721 |           6.04087 | -0.644%           | -0.434%         |
| V           | BCC                 | 3.0399  |    2.9824  |           2.99554 | -1.459%           | 0.441%          |
| Eu          | BCC                 | 4.61    |    4.4876  |           4.46714 | -3.099%           | -0.456%         |
| RbF         | Halite              | 5.65    |    5.63228 |           5.65835 | 0.148%            | 0.463%          |
| Ir          | FCC                 | 3.84    |    3.85393 |           3.8758  | 0.932%            | 0.568%          |
| Fe          | BCC                 | 2.856   |    2.86304 |           2.84658 | -0.330%           | -0.575%         |
| Ca          | FCC                 | 5.58    |    5.57682 |           5.5438  | -0.649%           | -0.592%         |
| ZrN         | Halite              | 4.577   |    4.58853 |           4.61825 | 0.901%            | 0.648%          |
| HfC0.99     | Halite              | 4.64    |    4.63428 |           4.66432 | 0.524%            | 0.648%          |
| W           | BCC                 | 3.155   |    3.17032 |           3.19123 | 1.148%            | 0.660%          |
| Ac          | FCC                 | 5.31    |    5.69621 |           5.65697 | 6.534%            | -0.689%         |
| SrTiO3      | Cubic perovskite    | 3.98805 |    3.9127  |           3.94006 | -1.203%           | 0.699%          |
| VN          | Halite              | 4.136   |    4.12493 |           4.09218 | -1.059%           | -0.794%         |
| AlSb        | Zinc blende (FCC)   | 6.1355  |    6.18504 |           6.23425 | 1.610%            | 0.796%          |
| CrN         | Halite              | 4.149   |    4.19086 |           4.15581 | 0.164%            | -0.836%         |
| Pt          | FCC                 | 3.912   |    3.94315 |           3.97787 | 1.684%            | 0.881%          |
| LiF         | Halite              | 4.03    |    4.08343 |           4.04621 | 0.402%            | -0.911%         |
| InP         | Zinc blende (FCC)   | 5.869   |    5.90395 |           5.95826 | 1.521%            | 0.920%          |
| Pd          | FCC                 | 3.859   |    3.9173  |           3.95384 | 2.458%            | 0.933%          |
| Ni          | FCC                 | 3.499   |    3.47515 |           3.50847 | 0.271%            | 0.959%          |
| Pb          | FCC                 | 4.92    |    4.98951 |           5.03854 | 2.409%            | 0.983%          |
| CsI         | Caesium chloride    | 4.567   |    4.66521 |           4.7131  | 3.199%            | 1.027%          |
| AlAs        | Zinc blende (FCC)   | 5.6605  |    5.6758  |           5.73449 | 1.307%            | 1.034%          |
| CdTe        | Zinc blende (FCC)   | 6.482   |    6.56423 |           6.63255 | 2.323%            | 1.041%          |
| CaVO3       | Cubic perovskite    | 3.767   |    3.83041 |           3.8712  | 2.766%            | 1.065%          |
| Cu          | FCC                 | 3.597   |    3.57743 |           3.61612 | 0.532%            | 1.082%          |
| Ce          | FCC                 | 5.16    |    4.67243 |           4.72297 | -8.470%           | 1.082%          |
| Rh          | FCC                 | 3.8     |    3.80597 |           3.84876 | 1.283%            | 1.124%          |
| ZnS         | Zinc blende (FCC)   | 5.42    |    5.38737 |           5.44799 | 0.516%            | 1.125%          |
| GaP         | Zinc blende (FCC)   | 5.4505  |    5.45162 |           5.51446 | 1.173%            | 1.153%          |
| CdSe        | Zinc blende (FCC)   | 6.05    |    6.14054 |           6.21525 | 2.731%            | 1.217%          |
| EuTiO3      | Cubic perovskite    | 7.81    |    3.90388 |           3.95191 | -49.399%          | 1.230%          |
| Ge          | Diamond (FCC)       | 5.658   |    5.67485 |           5.74952 | 1.618%            | 1.316%          |
| ZnO         | Halite (FCC)        | 4.58    |    4.33888 |           4.39772 | -3.980%           | 1.356%          |
| GaSb        | Zinc blende (FCC)   | 6.0959  |    6.13721 |           6.22085 | 2.050%            | 1.363%          |
| SrVO3       | Cubic perovskite    | 3.838   |    3.90089 |           3.95497 | 3.048%            | 1.386%          |
| InAs        | Zinc blende (FCC)   | 6.0583  |    6.10712 |           6.19643 | 2.280%            | 1.462%          |
| HfN         | Halite              | 4.392   |    4.51172 |           4.57948 | 4.269%            | 1.502%          |
| Na          | BCC                 | 4.23    |    4.20805 |           4.14451 | -2.021%           | -1.510%         |
| Ag          | FCC                 | 4.079   |    4.10436 |           4.16927 | 2.213%            | 1.582%          |
| KI          | Halite              | 7.07    |    7.08487 |           7.19969 | 1.834%            | 1.621%          |
| Yb          | FCC                 | 5.49    |    5.38726 |           5.48158 | -0.153%           | 1.751%          |
| MgO         | Halite (FCC)        | 4.212   |    4.194   |           4.26745 | 1.317%            | 1.751%          |
| KBr         | Halite              | 6.6     |    6.58908 |           6.71154 | 1.690%            | 1.858%          |
| KCl         | Halite              | 6.29    |    6.28374 |           6.41075 | 1.920%            | 2.021%          |
| AlP         | Zinc blende (FCC)   | 5.451   |    5.47297 |           5.3591  | -1.686%           | -2.080%         |
| K           | BCC                 | 5.23    |    5.39512 |           5.2723  | 0.809%            | -2.277%         |
| Ne          | FCC                 | 4.43    |    4.19502 |           4.29091 | -3.140%           | 2.286%          |
| NaF         | Halite              | 4.63    |    4.57179 |           4.68341 | 1.154%            | 2.442%          |
| KF          | Halite              | 5.34    |    5.30883 |           5.44154 | 1.902%            | 2.500%          |
| NaBr        | Halite              | 5.97    |    5.92308 |           6.09528 | 2.098%            | 2.907%          |
| NaI         | Halite              | 6.47    |    6.43731 |           6.66885 | 3.073%            | 3.597%          |
| NaCl        | Halite              | 5.64    |    5.58813 |           5.80867 | 2.991%            | 3.947%          |
| Cr          | BCC                 | 2.88    |    2.9689  |           2.8498  | -1.049%           | -4.012%         |
| Cs          | BCC                 | 6.05    |    6.25693 |           6.00337 | -0.771%           | -4.052%         |
| CsF         | Halite              | 6.02    |    6.01228 |           6.3087  | 4.796%            | 4.930%          |
| Ar          | FCC                 | 5.26    |    5.36316 |           5.67732 | 7.934%            | 5.858%          |
| CsCl        | Caesium chloride    | 4.123   |    4.1437  |           4.39261 | 6.539%            | 6.007%          |
| LiCl        | Halite              | 5.14    |    5.08424 |           4.14286 | -19.400%          | -18.516%        |
| LiBr        | Halite              | 5.5     |    5.44538 |           4.25633 | -22.612%          | -21.836%        |


# References

```txt
Qi, J., Ko, T.W., Wood, B.C. et al. Robust training of machine learning interatomic potentials with dimensionality reduction and stratified sampling. npj Comput Mater 10, 43 (2024). https://doi.org/10.1038/s41524-024-01227-4
```
