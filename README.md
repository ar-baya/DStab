# DStab: estimating clustering quality by distance stability
[read paper here](https://doi.org/10.1007/s10044-023-01175-7)

---

Folder DStab/dist contains packages, which can be
- installed from within DStab/dist by: python3 -m pip install ./DStab-0.0.1-py3-none-any.whl
- uninstalled from within DStab/dist by: python3 -m pip uninstall ./DStab-0.0.1-py3-none-any.whl

---
Folder DStab/examples_data/ contains examples and data.

![Fig.1 - R15 Linear Sum](https://github.com/ar-baya/DStab/blob/main/examples_data/R15_res_tmp.png "R15 Linear Sum")

![Fig.2 - D31 Linear Sum](https://github.com/ar-baya/DStab/blob/main/examples_data/R15_res_tmp.png "D31 Linear Sum")

Figures 1 and 2 can be generated with the data provided in examples_data by 
1. Running <span style="color:red"> python3 DStab.km.toy.py </span>. This script creates .mat files with scores
2. Running <span style="color:red"> python3 DStab_plot_toy.py </span>. This script creates .png plot files.
3. Finally <span style="color:red"> python3 stats_toy.py </span> calculates statistics and p-values
