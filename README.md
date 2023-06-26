# DStab: estimating clustering quality by distance stability
[read paper here](https://doi.org/10.1007/s10044-023-01175-7)

---

Folder DStab/dist contains packages, which can be
- installed from within DStab/dist by: python3 -m pip install ./DStab-0.0.1-py3-none-any.whl
- uninstalled from within DStab/dist by: python3 -m pip uninstall ./DStab-0.0.1-py3-none-any.whl

---
Folder DStab/examples_data/ contains examples and data.

Figure 1
![Fig.1 - R15 Linear Sum](https://github.com/ar-baya/DStab/blob/main/examples_data/R15_res_tmp.png "R15 Linear Sum")

Figure 2
![Fig.2 - D31 Linear Sum](https://github.com/ar-baya/DStab/blob/main/examples_data/R15_res_tmp.png "D31 Linear Sum")

---
Figures 1 and 2 can be generated with the data provided in examples_data by 
1. Running <span style="color:red">python3 DStab.km.toy.py</span>. This script creates .mat files with scores
2. Running <span style="color:red">python3 DStab_plot_toy.py</span>. This script creates .png plot files.
3. Finally <span style="color:red">python3 stats_toy.py</span> calculates statistics and p-values
---
R15 Dataset stats (Figure 1):
```
Name File:  R15
k:  15.0  stat:  0.0  pval:  3.391594415816857e-33
k:  14.0  stat:  1529.0  pval:  1.707259613411043e-29
k:  9.0  stat:  1529.0  pval:  1.707259613411043e-29
k:  13.0  stat:  5941.0  pval:  4.2201582437941816e-20
k:  10.0  stat:  5983.0  pval:  5.0764619648336754e-20
```
D31 Dataset stats (Figure 2):
```
Name File:  D31
k:  31.0  stat:  7889.0  pval:  1.4611466165743396e-16
k:  15.0  stat:  20799.0  pval:  0.022360594121253588
k:  7.0  stat:  23913.0  pval:  0.30181092723680597
k:  6.0  stat:  24878.0  pval:  0.47685023114742003
k:  32.0  stat:  26738.0  pval:  0.7969472149528465

```
