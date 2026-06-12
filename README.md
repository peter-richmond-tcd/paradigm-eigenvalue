# Spectral signatures of scientific paradigm formation

Code and processed data for the paper:

> **Spectral signatures of scientific paradigm formation:
> an eigenvalue analysis of concept co-occurrence networks**
>
> Peter Richmond, Trinity College Dublin
>
> *Physica A: Statistical Mechanics and its Applications* (submitted 2026)

## Overview

This repository contains all Python code and processed data needed
to reproduce the figures and tables in the paper. The central
analysis constructs a concept co-occurrence matrix from a large
bibliometric corpus drawn from the OpenAlex database, computes its
leading eigenvalues year by year across 2005--2023, and uses the
resulting spectral observables to characterise the emergence of deep
learning as a scientific paradigm.

The repository is organised as follows.

---

## Repository structure

```
paradigm-eigenvalue/
│
├── README.md                           This file
│
├── code/
│   ├── fetch_year.py                   Retrieves raw corpus from OpenAlex API
│   ├── eigenvalue_analysis.py          Global eigenvalue trajectory (Fig 1, Fig 2)
│   ├── eigenvalue_extensions.py        Normalisation and field analysis (Fig 2, Fig 3)
│   ├── normalised_field_analysis.py    Diffusion analysis (Fig 4, Fig 6)
│   ├── physics_concept_trajectories.py Physics field penetration (Fig 5, Fig S1)
│   └── requirements.txt               Python package versions
│
├── data/
│   ├── eigenvalue_results.csv          Global eigenvalue results (Table 1)
│   ├── field_eigenvalues.csv           Field-resolved lambda_1 values
│   ├── tier_eigenvalues.csv            Tier-aggregated lambda_1 values
│   ├── normalised_field_eigenvalues.csv Normalised structural coherence by field
│   ├── normalised_tier_eigenvalues.csv  Normalised structural coherence by tier
│   ├── diffusion_lag.csv               Diffusion front years by field (Fig 6)
│   └── cs_top_concepts_by_year.csv     CS leading eigenvector composition (Fig 3)
│
└── figures/
    ├── eigenvalue_plot.pdf             Fig 1: Global eigenvalue trajectory
    ├── extended_plots.pdf              Fig 2: Volume normalisation and extensions
    ├── cs_concept_trajectories.pdf     Fig 3: CS eigenvector composition
    ├── diffusion_plots.pdf             Fig 4: Normalised coherence by field/tier
    ├── physics_dl_penetration_plot.pdf Fig 5: DL penetration into physics fields
    ├── diffusion_lag_plot.pdf          Fig 6: Diffusion front years
    └── physics_heatmaps.pdf            Fig S1: Physics field heat-maps (supplementary)
```

---

## Requirements

- Python 3.12
- The following libraries (exact versions in `code/requirements.txt`):
  - NumPy
  - SciPy
  - pandas
  - Matplotlib

Install all dependencies with:

```bash
pip install -r code/requirements.txt
```

---

## Reproducing the results

The analysis runs in four stages. Each script writes its outputs
to the directory specified in its header, defaulting to
`~/Desktop/AI_cognition/paradigm_data/`. You can change this path
by editing the `DATA_DIR` and `ANALYSIS_DIR` variables near the top
of each script.

### Stage 1: Retrieve the raw corpus from OpenAlex

```bash
python code/fetch_year.py
```

Edit the `TARGET_YEAR` variable near the top of `fetch_year.py`
and run the script once for each year from 2005 to 2023:

    python code/fetch_year.py   # with TARGET_YEAR = 2005
    python code/fetch_year.py   # with TARGET_YEAR = 2006
    # ... repeat through 2023

Each run takes 1–3 hours. Files already successfully downloaded
are skipped automatically, so the script can be safely interrupted
and restarted. Set your email address in the `EMAIL` variable
to use the OpenAlex polite pool and avoid rate limiting.

The raw corpus is not included in this repository due to its size
(~2.85 million paper-years). The data are freely available from
https://openalex.org under a CC0 licence.
### Stage 2: Global eigenvalue analysis

```bash
python code/eigenvalue_analysis.py
```

Reads the deduplicated corpus, builds the global concept vocabulary,
and computes the leading eigenvalues year by year. Produces
`eigenvalue_results.csv`, `eigenvalue_plot.pdf`, and
`top_concepts_{year}.json` files for selected years.

Runtime: approximately 90 seconds on an Apple M3 Pro (18 GB RAM).

### Stage 3: Normalisation and field-resolved analysis

```bash
python code/eigenvalue_extensions.py
python code/normalised_field_analysis.py
```

Run these in order. `eigenvalue_extensions.py` computes field- and
tier-resolved eigenvalues and the CS concept trajectories.
`normalised_field_analysis.py` normalises by paper count and
computes diffusion front years. Together these produce Figures 2--4
and Figure 6.

Runtime: approximately 15 minutes for `eigenvalue_extensions.py`;
under 5 seconds for `normalised_field_analysis.py`.

### Stage 4: Physics field penetration analysis

```bash
python code/physics_concept_trajectories.py
```

Computes DL penetration fractions and DL/native ratios for the six
physics-adjacent fields, and produces the concept heat-maps.
Produces Figure 5 and Supplementary Figure S1.

Runtime: approximately 25 minutes on an Apple M3 Pro.

---

## Reproducing figures from processed data only

If you do not wish to retrieve the full OpenAlex corpus, all
figures and tables in the paper can be reproduced directly from
the processed CSV files in the `data/` directory. The plotting
routines are embedded in each analysis script; to run them
standalone on the pre-computed CSVs, set the flag

```python
LOAD_FROM_CSV = True
```

near the top of each script, which bypasses the corpus-reading
stage and reads directly from `data/`.

---

## Data availability

The processed CSV files in `data/` are provided under a
CC0 (public domain) licence and may be freely reused.

The raw OpenAlex corpus is available at https://openalex.org
under a CC0 licence.

---

## Citation

If you use this code or data in your research, please cite:

```
Richmond, P. (2026). Spectral signatures of scientific paradigm
formation: an eigenvalue analysis of concept co-occurrence networks.
Physica A: Statistical Mechanics and its Applications.
```

and the Zenodo software deposit:

```
Richmond, P. (2026). paradigm-eigenvalue: code and data.
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## Licence

The code in this repository is released under the
[MIT Licence](LICENSE).

The processed data files are released under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).

---

## Contact

Peter Richmond
School of Physics, Trinity College Dublin
peter_richmond@icloud.com
