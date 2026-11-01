# Reproducing the writing sample

Run from the repository root. Outputs go to `results/simulation/` and `results/lendingclub/`.
Copy the figures into `paper/figures/` (or the LaTeX folder) under the same names.

## Data (once)

| Step | Command |
|---|---|
| Download and filter the LendingClub loan dataset | `python src/prepare_lendingclub.py --raw data/raw/accepted_2007_to_2018Q4.csv` |
| Fit p_hat and c_hat on loans before 2015, predict validation and 2015 loans (settings in Appendix D) | `python src/train_models.py` |

## Simulation study (Sections 3 and 5, Appendix C)

| Paper item | Command | Output |
|---|---|---|
| Figure 1(a), review shares 70.0 / 79.8 / 83.0 / 82.1, S_cc, a_low | `python src/simulation.py fig1a` | `fig_shape.png` |
| Figure 1(b), uniqueness check (80 levels, width at most 0.02) | `python src/simulation.py fig1b --step 0.001` | `fig_unique.png` |
| Figure 2, Section 5 numbers (slope -0.51, 11.7 to 0.5 pp) | `python src/simulation.py fig2` | `fig_theorem2.png`, `fig_theorem2_data.npz` |
| Bias paragraph of Section 5 (factor 1.6, direction factor about 3, slope 0.83), printed | `python src/simulation.py fig2 --plot-only --with-bias` | printed, plus a three-panel figure not used in the paper |
| Redraw Figure 2 only | `python src/simulation.py fig2 --plot-only` | `fig_theorem2.png` |

## LendingClub (Section 6)

| Paper item | Command | Output |
|---|---|---|
| Levels, guarantees, Figure 3, Figure 4, Table 1, numbers of Section 6 | `python src/lendingclub.py run --n 100` | `lc_*.csv`, `fig_lc_weight.png`, `fig_lc_frontier.png` |
| Redraw Figures 3 and 4 only | `python src/lendingclub.py figs` | same figures |
| Diagnostics of the estimates and weight from 2015 loans without outcomes (gap paragraph) | `python src/lendingclub.py shift --n 100` | `lc_shift_*.csv`, `fig_lc_shift.png` |
| Calibration on loans before 2015 (Discussion: 20 of 21 levels exceed severity, 128% and 110% at $60) | `python src/lendingclub.py delay --n 100` | `lc_delay_summary.csv` |
| Plug-in cut-off without calibration (Section 6, "Why calibrate") | `python src/lendingclub.py plugin --n 100` | `lc_plugin_summary.csv` |
| Default rate and loss per default by loan-amount fifth (13.3% vs 14.9%, more than six times) | `python src/lendingclub.py quintiles` | printed |
