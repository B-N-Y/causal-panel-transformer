# Causal Panel Transformer (CPT)

Code and result files for the paper

> **Identifiability of Latent Causal Dynamics with Observed Regimes: Levers, Limits and an Estimator**

The repository contains the simulator used throughout the paper, the Causal Panel Transformer (CPT), the end-to-end pipeline
(dimension selection, Stage A, choice of the Stage-B lever by split-half stability, edge typing), the scripts that run the
competing methods, the job lists that define every run, and the result files behind every number in Sections 7 and 8 and Appendix D.

## Layout

| Folder | Content |
|---|---|
| `cpt/` | Simulator (`simulation.py`, Counterexample 5.11 in `simulation_bs.py`), CPT model and losses, encoders, dimension selection, Stage-B levers (`b3_pattern.py`: B3 scan and uniqueness check), edge typing, pipeline |
| `baselines/` | Runners for TDRL, iVAE, iSDS and NCTRL, on our generator and on NCTRL's generator |
| `experiments/` | Estimator-free validation and the misspecified-pattern experiment, CPT evaluation, pipeline evaluation, dimension selection, runs on the competitors' generators, random-rotation and PCA baselines and thresholds, nearly proportional profiles, panels, sample size and label errors (`section86.py`), light tunnel (`real_data_citris.py`, `citris_uniform_eval.py`) |
| `analysis/` | `summarize_runs.py` (summaries of the runs from `results/`), `table3_details.py` (tests of Table 3 and Appendix D), `light_tunnel_rotation_check.py` (Section 8.7) |
| `jobs/` | One job list per run: each line is one command, executed from the repository root |
| `slurm/` | Generic SLURM array template that runs one line of a job list per array task |
| `results/` | Result files of every run used in the paper (one folder per run, CSV) |
| `scripts/` | `fetch_third_party.sh`: fetches the competitors' code at the commits used in the paper; `prepare_light_tunnel.py`: prepares the light-tunnel data of Section 8.7 |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_third_party.sh   # needed for TDRL, iVAE, iSDS and NCTRL, and for every script that trains an encoder
```

The competitors' code is not redistributed. `scripts/fetch_third_party.sh` clones TDRL, NCTRL and iSDS into `third_party/` at
the commits used in the paper; each keeps its own license.

## Result files and summaries

The result files behind Sections 7 and 8 and Appendix D are shipped in `results/`, and the summaries can be recomputed from them
without training anything:

```bash
python analysis/summarize_runs.py --results results
python experiments/summarize_pipeline.py 'results/PIPE160/pipeline_*_s*.csv'
python experiments/summarize_pipeline.py 'results/PIPE150/pipeline_*_s*.csv' --extra
```

## Rerunning the experiments

Each run is defined by a job list in `jobs/`. A single line can be run directly from the repository root, for example

```bash
head -n 1 jobs/PIPE160.txt | bash
```

and a whole run can be submitted as a SLURM array with `slurm/run_joblist.sh` (instructions at the top of that file; replace the
placeholders for your cluster). New results are written to `outputs/`. Runs that train an encoder used one GPU each; they also
run on a CPU, but TDRL and NCTRL are slow there. Results may differ from the shipped files in the last digits because of GPU
non-determinism.

The estimator-free analyses run on a CPU in minutes:

```bash
python experiments/estimator_free_validation.py                 # Table 2 (results/TABLE2)
bash jobs/B3_MISSPEC.txt                                        # misspecified zero pattern on affine copies (results/B3_MISSPEC; basis of the statement on misspecified patterns in Section 9)
python experiments/random_rotation_baseline.py --out results/RANDOM_ROTATION/random_rotation_baseline.csv   # Section 8.3
bash jobs/DELTA_EF.txt                                          # nearly proportional profiles on affine copies, Section 8.5
bash jobs/PCA_BASELINE.txt                                      # linear baseline, Table 3
python analysis/table3_details.py --out_dir results/TABLE3_DETAILS                         # tests of Table 3, Appendix D
python experiments/rotation_thresholds.py --out results/ROTATION_THRESHOLD/rotation_thresholds.csv   # criterion of Section 8.1
```

### Light tunnel (Section 8.7)

The data are the public benchmark of Gamella, Bing and Runge (2025), built on the Causal Chambers (Gamella, Peters and Bühlmann,
2025) and distributed under the CC BY 4.0 license; they are not redistributed here. Download `lt_crl_benchmark_v1.zip` (2.8 GB,
MD5 `8ec492afffd0142abd70b1fa7082b104`) from the Causal Chambers repository into `data/`, then

```bash
python scripts/prepare_light_tunnel.py data/lt_crl_benchmark_v1.zip data/light_tunnel
```

writes the 100,000 images of the sequence `citris_1` and its recorded factors to `data/light_tunnel/`. The job lists `LT_ENCODERS`,
`LT_RC`, `LT_RC_BLOCKS` and `LT_UNIFORM` then run as above. The random-rotation check of Section 8.7 uses the output of `LT_RC`:

```bash
python analysis/light_tunnel_rotation_check.py outputs/LT_RC/citris_RC_k5_s0.npz data/light_tunnel/citris_1_truth.csv results/LT_RC/rotation_check.csv
```

Run `B3_UNIQ_DEV` reads the Stage-A latents saved by `B3_PIPE` and `B3_BS` (`--save_latents`, in `outputs/z/`). Run `R7T_FULL` takes
about nine hours per seed: each line trains for at most 3.6 hours, saves a checkpoint in `outputs/ckpt/` and resumes when the same line
is run again, so the array is submitted three or four times in a chain (`sbatch --dependency=aftercorr:<previous job>`).

The pipeline evaluation (`experiments/evaluate_pipeline.py`) applies B2 to the residuals pooled across regimes by default;
`--b2_variant single` applies it to the regime with the most observations, as in Section 7, and `--b2_extra pooled|single`
additionally reports a second B2 variant on the same Stage-A fit (columns `*_extra`). Run `PIPE160` used single-regime B2 as
the primary variant. Run `PIPE150` used pooled B2 as the primary variant, so its primary columns (`selection`, `certificate`,
`mcc_resolved` and the edge scores) refer to the pooled solution; Section 8.5 uses its single-regime solution, which is in the
`*_extra` columns, and the edge scores of that solution, which are in `results/PIPE150/rescore_edges_single_regime.csv`. That
file is written by

```bash
python experiments/rescore_edges_single_regime.py --runs 'outputs/pipeline_v7b2_*_s*.csv' --zdir outputs/z
```

from the Stage-A latents saved by `jobs/PIPE150.txt` (option `--save_latents`); the latents themselves are not shipped, and
rerunning the job list regenerates them.

## Runs and where they are used

| Paper | Runs |
|---|---|
| Section 7: prior ablation, Gaussian against Student-t5 transition prior (seeds 500–509) | `PRIOR_ABLATION` |
| Section 8.2, Table 2: estimator-free validation | `TABLE2` |
| Section 8.1: dataset-specific solved criterion | `ROTATION_THRESHOLD` |
| Section 8.3: random-rotation baseline | `RANDOM_ROTATION`; raw MCC of the estimators from `R10`, `R1T`, `R5T`, `R6B` |
| Section 8.4, Table 3: our generator, main cells | CPT `R10`; TDRL `R1T`, `R5T`; iVAE `R6B`; iSDS `ISDS2`; NCTRL `R22`, `R22v2` |
| Table 3: main cells, fresh seeds (200–209) | CPT and TDRL `R10F`; iSDS `ISDS_R10F`; NCTRL `NCTRL_R10F` (29 files: seed 208 of `E4_none_t5` diverged and wrote none); PCA `PCA_BASELINE` |
| Table 3: hard settings | CPT `R14`; TDRL `K1v2_tdrl`; NCTRL `K1v2_nctrl`; iSDS `K1_isds`, `K1v2_isds_T1000` |
| Table 3: TDRL's generator | `R7T_FULL` (TDRL, 50 epochs, resumed from checkpoints across a chain of jobs), `R13` (CPT); earlier budget `R7T`, `R11`, `R12` |
| Table 3: NCTRL's generator | `R15`, `R15c`, `R15c_s7_v2` |
| Table 3: tests and Holm correction, strict criterion; Appendix D, Table 5 | `TABLE3_DETAILS` (from the runs above) |
| Section 8.5: dimension rule on its own | `DIMV2` |
| Section 8.5: end-to-end pipeline (seeds 150–169) | `PIPE150` (CPT and TDRL encoders; single-regime edges in `rescore_edges_single_regime.csv`), `PIPE160` (two CPT fits) |
| Section 8.5: $t_5$ scenario without the isolated component (seeds 800–809) | `B2_NOSPURIOUS` |
| Section 8.5: nearly proportional profiles (seeds 700–709; affine copies 700–719) | `DELTA_PIPE`, `DELTA_EF` |
| Section 8.5: lever B3, identifiable scenario (seeds 600–619) | `B3_PIPE` (development), `B3_PIPE_FRESH` |
| Section 8.5: lever B3, Counterexample 5.11 (seeds 650–669) | `B3_BS` (development), `B3_BS_FRESH`; uniqueness check on the development data `B3_UNIQ_DEV` |
| Section 8.6: panels, sample size and errors in the regime labels (seeds 900–909) | `S86` |
| Section 8.7, Table 4: light tunnel, held-out images | `LT_UNIFORM` |
| Section 8.7: encoders on the light tunnel (Transformer, iVAE, TDRL, PCA; seeds 0–9) | `LT_ENCODERS` |
| Section 8.7: regime contrast on the training sequence, random-rotation check | `LT_RC` |
| Section 8.7: ten blocks of 10,000 periods | `LT_RC_BLOCKS` |
| Appendix D: wall-clock times | `R10`, `R1T`, `R5T`, `R6B`, `ISDS2`, `R22` |
| Appendix D: sensitivity of the competitors to their settings | `K4v2` |
| Appendix D: encoder ablation | `R18a`, `R18b` |
| Appendix D: held-out likelihood, first-order linear against richer transition | `R10` and `R12_CLASS` (model class), `R13` (TDRL's generator), `R15` (NCTRL's generator) |

`RANDOM_ROTATION` contains one row per cell and seed (`random_rotation_baseline.csv`) and one row per random rotation
(`random_rotation_draws.csv`). Some older runs were produced by earlier versions of the same scripts and therefore contain fewer
columns than the current code writes; no column was added or changed after the fact.

## License

MIT (see `LICENSE`). The third-party code fetched by `scripts/fetch_third_party.sh` is under the licenses of its authors.
