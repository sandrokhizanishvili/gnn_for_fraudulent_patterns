# Hyperparameter tuning for Sandro's thesis on Leonardo — instructions

**What this is.** This package tunes four graph neural networks (GIN, PNA, GATv2, a graph transformer) on two feature
sets. It follows the search protocol of Luo et al. (2024): 100 random trials per (model, feature set), 2 seeds each,
then a 5-seed retrain of each winner. Every task is a single-GPU SLURM job. Plain Python and bash, no notebooks.
You only edit `cineca/config.sh`.

**Assumptions to confirm** (tell Sandro if any is wrong):
- login nodes have internet and compute nodes don't;
- partition `boost_usr_prod`, QoS `normal` (24 h) and `boost_qos_dbg` (30 min) are available to your account;
- the GPU driver is ≥ 525. The CUDA 12.6 wheels need it; `nvidia-smi` on a compute node shows it.

## 1. Upload and unpack

Download `handoff_loris.zip` (about 1.2 GB) from the Google Drive link and copy it to `$WORK/aml_thesis/` (scp or
rsync to a login node). Then:
```bash
cd $WORK/aml_thesis
unzip handoff_loris.zip    # README_LORIS.md + handoff_code.tar.gz + handoff_data.tar.gz (or: python3 -m zipfile -e handoff_loris.zip .)
tar -xzf handoff_data.tar.gz && (cd Data && sha256sum -c SHA256SUMS)    # Data/, 4.4 GB, all four lines "OK"
mkdir -p code && tar -xzf handoff_code.tar.gz -C code && cd code         # code/gnn_core.py, code/cineca/
```
Edit `cineca/config.sh`:
- set `ACCOUNT`;
- check that `MODULES` names a Python 3.11–3.13 module (`module avail python`);
- the paths default to `$WORK/aml_thesis/{Data,env,Outputs}`.

`OUT_DIR` needs up to ~400 GB free while tuning runs: every running trial keeps its 100 epoch checkpoints until both of
its seeds are done, then only one per seed stays (please check the project quota, e.g. with `cindata`).

## 2. Environment (once, on a login node, ~15 min)
```bash
bash cineca/setup_env.sh        # venv in ENV_DIR + pip install of requirements.lock + a CPU check; ends with "OK"
```
If pip gets killed, run it again (installed packages are skipped). If it still fails, use the backup.
- Sandro sends `handoff_env.tar.gz`.
- Unpack it: `mkdir -p $WORK/aml_thesis/env && tar -xzf handoff_env.tar.gz -C $WORK/aml_thesis/env`.
- Activate it and run `conda-unpack`.
- Set `MODULES=""` in `config.sh`.

## 3. Per-user limits (before step 6)

Up to 8 job arrays run at once, with `MAX_PARALLEL` (default 32) tasks each, so up to 256 GPUs. Check what your
project may use:
```bash
sacctmgr show qos normal format=Name,MaxWall,MaxTRESPU%40,MaxJobsPU,MaxSubmitPU
sacctmgr show assoc user=$USER format=Account,QOS%40,MaxJobs,MaxSubmit
scontrol show config | grep -i MaxArraySize
```
- If the limits are lower, set `MAX_PARALLEL` in `config.sh` to about (GPU limit) / 8.
- Excess tasks just wait in the queue.
- If `sbatch` refuses a submission (submit limit), `run_step.sh` says so and continues. Run the same step later and
  only the missing tasks go in.

## 4. Commands, in order (from `$WORK/aml_thesis/code`)

| # | Command | What / how long | GPU-hours |
|---|---|---|---|
| 1 | `bash cineca/run_step.sh check` | GPU check in the debug queue, ~5 min; `logs/check_*.out` ends with `OK` | < 0.1 |
| 2 | `bash cineca/run_step.sh probe` | which network depths fit on an A100; 4 tasks, a few hours | ≤ 12 |
| 3 | `bash cineca/run_step.sh draw` | the fixed trial list (login node, seconds); prints the list and the depths that fit | 0 |
| 4 | `bash cineca/run_step.sh tune --first 10` | trials 0–9 of every model and feature set (160 tasks), ~1–2 days | ≈ 1,500–2,600 |
| 5 | `bash cineca/pack_results.sh` | **send Sandro `results_small.tar.gz` and wait for his OK** | 0 |
| 6 | `bash cineca/run_step.sh tune` | the remaining trials (1,600 tasks in all); **re-run once a day** until `status.sh` shows all done | ≈ 15,000 (likely) – 27,000 (upper bound), incl. step 4 |
| 7 | `bash cineca/run_step.sh aggregate` | checks + the winner of each pair (login node, minutes) | 0 |
| 8 | `bash cineca/run_step.sh final` | 5-seed retrain of the 8 winners (40 tasks), ~2–3 days | ≈ 260–460 |
| 9 | `bash cineca/pack_results.sh` | the three archives to send back (section 6) | 0 |

**If `check` or `probe` fails, send Sandro the `logs/` folder and stop.** The probe is the first real-data test.

Wall clock of step 6 depends on how many GPUs run at once: about 3 days at 256 GPUs and about 10 at 64, plus queue
time. Some trials need more than one 24 h job; they continue from their last epoch when step 6 is run again.

## 5. Progress and failures
```bash
bash cineca/status.sh     # per step and pair: done / running / queued / timeout / failed / missing
```
- **timeout** is normal for long trials: run the same step again and they resume.
- **failed**: look at `logs/<job>_<jobid>_<task>.err`, re-run the step once, and if it fails again send the log to Sandro.
- Finished work is never redone.
- Logs live in `code/logs/`.

## 6. What to send back
`pack_results.sh` writes three archives next to `Outputs/` and prints their sizes:
- `results_small.tar.gz`: every small result file and the logs, tens of MB. This is also the early check-in file.
- `results_predictions.tar.gz`: per-transaction predictions of the final retrain, a few GB.
- `results_models.tar`: the kept checkpoints, roughly 10–60 GB.

Send all three: Google Drive (institutional account) works, or WeTransfer for files under its size limit.
`split -b 2G <file> <file>.part-` cuts a large file if needed. **Please keep the full `Outputs/` folder on Cineca
until Sandro confirms he has received everything.**
