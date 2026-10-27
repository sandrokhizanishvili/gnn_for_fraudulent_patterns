# Hyperparameter tuning for Sandro's thesis on Leonardo — instructions

**What this is.** This package tunes four graph neural networks (GIN, PNA, GATv2, a graph transformer) on two feature
sets, following Luo et al. (2024): 100 random trials per (model, feature set), 2 seeds each, then a 5-seed retrain of
each winner. Every task is a single-GPU SLURM job. The scripts here are `config.sh` (settings), `env.sh` (sourced by the
others), `run.sh` (the one command), `job.sbatch` (the one job script), `luo.py` (every step) and this README.
The Python packages are `env/environment.yml`. What the code does is in the repository's `CODE_GUIDE.md`.

**Where to type.** Every command below is typed on a Leonardo login node (the Linux shell you get over SSH), never
on your own computer. The training itself runs on GPU compute nodes: `run.sh` submits it to SLURM for you.

**Assumptions to confirm** (tell Sandro if any is wrong): login nodes have internet and compute nodes don't;
partition `boost_usr_prod`, QoS `normal` (24 h) and `boost_qos_dbg` (30 min) are available to your account; the GPU
driver is ≥ 525 (the CUDA 12.6 wheels need it).

## 1. Code, data, settings (once, on a login node)

Code comes from GitHub; the data (4.4 GB, not in git) comes once as `handoff_data.tar.gz` from the Google Drive link.
Starting from nothing:
```bash
mkdir -p $WORK/aml_thesis && cd $WORK/aml_thesis
git clone https://github.com/sandrokhizanishvili/gnn_for_fraudulent_patterns.git code
# download handoff_data.tar.gz (1.2 GB) from the Drive link into this folder, then:
tar -xzf handoff_data.tar.gz && (cd Data && sha256sum -c SHA256SUMS)   # Data/, four lines "OK"
rm handoff_data.tar.gz
```
Put your account in `code/cineca/config.local.sh` (not in git, so `git pull` never touches it):
```bash
cat > code/cineca/config.local.sh <<'EOF'
export ACCOUNT="<your project account>"      # saldo -b
EOF
```
Every other setting has a default in `cineca/config.sh` (paths `$WORK/aml_thesis/{Data,env,Outputs}`, `MAX_PARALLEL`
32); override any of them with the same `export` line in `config.local.sh`. `OUT_DIR` needs up to ~400 GB free while
tuning runs (every running trial keeps its 100 epoch checkpoints until both seeds are done, then one per seed).

**Updates:** when Sandro says the code changed, `cd $WORK/aml_thesis/code && git pull`. Nothing else to do.

## 2. Environment (once, on a login node, ~15 min)
```bash
cd $WORK/aml_thesis/code
conda env create -p $WORK/aml_thesis/env -f env/environment.yml
```
If `conda` is not found, load it first (`module avail anaconda`, then e.g. `module load anaconda3`) and add the same
name to `config.local.sh` as `export MODULES="anaconda3"`, so the jobs load it too. If a compute node then cannot see
the environment, tell Sandro (a conda-pack archive is the fallback).

## 3. Per-user limits (before step 6)

Up to 8 job arrays run at once, with `MAX_PARALLEL` tasks each. Check your limits:
```bash
sacctmgr show qos normal format=Name,MaxWall,MaxTRESPU%40,MaxJobsPU,MaxSubmitPU
sacctmgr show assoc user=$USER format=Account,QOS%40,MaxJobs,MaxSubmit
```
If they are lower, set `MAX_PARALLEL` in `config.local.sh` to about (GPU limit) / 8. Excess tasks wait in the queue;
if `sbatch` refuses a submission, `run.sh` says so and the same step later submits only what is missing.

## 4. Commands, in order (from `$WORK/aml_thesis/code`)

| # | Command | What / how long | GPU-hours |
|---|---|---|---|
| 1 | `bash cineca/run.sh check` | GPU check in the debug queue, ~5 min; `logs/check_*.out` ends with `OK` | < 0.1 |
| 2 | `bash cineca/run.sh probe` | which network depths fit on an A100; 4 tasks, a few hours | ≤ 12 |
| 3 | `bash cineca/run.sh draw` | the fixed trial list (login node, seconds) | 0 |
| 4 | `bash cineca/run.sh tune --first 10` | trials 0–9 of every pair (160 tasks), ~1–2 days | ≈ 1,500–2,600 |
| 5 | `bash cineca/run.sh pack` | **send Sandro `results_small.tar.gz` and wait for his OK** | 0 |
| 6 | `bash cineca/run.sh tune` | the remaining trials (1,600 tasks in all); **re-run once a day** until `status` shows all done | ≈ 15,000–27,000 incl. step 4 |
| 7 | `bash cineca/run.sh aggregate` | checks + the winner of each pair (login node, minutes) | 0 |
| 8 | `bash cineca/run.sh final` | 5-seed retrain of the 8 winners (40 tasks), ~2–3 days | ≈ 260–460 |
| 9 | `bash cineca/run.sh pack` | the three archives to send back (section 6) | 0 |

**If `check` or `probe` fails, send Sandro the `logs/` folder and stop.** Step 6 takes about 3 days at 256 GPUs and
about 10 at 64, plus queue time; trials that need more than one 24 h job continue from their last epoch when the step
is run again.

## 5. Progress and failures
```bash
bash cineca/run.sh status     # per step and pair: done / running / queued / timeout / failed / missing
```
- **timeout** is normal for long trials: run the same step again and they resume.
- **failed**: look at `logs/<job>_<jobid>_<task>.err`, re-run the step once, and if it fails again send the log to Sandro.
- Finished work is never redone. Logs live in `code/logs/`.

## 6. What to send back
`run.sh pack` writes three archives next to `Outputs/` and prints their sizes: `results_small.tar.gz` (every small
result file and the logs, tens of MB; also the early check-in file), `results_predictions.tar.gz` (per-transaction
predictions of the final retrain, a few GB), `results_models.tar` (the kept checkpoints, 10–60 GB). Google Drive
(institutional account) or WeTransfer; `split -b 2G <file> <file>.part-` cuts a large file. **Keep the full `Outputs/`
folder on Cineca until Sandro confirms he has received everything.**
