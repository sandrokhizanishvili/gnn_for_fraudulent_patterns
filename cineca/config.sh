# =====================================================================================
#  config.sh — the ONLY file to edit. Every job and script reads its settings from here.
# =====================================================================================

# ---- must check / edit --------------------------------------------------------------
export ACCOUNT="CHANGE_ME"                  # project account to charge (list yours with: saldo -b)
export PARTITION="boost_usr_prod"           # Leonardo Booster GPU partition
export QOS="normal"                         # 24 h max per job
export QOS_DEBUG="boost_qos_dbg"            # 30-min debug queue, used only by "run_step.sh check"
export MODULES="python/3.11.6--gcc--8.5.0"  # a Python 3.11-3.13 module (list: module avail python);
                                            # set to "" if you use the conda-pack env (handoff_env.tar.gz)
export BASE="$WORK/aml_thesis"              # the folder holding code/ Data/ env/ (see README_LORIS.md)
export DATA_DIR="$BASE/Data"                # train_graph.pt, val_graph.pt, test_graph.pt, feature_meta.json
export ENV_DIR="$BASE/env"                  # the Python environment (setup_env.sh creates it)
export OUT_DIR="$BASE/Outputs"              # every result goes here (needs up to ~400 GB free during tuning)
export N_TRIALS=100                         # trials per (operator, config); read once by "run_step.sh draw"
export MAX_PARALLEL=32                      # running tasks per job array (8 arrays -> up to 256 GPUs);
                                            # lower it if your per-user limits are smaller (README section 3)

# ---- usually leave as is ------------------------------------------------------------
export CPUS=8                               # CPU cores per GPU task (a quarter of a node)
export MEM=auto                             # job memory: auto = from the probe (1.5 x peak + 4 GB); or e.g. 64G
export MEM_PROBE=120G                       # memory of the probe jobs (a quarter of a node)
export TIME_CHECK=00:30:00
export TIME_PROBE=08:00:00
export TIME_TUNE=24:00:00
export TIME_FINAL=24:00:00
