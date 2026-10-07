# Shared environment for anti-Stokes batch jobs on DeltaAI (source this file).
export PATH=/u/glwagner/opt/julia-1.11.9/bin:$PATH
# Let CUDA.jl use its own runtime artifacts rather than the HPC SDK libraries on LD_LIBRARY_PATH.
unset LD_LIBRARY_PATH
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-8}
export JULIA_NUM_PRECOMPILE_TASKS=${JULIA_NUM_PRECOMPILE_TASKS:-8}
export JULIA_PKG_PRECOMPILE_AUTO=0
cd /u/glwagner/NumericalWaveTanks.jl
mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  start: $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
