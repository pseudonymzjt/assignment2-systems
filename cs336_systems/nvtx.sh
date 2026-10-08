# 生成 forward profile
uv run nsys profile -o profile_fwd \
    --trace=cuda,nvtx,osrt \
    --capture-range=nvtx \
    --nvtx-capture=benchmark_run \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    --capture-range-end=none \
    python cs336_systems/benchmark.py --mode forward

# 生成 train (forward + backward + optimizer) profile
uv run nsys profile -o profile_train \
    --trace=cuda,nvtx,osrt \
    --capture-range=nvtx \
    --nvtx-capture=benchmark_run \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    --capture-range-end=none \
    python cs336_systems/benchmark.py --mode train