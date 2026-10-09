set -e

# 公共 nsys 参数，避免每次重复书写
NSYS_ARGS="--trace=cuda,nvtx,osrt --capture-range=nvtx --nvtx-capture=benchmark_run --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=none --force-overwrite=true"

echo "=========================================================="
echo "1/4: Running Forward Pass [Full Precision - FP32]"
echo "=========================================================="
nsys profile -o profile_fwd_fp32 \
    $NSYS_ARGS \
    python cs336_systems/benchmark.py --mode forward --model-size small --context-length 256

echo "=========================================================="
echo "2/4: Running Forward Pass [Mixed Precision - BF16]"
echo "=========================================================="
nsys profile -o profile_fwd_bf16 \
    $NSYS_ARGS \
    python cs336_systems/benchmark.py --mode forward --mixed-precision --model-size small --context-length 256

echo "=========================================================="
echo "3/4: Running Train Pass (Fwd+Bwd+Opt) [Full Precision - FP32]"
echo "=========================================================="
nsys profile -o profile_train_fp32 \
    $NSYS_ARGS \
    python cs336_systems/benchmark.py --mode train --model-size small --context-length 256

echo "=========================================================="
echo "4/4: Running Train Pass (Fwd+Bwd+Opt) [Mixed Precision - BF16]"
echo "=========================================================="
nsys profile -o profile_train_bf16 \
    $NSYS_ARGS \
    python cs336_systems/benchmark.py --mode train --mixed-precision --model-size small --context-length 256

echo "=========================================================="
echo "All profiling complete! Generated reports:"
echo " - profile_fwd_fp32.nsys-rep"
echo " - profile_fwd_bf16.nsys-rep"
echo " - profile_train_fp32.nsys-rep"
echo " - profile_train_bf16.nsys-rep"
echo "=========================================================="