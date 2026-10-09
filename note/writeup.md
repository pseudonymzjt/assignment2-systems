### benchmark scripts

2. Time the forward, backward, and optimizer step for the model sizes described in 
Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of 
timings over 10 measurement steps. How long does a forward pass take? How about a 
backward pass? Do you see high variability across measurements, or is the standard deviation 
small?
> With 5 warm-up steps, the forward pass took 32.367 ms on average with a standard deviation of 0.480 ms. The forward-and-backward pass took 104.109 ms with a standard deviation of 7.028 ms, while the full training step including the optimizer step took 104.862 ms with a standard deviation of 6.870 ms. The forward measurements had low variability, while the forward-and-backward and full training measurements showed moderate variability.

3. One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis 
without the warm-up steps. How does this affect your results? Why do you think this 
happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be 
different?
> Without warm-up steps, the measurements were highly variable: the forward pass took 162.177 ± 410.570 ms, and the forward-and-backward pass took 245.762 ± 458.231 ms. This is likely because the first measured steps include CUDA context initialization, kernel loading, memory allocator setup, and other one-time overheads. With one or two warm-up steps, the results become close to the five-warm-up-step results, although small differences remain because of GPU scheduling, clock frequency, and system load.

### nsys profile
1. What is the total time spent on your forward pass? Does it match what we had measured 
before with the Python standard library?
> The total forward pass GPU kernel execution time measured in Nsight is approximately 12.05 ms per pass, which is noticeably shorter than the 33.6 ms measured via Python's standard library. This discrepancy is primarily driven by CPU launch overhead, PyTorch framework runtime overhead, and the profiling/tracing overhead introduced by nsys.
2. What CUDA kernel takes the most cumulative GPU time during the forward pass? How 
many times is this kernel invoked during a single forward pass of your model? Is it the same 
kernel that takes the most runtime when you do both forward and backward passes? (Hint: 
look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using 
NVTX ranges to identify which parts of the model are responsible for which kernels.)
> During a forward pass, magma_sgemmEx_kernel consumes the most cumulative GPU time (54.7%) and is invoked 65 times per single pass (130 times across 2 measurement steps). It remains the most time-consuming kernel in train mode, though its share drops to 16.9% due to the additional runtime introduced by backward GEMMs and elementwise optimizer passes.
3. Although the vast majority of FLOPs take place in matrix multiplications, you will notice 
that several other kernels still take a non-trivial amount of the overall runtime. What other 
kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the 
forward pass?
> [forward kernels](figures/fwd.png)
> [train kernels](figures/train.png)
> Besides matrix multiplications, non-trivial runtime is taken by various memory-bound elementwise kernels (at::native::elementwise_kernel / vectorized_elementwise_kernel), which together account for ~25% of the total forward time. Specifically, these correspond to the attention mask (torch.where, ~6.0%), the exponential operation in softmax (exp_kernel_cuda, ~5.0%), and elementwise tensor operations like residual connections (CUDAFunctor_add) and attention scaling.
4. Profile running one complete training step with your implementation of AdamW (i.e., the 
forward pass, computing the loss and running a backward pass, and finally an optimizer step, 
as you’d do during training). How does the fraction of time spent on matrix multiplication 
change, compared to doing inference (forward pass only)? How about other kernels?
> This question has been answered in 2 and 3.
5. Compare the runtime of the softmax operation versus the matrix multiplication operations 
within the self-attention layer of your model during a forward pass. How does the difference 
in runtimes compare to the difference in FLOPs?
> While the matrix multiplication (`final matmul`) requires roughly $2 \cdot d_k / 3 \approx 21\times$ more FLOPs than the softmax operation ($2 B H L^2 d_k$ vs. $3 B H L^2$), their actual runtimes are nearly identical (~108 µs each). This vast discrepancy occurs because GEMM is **compute-bound** and achieves high arithmetic intensity on GPU compute units, whereas softmax is **memory-bound** and bottlenecked by DRAM read/write bandwidth across multiple elementwise passes.

### mixed precision
> As expected, in [code provided](../cs336_systems/mixed_precision.py), float32 offeres highest precision(10.0001), and using float16 all along gets the worst result(9.9531). And the third and fourth experiments show that improving precision after computation makes no sense(both 10.0021).

1. Consider the following model:
```python
class ToyModel(nn.Module):
def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()
def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.ln(x)
    x = self.fc2(x)
    return x
```
Suppose we are training the model on a GPU and that the model parameters are originally in 
FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
• the model parameters within the autocast context?
• the output of the first feed-forward layer (ToyModel.fc1)?
• the output of layer norm (ToyModel.ln)?
• the model’s predicted logits?
• the loss?
• the model’s gradients?
> float32; float16; float32; float16; float32; float32
2. You should have seen that FP16 mixed precision autocasting treats the layer normalization 
layer differently than the feed-forward layers. What parts of layer normalization are sensitive 
to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer 
normalization differently? Why or why not?
> LN is easier to underflow or overflow cause'of the unenough 5 expo digits. The mantissa of BF16 is only 7, not very accurate, may leading to unstability of result.
3. Modify your benchmarking script to optionally run the model using mixed precision with 
BF16. Time the forward and backward passes with and without mixed-precision for each 
language model size described in 
Section 2.1.2. Compare the results of using full precision 
versus mixed precision, and comment on any trends as model size changes. You may find the 
nullcontext no-op context manager to be useful.
[bf16](figures/train_bf16.png)
[fp32](figures/train_fp32.png)
> Comparing full-precision (FP32) to mixed-precision (BF16) on a complete training step, BF16 achieves roughly a 1.95× speedup in overall runtime (reducing total benchmark time from ~1.76s down to ~0.90s), with the backward pass exhibiting the most prominent acceleration (~2.95× speedup).
> Across kernel breakdowns, the dominant SGEMM kernels (originally accounting for >65% of GPU time) are replaced by Tensor Core CUTLASS BF16 kernels whose aggregate execution time drops by ~4.7×.
> As model parameters and sequence dimensions grow, the workload transitions from being dispatch/bandwidth-bound to compute-bound, allowing the BF16 Tensor Core arithmetic throughput advantage to become increasingly pronounced while cutting backward activation memory bandwidth.