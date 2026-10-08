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
