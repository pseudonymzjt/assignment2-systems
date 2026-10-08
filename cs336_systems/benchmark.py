# cs336_systems/benchmark.py

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import cs336_basics.model

# 替换 Attention 实现为带 NVTX 注解的版本
from cs336_basics.plugins import annotated_scaled_dot_product_attention
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(args, device):
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    return model


def run_step(model, optimizer, inputs, targets, mode):
    if mode == "forward":
        with nvtx.range("forward_pass"):
            with torch.no_grad():
                model(inputs)
        return

    # 针对 backward 和 train(包含 forward + backward + optimizer)
    optimizer.zero_grad(set_to_none=True)

    with nvtx.range("forward_pass"):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

    with nvtx.range("backward_pass"):
        loss.backward()

    # 修复原脚本里对 mode 判断的 bug
    if mode == "train":
        with nvtx.range("optimizer_step"):
            optimizer.step()


def benchmark(args):
    device = torch.device(args.device)

    model = build_model(args, device)
    model.train()

    inputs = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    # 1. Warm-up steps（不打 profile 标记，让 GPU 预热、分配好显存缓存）
    for _ in range(args.warmup_steps):
        run_step(model, optimizer, inputs, targets, args.mode)
        synchronize(device)

    timings = []

    # 2. Measurement steps
    # 打上 "benchmark_run" 标记，配合 nsys --capture-range=nvtx 过滤掉 warmup
    nvtx.range_push("benchmark_run")
    for _ in range(args.steps):
        synchronize(device)
        start = time.perf_counter()

        run_step(model, optimizer, inputs, targets, args.mode)

        synchronize(device)
        end = time.perf_counter()

        timings.append(end - start)

    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0

    print(f"mode: {args.mode}")
    print(f"device: {device}")
    print(f"warmup steps: {args.warmup_steps}")
    print(f"measurement steps: {args.steps}")
    print(f"mean: {mean * 1000:.3f} ms")
    print(f"std:  {std * 1000:.3f} ms")

    nvtx.range_pop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["forward", "backward", "train"],
        default="train",
        help="forward only, forward+backward, or forward+backward+optimizer",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    # 做 nsys profile 时，steps 设小一点（比如 1 到 3），避免生成的报告过大
    parser.add_argument("--steps", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=10000)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())