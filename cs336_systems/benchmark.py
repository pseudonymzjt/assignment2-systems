# cs336_systems/benchmark.py
import contextlib
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

def get_autocast_context(mixed_precision: bool, device_type: str = "cuda"):
    if mixed_precision:
        # 使用作业要求的 BF16
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        # 不使用时作为 no-op
        return contextlib.nullcontext()

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

def run_step(model, optimizer, inputs, targets, mode, mixed_precision, annotate=True):
    amp_ctx = get_autocast_context(mixed_precision)

    # 辅助函数：根据 annotate 决定是否发射 NVTX 标记
    def maybe_nvtx(name):
        return nvtx.range(name) if annotate else contextlib.nullcontext()

    if mode == "forward":
        with maybe_nvtx("forward_pass"):
            with torch.no_grad():
                with amp_ctx:
                    model(inputs)
        return

    # mode 为 backward 或 train
    optimizer.zero_grad(set_to_none=True)

    # 1. Forward 阶段
    with maybe_nvtx("forward_pass"):
        with amp_ctx:
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

    # 2. Backward 阶段
    with maybe_nvtx("backward_pass"):
        loss.backward()

    # 3. Optimizer 阶段
    if mode == "train":
        with maybe_nvtx("optimizer_step"):
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

    # 确保 GPU 完全就绪后再进入 warmup
    synchronize(device)

    # 1. Warm-up steps
    # 注意：annotate=False，完全不打 NVTX 标签，确保干净地跑满 warmup 轮次
    for _ in range(args.warmup_steps):
        run_step(
            model,
            optimizer,
            inputs,
            targets,
            args.mode,
            args.mixed_precision,
            annotate=False,
        )
        synchronize(device)

    timings = []

    # 2. Measurement steps
    # 打上 "benchmark_run" 标记，用于 nsys --nvtx-capture 精确捕获
    nvtx.range_push("benchmark_run")
    for _ in range(args.steps):
        synchronize(device)
        start = time.perf_counter()

        # 正式步骤：annotate=True 打上详细的 NVTX 标记
        run_step(
            model,
            optimizer,
            inputs,
            targets,
            args.mode,
            args.mixed_precision,
            annotate=True,
        )

        synchronize(device)
        end = time.perf_counter()

        timings.append(end - start)

    # 统计完成立即 pop
    nvtx.range_pop()

    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0

    print(f"mode: {args.mode}")
    print(f"mixed precision: {args.mixed_precision}")
    print(f"device: {device}")
    print(f"warmup steps: {args.warmup_steps}")
    print(f"measurement steps: {args.steps}")
    print(f"mean: {mean * 1000:.3f} ms")
    print(f"std:  {std * 1000:.3f} ms")
    
MODEL_CONFIGS = {
    "tiny": {
        "d_model": 512,
        "d_ff": 1344,
        "num_layers": 8,
        "num_heads": 16,
    }
    ,
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Benchmarking Script")

    # 1. 运行模式与步骤配置
    parser.add_argument(
        "--mode",
        choices=["forward", "backward", "train"],
        default="train",
        help="forward only, forward+backward, or forward+backward+optimizer",
    )
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--steps", type=int, default=5, help="Number of measurement steps")

    # 2. 数据与批量配置
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--context-length", type=int, default=256, help="Sequence context length")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")

    # 3. 快捷模型预设（来自 Table 1）
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large", "xl", "10B"],
        default=None,
        help="Optional model size preset from Table 1. If set, overrides d-model, num-layers, num-heads, d-ff.",
    )

    # 4. 详细模型架构参数（若不传 --model-size 则使用这套默认值）
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 5. 训练与硬件参数
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision with bfloat16",
    )

    args = parser.parse_args()

    # 如果指定了预设尺寸，覆盖架构维度；未指定的超参（如 lr, vocab_size, context_length）不受影响
    if args.model_size is not None:
        cfg = MODEL_CONFIGS[args.model_size]
        args.d_model = cfg["d_model"]
        args.d_ff = cfg["d_ff"]
        args.num_layers = cfg["num_layers"]
        args.num_heads = cfg["num_heads"]

    return args


if __name__ == "__main__":
    benchmark(parse_args())