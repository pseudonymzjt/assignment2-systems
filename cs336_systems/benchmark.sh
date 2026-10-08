#!/bin/bash
# cs336_systems/benchmark.sh

warmup_steps=(0 1 2 5)

# forward
for wm in "${warmup_steps[@]}"
do
    echo "run with $wm warmup steps"
    uv run python -m cs336_systems.benchmark \
    --mode forward \
    --warmup-steps $wm \
    --steps 10

    # backward
    uv run python -m cs336_systems.benchmark \
    --mode backward \
    --warmup-steps $wm \
    --steps 10

    # train
    uv run python -m cs336_systems.benchmark \
    --mode train \
    --warmup-steps $wm \
    --steps 10
done