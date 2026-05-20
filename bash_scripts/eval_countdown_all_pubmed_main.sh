#!/usr/bin/env bash
# Evaluate SmolLM2-360M-Instruct checkpoints (per-checkpoint launches)

set -euo pipefail

MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCS=8
BATCH_SIZE=32
RESULTS="results/pubmed_eval_results_ff_grpo_u_star.json"
SCRIPT="scripts/eval_pubmed_accelerate.py"

# -- Adapter paths --
ADAPTERS=(
    "logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][U-star-trace][cls][0.001][peft_8][s:400][v15]trl-grpo-20260506-040147"
    "logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][U-star-trace][cls][0.001][peft_8][s:400][v15]trl-grpo-20260505-205450"
    "logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][U-star-trace][cls][0.001][peft_8][s:400][v15]trl-grpo-20260505-134547"
    "logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][U-star-cov][cls][0.001][peft_8][s:400][v15]trl-grpo-20260506-105858"
    "logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][U-star-cov][cls][0.001][peft_8][s:400][v15]trl-grpo-20260506-175555"
)

# -- Checkpoint lists --
PUBMED_CKPTS=(20 40 60 80 100 200 400)

# -- Helper --
run_eval () {
    local adapter_path="$1"
    shift
    CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
        --num_processes $NUM_PROCS --multi_gpu \
        $SCRIPT \
        --model-name "$MODEL" \
        --adapter-path "$adapter_path" \
        --batch-size $BATCH_SIZE \
        --save-results "$RESULTS" \
        "$@"
}

# -- Main loop --
for adapter in "${ADAPTERS[@]}"; do
    adapter_name=$(basename "$adapter")

    for ckpt in "${PUBMED_CKPTS[@]}"; do
        echo "--- $adapter_name checkpoint-$ckpt ---"
        run_eval "$adapter/checkpoint-$ckpt"
    done

    echo ""
done

echo "All done! Results in: $RESULTS"