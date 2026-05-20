#!/usr/bin/env bash
# Evaluate SmolLM2-360M-Instruct checkpoints (per-checkpoint launches)

set -euo pipefail

MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCS=8
BATCH_SIZE=32
RESULTS="results/pubmed_eval_results_ff.json"
SCRIPT="scripts/eval_pubmed_accelerate.py"

# -- Adapter paths --
GRPO_COUNTDOWN1="logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853"
GRPO_COUNTDOWN2="logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840"
SFT_PUBMED1="logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853][v15]trl-sft-20260424-233216"
SFT_PUBMED2="logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840][v15]trl-sft-20260425-053058"
GRPO_PUBMED1="logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853]trl-grpo-20260424-042637"
GRPO_PUBMED2="logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840]trl-grpo-20260425-001819"

# -- Checkpoint lists --
COUNTDOWN_CKPTS=(100 200 240 280 320 360 400 500 600 700 800 900 1000)
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

# echo "============================================================"
# echo "1/4  Base"
# echo "============================================================"
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
#     --num_processes $NUM_PROCS --multi_gpu \
#     $SCRIPT \
#     --model-name "$MODEL" \
#     --batch-size $BATCH_SIZE \
#     --save-results "$RESULTS"

# echo "============================================================"
# echo "2/4  GRPO on Countdown"
# echo "============================================================"
# for ckpt in "${COUNTDOWN_CKPTS[@]}"; do
#     echo "--- GRPO_COUNTDOWN1 checkpoint-$ckpt ---"
#     run_eval "$GRPO_COUNTDOWN1/checkpoint-$ckpt"
# done

# for ckpt in "${COUNTDOWN_CKPTS[@]}"; do
#     echo "--- GRPO_COUNTDOWN2 checkpoint-$ckpt ---"
#     run_eval "$GRPO_COUNTDOWN2/checkpoint-$ckpt"
# done

echo "============================================================"
echo "3/4  SFT on PubMedQA  (pre-adapter: GRPO-countdown)"
echo "============================================================"
for ckpt in "${PUBMED_CKPTS[@]}"; do
    echo "--- SFT_PUBMED1 checkpoint-$ckpt ---"
    run_eval "$SFT_PUBMED1/checkpoint-$ckpt" --pre-adapters "$GRPO_COUNTDOWN1"
done

for ckpt in "${PUBMED_CKPTS[@]}"; do
    echo "--- SFT_PUBMED2 checkpoint-$ckpt ---"
    run_eval "$SFT_PUBMED2/checkpoint-$ckpt" --pre-adapters "$GRPO_COUNTDOWN2"
done

echo "============================================================"
echo "4/4  GRPO on PubMedQA  (pre-adapter: GRPO-countdown)"
echo "============================================================"
for ckpt in "${PUBMED_CKPTS[@]}"; do
    echo "--- GRPO_PUBMED1 checkpoint-$ckpt ---"
    run_eval "$GRPO_PUBMED1/checkpoint-$ckpt" --pre-adapters "$GRPO_COUNTDOWN1"
done

for ckpt in "${PUBMED_CKPTS[@]}"; do
    echo "--- GRPO_PUBMED2 checkpoint-$ckpt ---"
    run_eval "$GRPO_PUBMED2/checkpoint-$ckpt" --pre-adapters "$GRPO_COUNTDOWN2"
done

echo ""
echo "All done! Results in: $RESULTS"
