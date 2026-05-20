#!/usr/bin/env bash
# Evaluate SmolLM2-360M-Instruct checkpoints

set -euo pipefail

MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCS=8
BATCH_SIZE=128
RESULTS="results/countdown_eval_results_f_all.json"
SCRIPT="scripts/eval_countdown.py"

# -- Adapter paths --
GRPO_COUNTDOWN1="logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853"
GRPO_COUNTDOWN2="logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840"
SFT_PUBMED1="logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853][v15]trl-sft-20260424-233216"
SFT_PUBMED2="logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840][v15]trl-sft-20260425-053058"
GRPO_PUBMED1="logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-110853]trl-grpo-20260424-042637"
GRPO_PUBMED2="logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14]trl-grpo-20260423-203840]trl-grpo-20260425-001819"

# echo "============================================================"
# echo "1/4  Base"
# echo "============================================================"
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
#     --num_processes $NUM_PROCS --multi_gpu \
#     $SCRIPT \
#     --model-name "$MODEL" \
#     --batch-size $BATCH_SIZE \
#     --save-results "$RESULTS"

echo "============================================================"
echo "2/4  GRPO on Countdown"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    $SCRIPT \
    --model-name "$MODEL" \
    --adapter-base-path "$GRPO_COUNTDOWN1" \
    --checkpoints "100,200,240,280,320,360,400,500,600,700,800,900,1000" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    $SCRIPT \
    --model-name "$MODEL" \
    --adapter-base-path "$GRPO_COUNTDOWN2" \
    --checkpoints "100,200,240,280,320,360,400,500,600,700,800,900,1000" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo "============================================================"
echo "3/4  SFT on PubMedQA  (pre-adapter: GRPO-countdown)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    $SCRIPT \
    --model-name "$MODEL" \
    --adapter-base-path "$SFT_PUBMED1" \
    --checkpoints "20,40,60,80,100,200,400" \
    --pre-adapters "$GRPO_COUNTDOWN1" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    $SCRIPT \
    --model-name "$MODEL" \
    --adapter-base-path "$SFT_PUBMED2" \
    --checkpoints "20,40,60,80,100,200,400" \
    --pre-adapters "$GRPO_COUNTDOWN2" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo "============================================================"
echo "4/4  GRPO on PubMedQA  (pre-adapter: GRPO-countdown)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    $SCRIPT \
    --model-name "$MODEL" \
    --adapter-base-path "$GRPO_PUBMED1" \
    --checkpoints "20,40,60,80,100,200,400" \
    --pre-adapters "$GRPO_COUNTDOWN1" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    $SCRIPT \
    --model-name "$MODEL" \
    --adapter-base-path "$GRPO_PUBMED2" \
    --checkpoints "20,40,60,80,100,200,400" \
    --pre-adapters "$GRPO_COUNTDOWN2" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo ""
echo "All done! Results in: $RESULTS"

