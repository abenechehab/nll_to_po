#!/usr/bin/env bash
# Evaluate SmolLM2-360M-Instruct checkpoints on Countdown across 4 training stages.
#
# Each stage's pre-adapter is the single adapter that was merged at training time:
#   SFT-countdown:  base (no pre-adapter)
#   GRPO-countdown: base + merge SFT-countdown
#   SFT-pubmedqa:   base + merge GRPO-countdown
#   GRPO-pubmedqa:  base + merge GRPO-countdown

set -euo pipefail

MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCS=8
BATCH_SIZE=128
RESULTS="results/countdown_eval_results2.json"

# -- Adapter paths --
SFT_COUNTDOWN="logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[peft][v13]trl-sft-20260413-151207"
GRPO_COUNTDOWN="logs/SmolLM2-360M-Instruct/Countdown-Task-GOLD/[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14][adapter-[peft][v13]trl-sft-20260413-151207]trl-grpo-20260415-104358"
SFT_PUBMED="logs/SmolLM2-360M-Instruct/pubmed_qa/[peft][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14][adapter-[peft][v13]trl-sft-20260413-151207]trl-grpo-20260415-104358][v15]trl-sft-20260416-123225"
GRPO_PUBMED="logs/SmolLM2-360M-Instruct/pubmed_qa/[pubmedbert-base-embeddings-8M][Id][cls][0.001][peft_8][s:400][v15][adapter-[embeddinggemma-300m][Id][cls][0.001][peft_8][s:1000][v14][adapter-[peft][v13]trl-sft-20260413-151207]trl-grpo-20260415-104358]trl-grpo-20260416-123508"

echo "============================================================"
echo "1/4  SFT on Countdown  (no pre-adapters)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    scripts/eval_countdown.py \
    --model-name "$MODEL" \
    --adapter-base-path "$SFT_COUNTDOWN" \
    --checkpoints "20,60,100,140,180,220,260,300,400" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo "============================================================"
echo "2/4  GRPO on Countdown  (pre-adapter: SFT-countdown)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    scripts/eval_countdown.py \
    --model-name "$MODEL" \
    --adapter-base-path "$GRPO_COUNTDOWN" \
    --checkpoints "100,200,240,280,320,360,400,500,600,700,800,900,1000" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo "============================================================"
echo "3/4  SFT on PubMedQA  (pre-adapter: GRPO-countdown)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    scripts/eval_countdown.py \
    --model-name "$MODEL" \
    --adapter-base-path "$SFT_PUBMED" \
    --checkpoints "20,40,60,80,100,200,400" \
    --pre-adapters "$GRPO_COUNTDOWN" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo "============================================================"
echo "4/4  GRPO on PubMedQA  (pre-adapter: GRPO-countdown)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
    --num_processes $NUM_PROCS --multi_gpu \
    scripts/eval_countdown.py \
    --model-name "$MODEL" \
    --adapter-base-path "$GRPO_PUBMED" \
    --checkpoints "20,40,60,80,100,200,400" \
    --pre-adapters "$GRPO_COUNTDOWN" \
    --batch-size $BATCH_SIZE \
    --save-results "$RESULTS"

echo ""
echo "All done! Results in: $RESULTS"