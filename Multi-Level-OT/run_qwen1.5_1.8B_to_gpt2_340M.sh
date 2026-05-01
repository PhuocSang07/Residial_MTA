#!/bin/bash
# Distill VoCuc/Qwen1.5_1.8B_SFT_Dolly (teacher, 2048-dim, 24 layers)
#        → openai-community/gpt2-medium (student, 1024-dim, 24 layers)
# MultiLevelOT (cross-tokenizer) + MTA Span + Entropy Weight
#
# Hyperparameters theo paper (AAAI 2025):
#   lr=1e-6, distil_factor=0.15 (α), f=1 (sequence-level ranking + top-50)
#
# Layer mapping (6 pairs, split_layer_mapping "0,1,6"):
#   MTA-style: last 6 layers step=2 (both 24L → 1:1) — mirrors MTA Qwen1.5 exactly
#   Word spans  (lower):  student 14 ↔ teacher 14
#   Phrase spans (higher): student 16,18,20,22,24 ↔ teacher 16,18,20,22,24

export CUDA_VISIBLE_DEVICES=0
export DS_IGNORE_CUDA_DETECTION=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dolly data: train.jsonl / valid.jsonl từ distillm-master
export DOLLY_DATA_DIR="$SCRIPT_DIR/../MTA/distillm-master/processed_data/dolly/full/gpt2"

python finetuning.py \
  --model_name openai-community/gpt2-medium \
  --dataset.file "$SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py" \
  --lr 5e-4 \
  --num_epochs 20 \
  --batch_size_training 4 \
  --val_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --output_dir "$SCRIPT_DIR/output/qwen1.5-1.8B-to-gpt2-340M" \
  --distillation \
  --distillation_config_model_name VoCuc/Qwen1.5_1.8B_SFT_Dolly \
  --distillation_config_distil_factor 0.15 \
  --distillation_config_cross_entropy_factor 1.0 \
  --distillation_config_student_temperature 1.0 \
  --distillation_config_teacher_temperature 2.0 \
  --save_step 4500 \
  --f 1 \
  --span_loss_weight 0.1 \
  --entropy_weight \
  --student_layer_mapping "14,16,18,20,22,24" \
  --teacher_layer_mapping "14,16,18,20,22,24" \
  --split_layer_mapping "0,1,6" \
  --use_phrase_spans \
  --context_length 1024 \
  --student_hidden_size 1024 \
  --teacher_hidden_size 2048
