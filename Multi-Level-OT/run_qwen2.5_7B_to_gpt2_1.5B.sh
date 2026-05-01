#!/bin/bash
# Distill VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT (teacher, 3584-dim, 28 layers)
#        → openai-community/gpt2-xl (student, 1600-dim, 48 layers)
# MultiLevelOT (cross-tokenizer) + MTA Span + Entropy Weight
#
# Hyperparameters theo paper (AAAI 2025):
#   lr=1e-6, distil_factor=0.15 (α), f=1 (sequence-level ranking + top-50)
#
# Layer mapping (6 pairs, split_layer_mapping "0,1,6"):
#   MTA-style: latter half evenly spaced (student 48L step~4.8, teacher 28L step~2.8)
#   Word spans  (lower):  student 24 ↔ teacher 14
#   Phrase spans (higher): student 29,34,38,43,48 ↔ teacher 17,20,22,25,28

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dolly data: train.jsonl / valid.jsonl từ distillm-master
export DOLLY_DATA_DIR="$SCRIPT_DIR/../MTA/distillm-master/processed_data/dolly/full/gpt2"

python finetuning.py \
  --model_name openai-community/gpt2-xl \
  --dataset.file "$SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py" \
  --lr 1e-3 \
  --num_epochs 15 \
  --batch_size_training 2 \
  --gradient_accumulation_steps 4 \
  --val_batch_size 8 \
  --output_dir "$SCRIPT_DIR/output/qwen2.5-7B-to-gpt2-1.5B" \
  --distillation \
  --distillation_config_model_name VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT \
  --distillation_config_distil_factor 0.15 \
  --distillation_config_cross_entropy_factor 1.0 \
  --distillation_config_student_temperature 1.0 \
  --distillation_config_teacher_temperature 2.0 \
  --distillation_config_pure_bf16 \
  --save_step 5000 \
  --f 1 \
  --span_loss_weight 0.1 \
  --entropy_weight \
  --student_layer_mapping "24,29,34,38,43,48" \
  --teacher_layer_mapping "14,17,20,22,25,28" \
  --split_layer_mapping "0,1,6" \
  --use_phrase_spans \
  --context_length 1024 \
  --student_hidden_size 1600 \
  --teacher_hidden_size 3584\
  --use_peft \
  --lora_r 256 \
  --lora_alpha 8 \
  --lora_dropout 0.1
