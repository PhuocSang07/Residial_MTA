#!/bin/bash
# Distill VoCuc/Mistral7B_Dolly_SFT (teacher, 4096-dim, 32 layers)
#        → openai-community/gpt2 (student, 768-dim, 12 layers)
# MultiLevelOT (cross-tokenizer) + MTA Span + Entropy Weight
#
# Hyperparameters theo paper (AAAI 2025):
#   lr=1e-6, distil_factor=0.15 (α), f=1 (sequence-level ranking + top-50)
#
# Layer mapping (3 pairs, split_layer_mapping "0,1,3"):
#   MTA-style: 1/2, 3/4, end — focus on latter layers (higher-level semantics)
#   Word spans  (lower):  student layer 6  ↔ teacher layer 16  (index 0)   [1/2]
#   Phrase spans (higher): student layers 9,12 ↔ teacher layers 24,32      [3/4, end]

export CUDA_VISIBLE_DEVICES=1
export DS_IGNORE_CUDA_DETECTION=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dolly data: train.jsonl / valid.jsonl từ distillm-master
export DOLLY_DATA_DIR="$SCRIPT_DIR/../MTA/distillm-master/processed_data/dolly/full/gpt2"

python finetuning.py \
  --model_name openai-community/gpt2 \
  --dataset.file "$SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py" \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 4 \
  --gradient_accumulation_steps 2 \
  --val_batch_size 8 \
  --output_dir "$SCRIPT_DIR/output/mistral-7B-to-gpt2-120M" \
  --distillation \
  --distillation_config_model_name VoCuc/Mistral7B_Dolly_SFT \
  --distillation_config_distil_factor 0.15 \
  --distillation_config_cross_entropy_factor 1.0 \
  --distillation_config_student_temperature 1.0 \
  --distillation_config_teacher_temperature 2.0 \
  --distillation_config_pure_bf16 \
  --save_step 500 \
  --f 1 \
  --span_loss_weight 0.1 \
  --entropy_weight \
  --student_layer_mapping "6,9,12" \
  --teacher_layer_mapping "16,24,32" \
  --split_layer_mapping "0,1,3" \
  --use_phrase_spans \
  --context_length 1024 \
  --student_hidden_size 768 \
  --teacher_hidden_size 4096
