#!/bin/bash
# Setup C — SpanResidual KD (with MTA): Mixtral-8x7B-Instruct -> GPT2-120M
# Loss: L = (1 - lambda_res)*L_SFT + lambda_res*L_res + gamma_span*L_SpanMTA
# Paper Table 4: cross-tokenizer, On et al. ICLR 2026 (target: Rouge-L ~22.62 on Dolly)
# Requires: projector_latest.pt from pretrain-mixtral8x7B-projectors.sh

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=68$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=./distillm-master

# Student: GPT2-120M (d_S=768, 12 layers)
CKPT_NAME="gpt2-base"
CKPT="openai-community/gpt2"

# Teacher: Mixtral-8x7B-Instruct (d_T=4096, 32 layers)
TEACHER_CKPT="mistralai/Mixtral-8x7B-Instruct-v0.1"
TEACHER_CKPT_NAME="mixtral-8x7B-instruct"

# Stage-1 projector checkpoint (P_T->A: 4096->64)
PROJECTOR_PATH="${BASE_PATH}/results/mixtral/projectors/spanresidual_mixtral8x7B/projector_latest.pt"

# Data (cross-tokenizer)
STUDENT_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
TEACHER_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/mixtral/"

# Paper Stage 2 hyperparameters
BATCH_SIZE=16
LR=1e-3
GRAD_ACC=8          # 16 * 8 = 128 global batch
EVAL_BATCH_SIZE=16
EPOCHS=10
MAX_LENGTH=256

LAMBDA_RES=0.5
GAMMA_SPAN=1.0
W_SPAN_LOSS=2.0

SAVE_PATH="${BASE_PATH}/results/gpt2/train/spanresidual_setup_C_0.1B_mixtral8x7B"
SEED=42

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type gpt2"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-type mistral"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --projector-load-path ${PROJECTOR_PATH}"
OPTS+=" --d-bottleneck 64"
OPTS+=" --lambda-res ${LAMBDA_RES}"
OPTS+=" --gamma-span ${GAMMA_SPAN}"
OPTS+=" --data-dir ${STUDENT_DATA_DIR}"
OPTS+=" --teacher-data-dir ${TEACHER_DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
OPTS+=" --warmup-ratio 0.1"
OPTS+=" --w-span-loss ${W_SPAN_LOSS}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 128"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --type adaptive-srkl"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
OPTS+=" --student-gen"
# Layer mappings: Mixtral 32 layers -> GPT2 12 layers
OPTS+=" --teacher_layer_mapping 10 21 32"
OPTS+=" --student_layer_mapping 4 8 12"
OPTS+=" --split_layer_mapping 0 1 3 3"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_residual_finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
