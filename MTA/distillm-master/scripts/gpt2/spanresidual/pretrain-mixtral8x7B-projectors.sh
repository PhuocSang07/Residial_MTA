#!/bin/bash
# Stage 1 — Projector pretraining: P_T->A and P_A->T for Mixtral-8x7B-Instruct teacher.
# Teacher hidden size: d_T=4096, bottleneck d_A=64.
# Paper: On et al. ICLR 2026 — Stage 1: epochs=10, lr=1e-3, wd=1e-4, cosine.
# Requires: processed_data/dolly/full/mixtral/ (run process_data_dolly_mixtral.sh first)

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=67$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=./distillm-master

# Teacher: Mixtral-8x7B-Instruct (d_T=4096, 32 layers)
TEACHER_CKPT="mistralai/Mixtral-8x7B-Instruct-v0.1"
TEACHER_CKPT_NAME="mixtral-8x7B-instruct"

# Data: Mixtral-tokenised Dolly
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/mixtral/"

# Paper Stage 1 hyperparameters
BATCH_SIZE=8           # reduced from 16 (Mixtral activations are much larger)
EVAL_BATCH_SIZE=8
GRAD_ACC=2             # effective 16
D_BOTTLENECK=64        # anchor dim d_A (paper)
PROJECTOR_EPOCHS=10
PROJECTOR_LR=1e-3

SAVE_PATH="${BASE_PATH}/results/mixtral/projectors/spanresidual_mixtral8x7B"
SEED=42

OPTS=""
OPTS+=" --model-path ${TEACHER_CKPT}"
OPTS+=" --model-type mistral"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
OPTS+=" --train-num -1"
OPTS+=" --train-ratio 1"
OPTS+=" --dev-ratio 1"
OPTS+=" --lr ${PROJECTOR_LR}"
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-pretrain-epochs ${PROJECTOR_EPOCHS}"
OPTS+=" --d-bottleneck ${D_BOTTLENECK}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --weight-decay 1e-4"
OPTS+=" --clip-grad 1.0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-min 1e-6"
OPTS+=" --max-length 256"
OPTS+=" --max-prompt-length 128"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --type projector-pretrain"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 50"
OPTS+=" --mid-log-num -1"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_residual_pretrain.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
