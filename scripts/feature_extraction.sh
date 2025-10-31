#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=${master_addr:=$ARNOLD_WORKER_0_HOST}
MASTER_PORT=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
NPUS_PER_NODE=$ARNOLD_WORKER_GPU
NNODES=$ARNOLD_WORKER_NUM
NODE_RANK=$ARNOLD_ID
trial_id=$ARNOLD_TRIAL_ID

echo MASTER_ADDR $MASTER_ADDR
echo MASTER_PORT $MASTER_PORT
echo NPUS_PER_NODE $NPUS_PER_NODE
echo NNODES $NNODES
echo NODE_RANK $NODE_RANK
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP))

MM_DATA=${MM_DATA:-'configs/feature_extract/data.json'}
MM_MODEL=${MM_MODEL:-'configs/feature_extract/model.json'}
MM_TOOL=${MM_TOOL:-'configs/feature_extract/tools.json'}
LOAD_PATH="/pretrained_model/mm_dir/Qwen2.5-VL-7B-Instruct"
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-layers 1 \
    --num-workers 8 \
    --hidden-size 3072 \
    --num-attention-heads 48 \
    --seq-length 24 \
    --max-position-embeddings 24 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --load $LOAD_PATH \
    --bf16 \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

export PYTHONPATH=$PYTHONPATH:`pwd`/MindSpeed-MM:`pwd`/MindSpeed
torchrun $DISTRIBUTED_ARGS s2v/tools/feature_extract/get_sora_feature.py \
    $GPT_ARGS \
    $MM_ARGS \
    --distributed-backend nccl

