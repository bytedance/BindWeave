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
VP=1
CP=2
MBS=1

GRAD_ACC_STEP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

SCRIPTHOME="$( cd "$(dirname "$0")" ; pwd -P )"
CODE_ROOT="$( cd "$(dirname "$0")"/../../../.. ; pwd -P )"
ROOT="$( cd "$(dirname "$0")"/../../../../../.. ; pwd -P )"


if [[ -z "${MM_DATA}" ]]; then
  MM_DATA="${SCRIPTHOME}/configs/pretrain/feature_data.json"
fi

if [[ -z "${MM_MODEL}" ]]; then
  MM_MODEL="${SCRIPTHOME}/pretrain_model.json"
fi

if [[ -z "${MM_TOOL}" ]]; then
  MM_TOOL="${ROOT}/MindSpeed-MM/mindspeed_mm/tools/tools.json"
fi

if [[ -z "${LOAD_PATH}" ]]; then
  LOAD_PATH=pretrained_model/wanx/Wan2.1-I2V-14B-720P-Diffusers/transformer/  # ensure the wandit weight be converted
fi

if [[ -z "${SAVE_PATH}" ]]; then
  SAVE_PATH=./debug_ckpts
fi

if [[ -z "${TRAIN_ITERS}" ]]; then
  TRAIN_ITERS=5000
fi

if [[ -z "${SAVE_INTERVAL}" ]]; then
  SAVE_INTERVAL=1000
fi



echo SAVE_PATH $SAVE_PATH

mkdir -p $SAVE_PATH

layerzero_config="${ROOT}/MindSpeed-MM/examples/wan2.1/zero_config.yaml"

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
    --virtual-pipeline-model-parallel-size ${VP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-workers 16 \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 5e-3 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters ${TRAIN_ITERS} \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 40 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --layerzero \
    --layerzero-config ${layerzero_config} \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval 10000 \
    --eval-iters 10 \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
"


export PYTHONPATH=$PYTHONPATH:${ROOT}/Megatron-LM:${ROOT}/MindSpeed:${ROOT}/MindSpeed-MM:${ROOT}/s2v
echo $PYTHONPATH

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS ${ROOT}/s2v/pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log

chmod 440 logs/train_${logfile}.log
# chmod -R 640 $SAVE_PATH
chmod -R 755 $SAVE_PATH

STEP_TIME=`grep "elapsed time per iteration" logs/train_${logfile}.log | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}'`
SPS=`awk 'BEGIN{printf "%.3f\n", '${GBS}'*1000/'${STEP_TIME}'}'`
echo "Elapsed Time Per iteration: $STEP_TIME, Average Samples per Second: $SPS"