#!/bin/bash
#SBATCH --job-name=k2v2-stage3-rope10m
#SBATCH --nodes=200
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-task=8
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_logs/stage3_rope10m-%j.out
#SBATCH --error=slurm_logs/stage3_rope10m-%j.err
#SBATCH --partition=higherprio
#SBATCH --exclusive
#SBATCH --distribution=block:block

### Network and NCCL Optimization
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=$((512 * 1024))
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH \
       UCX_TLS=dc \
       UCX_NET_DEVICES=mlx5_ib0:1 \
       CUDA_DEVICE_ORDER=PCI_BUS_ID \
       NCCL_SOCKET_IFNAME=eth0 \
       NCCL_DEBUG=WARN \
       NCCL_NET_GDR_LEVEL=5 \
       NCCL_MIN_NCHANNELS=32 \
       NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
       OMPI_MCA_coll_hcoll_enable=0 \
       OMPI_MCA_plm_rsh_no_tree_spawn=1 \
       OMPI_MCA_plm_rsh_num_concurrent=800 \
       NCCL_IB_QPS_PER_CONNECTION=4 \
       NCCL_PXN_DISABLE=1 \
       NCCL_IB_TIMEOUT=22

export UCX_NET_DEVICES=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NNODES=200

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export TRITON_CACHE_DIR="/tmp/triton-cache"

echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
TP=8
SAVE_ROOT="${YOUR_CKPT_PATH}/k2v2_stage3_attn128k_jais250k_rope10m_tp${TP}_bestfit"
CHECKPOINT_PATH="${SAVE_ROOT}/checkpoints"
TENSORBOARD_DIR="${SAVE_ROOT}/tensorboard"
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_DIR}

# Stage 3 Merged Data
TRAIN_DATA_PATH="${YOUR_DATA_PATH}/midtrain_stage3/merged"

TOKENIZER_TYPE=HuggingFaceTokenizer
TOKENIZER_MODEL="${YOUR_TOKENIZER_PATH}/tokenizers/jais250k"

# Load from Stage 2 Checkpoint (Iteration 45000)
LOAD_CHECKPOINT_PATH="${YOUR_CKPT_PATH}/k2v2_stage2_attn64k_jais250k_tp8_bestfit_fix/checkpoints/checkpoint_0045000"

export IMAGE=/lustrefs/shared/megatron-images/pt_24.10_te_1.13.sqsh
export NVME_IMAGE=/mnt/pt_24.10_te_1.13.sqsh

date
srun -l rsync -rvlogtP $IMAGE $NVME_IMAGE
echo "finished cp image"
date

VALID_DATA_PATH="${YOUR_DATA_PATH}/SlimPajama-627B/validation/Slimpajama.jsonl"

DATA_CACHE_DIR=${SAVE_ROOT}/data_cache
mkdir -p ${DATA_CACHE_DIR}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --rdzv_id $RANDOM
    --rdzv_backend c10d
    --rdzv_endpoint $head_node_ip:29500
)

# K2V2 Architecture Configuration (Stage 3: 128k Context, RoPE 10M)
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 131072
    --max-position-embeddings 131072
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 10000000  # 10M RoPE Base
    --swiglu
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --apply-layernorm-1p
)

DATA_ARGS=(
    --tokenizer-type ${TOKENIZER_TYPE}
    --tokenizer-model ${TOKENIZER_MODEL}
    --train-data-path ${TRAIN_DATA_PATH}
    --valid-data-path ${VALID_DATA_PATH}
    --data-cache-path ${DATA_CACHE_DIR}
    --no-mmap-bin-files
    --num-workers 2
    --split 100,0,0
)

# Training Hyperparameters (Stage 3: LR 6e-6, Constant, 17.5k steps)
TRAINING_ARGS=(
    --micro-batch-size 1
    # Global Batch Size Calculation:
    # 1600 GPUs / (TP 8 * CP 2 * PP 1) = 100 DP replicas.
    # 100 DP * 1 MBS = 100 GBS.
    --global-batch-size 100
    --seed 2
    --lr 6.0e-6
    --adam-beta1 0.9
    --adam-beta2 0.95
    --train-iters 17500
    --lr-decay-style constant
    --lr-warmup-iters 0
    --weight-decay 0.05
    --clip-grad 1.0
    --overlap-grad-reduce
    --overlap-param-gather
    --recompute-granularity 'full'
    --recompute-method 'uniform'
    --recompute-num-layers 1
    --attention-backend flash
    --bf16
)

# Parallelism: TP=8, CP=2
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --context-parallel-size 2  # Added CP for 128k sequence
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-timeout-minutes 60
)

LOGGING_ARGS=(
    --log-interval 10
    --save-interval 250
    --log-throughput
    --log-params-norm
    --tensorboard-dir ${TENSORBOARD_DIR}
    --wandb-project "k2v2_stage3"
    --wandb-exp-name "stage3-rope10m"
    --logging-level 20
)

CHECKPOINT_ARGS=(
    --eval-interval 2500
    --eval-iters 20
    --save $CHECKPOINT_PATH
    --load $LOAD_CHECKPOINT_PATH
    --ckpt-format torch_dist
    --ckpt-fully-parallel-load
    --use-persistent-ckpt-worker
)

srun --container-mounts="/lustrefs:/lustrefs,/var/tmp:/var/tmp,/opt/microsoft:/opt/microsoft" \
    --container-image=${NVME_IMAGE} \
    torchrun ${DISTRIBUTED_ARGS[@]} /workspace/megatron-lm/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${LOGGING_ARGS[@]}