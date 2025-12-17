#!/bin/bash
#SBATCH --job-name=k2v2-stage1
#SBATCH --nodes=400
#SBATCH --ntasks=400
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-task=8
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_logs/final-%x_%j.out
#SBATCH --error=slurm_logs/final-%x_%j.err
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
NNODES=400

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
SAVE_ROOT="${YOUR_CKPT_PATH}/k2v2_stage1_attn8k_jais250k_tp${TP}"
CHECKPOINT_PATH="${SAVE_ROOT}/checkpoints"
TENSORBOARD_DIR="${SAVE_ROOT}/tensorboard"
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_DIR}

RAW_DATA="${YOUR_DATA_PATH}/megamath/code:0.3717964534,${YOUR_DATA_PATH}/megamath/web-pro:0.2040901021,${YOUR_DATA_PATH}/megamath/web:3.4280315197,${YOUR_DATA_PATH}/wiki+/stackexchange:1.0744576405,${YOUR_DATA_PATH}/wiki+/hackernews:0.0517331457,${YOUR_DATA_PATH}/wiki+/wikipedia_extended:5.8555098927,${YOUR_DATA_PATH}/institutional-books-1.0:10.0055314671,${YOUR_DATA_PATH}/long_context/ubuntu_irc:0.0364406004,${YOUR_DATA_PATH}/long_context/cosmopedia_ultraTextbooks:0.8129494317,${YOUR_DATA_PATH}/long_context/pg19:0.0451954404,${YOUR_DATA_PATH}/papers/arxiv:0.4007897548,${YOUR_DATA_PATH}/papers/s2orc:1.8589822669,${YOUR_DATA_PATH}/papers/phil_papers:0.0142692523,${YOUR_DATA_PATH}/papers/pubmed:0.8356892759,${YOUR_DATA_PATH}/reasoning:0.8015795096,${YOUR_DATA_PATH}/common_pile_v0.1:1.3985004210,${YOUR_DATA_PATH}/reasoning_v1:0.8015795096,${YOUR_DATA_PATH}/math:23.9905356768,${YOUR_DATA_PATH}/ai:0.8015795096,${YOUR_DATA_PATH}/general:0.4002212587,${YOUR_DATA_PATH}/arabic:2.0011062934,${YOUR_DATA_PATH}/planning:0.8015795096,${YOUR_DATA_PATH}/TxT360-QA/cc_others:0.6651404441,${YOUR_DATA_PATH}/TxT360-QA/cc_6-10:0.4820846980,${YOUR_DATA_PATH}/TxT360-QA/cc_1-1:1.5406244475,${YOUR_DATA_PATH}/TxT360-QA/cc_2-5:1.3132260051,${YOUR_DATA_PATH}/open_coder/topo_sorted:0.4906121396,${YOUR_DATA_PATH}/open_coder/original:5.9692091139,${YOUR_DATA_PATH}/open_coder/fim:5.5200971901,${YOUR_DATA_PATH}/weborganizer-sample-2-17:28.0268580300"

TRAIN_DATA=""
IFS=',' read -ra ADDR <<< "$RAW_DATA"
for i in "${ADDR[@]}"; do
    path="${i%:*}"
    weight="${i##*:}"
    TRAIN_DATA+="$weight $path "
done

TOKENIZER_TYPE=HuggingFaceTokenizer
TOKENIZER_MODEL="${YOUR_TOKENIZER_PATH}/jais250k"
LOAD_CHECKPOINT_PATH="${YOUR_CKPT_PATH}/checkpoint_1249000_tp${TP}"

export IMAGE=/lustrefs/shared/megatron-images/pt_24.10_te_1.13.sqsh
export NVME_IMAGE=/mnt/pt_24.10_te_1.13.sqsh

date
srun -l rsync -rvlogtP $IMAGE $NVME_IMAGE
echo "finished cp image"
date

echo "Training Data Configured"

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

# K2V2 Architecture Configuration
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 8192
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
    --rotary-base 500000
    --swiglu
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --apply-layernorm-1p
)

DATA_ARGS=(
    --tokenizer-type ${TOKENIZER_TYPE}
    --tokenizer-model ${TOKENIZER_MODEL}
    --train-data-path ${TRAIN_DATA}
    --valid-data-path ${VALID_DATA_PATH}
    --data-cache-path ${DATA_CACHE_DIR}
    --no-mmap-bin-files
    --num-workers 2
    --split 99,1,0
)

# Training Hyperparameters
TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 1600   # 400 nodes * 4 MBS / 1 CP = 1600
    --seed 2
    --lr 1.5e-5
    --adam-beta1 0.9
    --adam-beta2 0.95
    --train-iters 135000
    --lr-decay-iters 135000
    --lr-decay-style cosine
    --min-lr 6.0e-6            # 1.5e-5 * 0.4 (lr_end_ratio)
    --lr-warmup-iters 500
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

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-timeout-minutes 60
)

LOGGING_ARGS=(
    --log-interval 10
    --save-interval 500
    --log-throughput
    --log-params-norm
    --tensorboard-dir ${TENSORBOARD_DIR}
    --wandb-project "k2v2_mid_train_stage1"
    --wandb-exp-name "stage1-k2v2"
    --logging-level 20
)

CHECKPOINT_ARGS=(
    --eval-interval 2500
    --eval-iters 40
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