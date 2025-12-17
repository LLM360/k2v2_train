#!/bin/bash
#SBATCH --job-name=k2gem-70b
#SBATCH --nodes=400
#SBATCH --ntasks=400
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --output=/lustrefs/users/runner/slurm/final-%x_%j.out
#SBATCH --error=/lustrefs/users/runner/slurm/final-%x_%j.err
#SBATCH --partition=higherprio
#SBATCH --distribution=block:block

### Increase the send queue depth and can turn NCCL communications into non-blocking.
### https://www.usenix.org/system/files/atc23-choi.pdf
# export NCCL_BUFFSIZE=8388608
### Improve performance by increasing buffer size for Send/Recv, Gather, Scatter and Alltoall communications
### https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
# export NCCL_P2P_NET_CHUNKSIZE=524288

# If reserved but unallocated memory is large, trying to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH \
       UCX_TLS=dc \
       UCX_NET_DEVICES=mlx5_ib0:1 \
       CUDA_DEVICE_ORDER=PCI_BUS_ID \
       NCCL_SOCKET_IFNAME=eth0 \
       NCCL_DEBUG=WARN \
       NCCL_NET_GDR_LEVEL=5 \
       NCCL_MIN_NCHANNELS=32 \
       NCCL_TOPO_FILE=/mnt/users/runner/scripts/ndv5-topo.xml \
       OMPI_MCA_coll_hcoll_enable=0 \
       OMPI_MCA_plm_rsh_no_tree_spawn=1 \
       OMPI_MCA_plm_rsh_num_concurrent=800
       NCCL_IB_QPS_PER_CONNECTION=4 \
       NCCL_P2P_NET_CHUNKSIZE=$((512*1024)) \
       NCCL_PXN_DISABLE=1

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

# EDIT the session below to fit your local paths.
CHECKPOINT_PATH=/mnt/users/runner/checkpoints/yolo/k2gem_250k_dense70_1200bsz_400nodes
WORK_DIR=/mnt/users/runner/workspace/code/K2V2_TRAIN
TOKENIZER_TYPE=HuggingFaceTokenizer
# Tokenizer can be obtained here: https://huggingface.co/LLM360/K2-V2/tree/main/250k
TOKENIZER_MODEL=/mnt/shared/initial-dataset/250k
# The dataset can be obtained here: https://huggingface.co/datasets/cerebras/SlimPajama-627B 
EVAL_DATA_PATH=/mnt/shared/initial-dataset/merged_stripped
CALC_WEIGHT_SCRIPT=/lustrefs/users/runner/workspace/code/K2V2_TRAIN/tools/calc_dataset_weights.py
DATA_BANK_PATH=/lustrefs/users/runner/workspace/code/K2V2_TRAIN/data_banks/final_opencoder_v5_databanks.json
PYTHON_PATH=/lustrefs/users/runner/miniconda3/bin/python

export IMAGE=/lustrefs/shared/megatron-images/pt_24.10_te_1.13.sqsh
export NVME_IMAGE=/mnt/pt_24.10_te_1.13.sqsh

date
srun -l rsync -rvlogtP $IMAGE $NVME_IMAGE
echo "finished cp image"
date

TRAIN_DATA=$($PYTHON_PATH $CALC_WEIGHT_SCRIPT $DATA_BANK_PATH)
echo $TRAIN_DATA

VALID_DATA="\
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk1
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk2
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk3
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk4
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk5"

TEST_DATA="\
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk1
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk2
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk3
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk4
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk5"

DATA_CACHE_DIR=${WORK_DIR}/data_cache/${SLURM_JOB_NUM_NODES}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --rdzv_id $RANDOM
    --rdzv_backend c10d
    --rdzv_endpoint $head_node_ip:29500
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 32768
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --init-method-std 0.01  # 1/sqrt(8192)
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 500000 # change with seq len
    --swiglu
    --untie-embeddings-and-output-weights
    # --no-masked-softmax-fusion # no use with flash-attn
    --no-position-embedding
    --apply-layernorm-1p
)


DATA_ARGS=(
    --tokenizer-type ${TOKENIZER_TYPE}
    --tokenizer-model ${TOKENIZER_MODEL}
    --train-data-path ${TRAIN_DATA}
    --valid-data-path ${VALID_DATA}
    --test-data-path ${TEST_DATA}
    --data-cache-path ${DATA_CACHE_DIR}
    --no-mmap-bin-files
    --num-workers 2
)


TRAINING_ARGS=(
    --micro-batch-size 3 # 2 # 3584GPUs
    --global-batch-size 1200  # 1792
    --seed 42
    --lr 1.5e-4
    --adam-beta1 0.9
    --adam-beta2 0.95
    --train-iters 1250000
    --lr-decay-iters 1250000 # train-iters * 1.0
    --lr-decay-style cosine
    --min-lr 0.0 # lr decay directly to zero
    --lr-warmup-iters 50 # test run
    --weight-decay 0.05
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
    # --recompute-activations
    --recompute-granularity 'full'
    --recompute-method 'uniform'
    --recompute-num-layers 1
    --attention-backend flash
)


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-timeout-minutes 60
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 500
    --log-throughput
    --log-memory-to-tensorboard
    --log-params-norm
    --log-timers-to-tensorboard
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --logging-level 20
)

CHECKPOINT_ARGS=(
    --eval-interval 500
    --eval-iters 51
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --ckpt-format torch_dist
    # --async-save
    --ckpt-fully-parallel-load # To be tested
    --use-persistent-ckpt-worker
    --ckpt-assume-constant-structure
)

srun --container-mounts="/lustrefs:/mnt,/var/tmp:/var/tmp,/opt/microsoft:/opt/microsoft" \
    --container-image=${NVME_IMAGE} \
    torchrun ${DISTRIBUTED_ARGS[@]} $WORK_DIR/pretrain_llm360.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${LOGGING_ARGS[@]}

