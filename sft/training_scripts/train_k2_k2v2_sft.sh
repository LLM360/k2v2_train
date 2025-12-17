#!/bin/bash
#SBATCH --job-name=k2-0085000-verl-k2v2-sft-IFM
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --output=./stdout/%x_%j.out
#SBATCH --error=./stdout/%x_%j.err
#SBATCH --qos=iq
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=ALL

WEBHOOK_URL="https://hooks.slack.com/your/webhook/url"

# Fail fast so we can alert on non-zero exit
set -euo pipefail

# Send a one-line Slack message only if the job exits non-zero
trap 'rc=$?;
if [[ $rc -ne 0 ]]; then
  msg="Job $SLURM_JOB_NAME ($SLURM_JOB_ID) FAILED with code $rc on $(hostname)"
  payload=$(printf "{\"text\":\"%s\"}" "$msg")
  curl -sS -X POST -H "Content-Type: application/json" -d "$payload" "$WEBHOOK_URL" || true
fi' EXIT

# Prepare environment
source /path/to/your/conda.sh
conda activate your_env

# Get node info
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
MASTER_NODE_IP=${nodes[0]}
# export WANDB_API_KEY=your_wandb_api_key
export WANDB_API_KEY=your_wandb_api_key

cd /path/to/your/verl

# --- Custom chat template (Jinja) ---
CUSTOM_TEMPLATE=$(cat <<'JINJA'
{% set system_message = 'You are a helpful assistant. To answer the user\'s question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.' %}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}{% if system_message is defined %}{{ '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}{% endif %}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>\n' }}{% endif %}{% endfor %}
JINJA
)
# strip the single trailing newline heredocs add
CUSTOM_TEMPLATE=${CUSTOM_TEMPLATE%$'\n'}
# (Your template is one physical line, so you don't need to convert newlines)
export CUSTOM_TEMPLATE
# ------------------------------------

# Launch distributed training
srun --cpu-bind=none \
  bash -c '
    set -euo pipefail
    HF_HOME=/path/to/your/hf_cache/hf_cache_$(hostname)_$SLURM_PROCID
    TRITON_HOME=/tmp/triton_cache
    HYDRA_FULL_ERROR=1
    mkdir -p $HF_HOME
    mkdir -p $TRITON_HOME
    export HF_HOME
    export TRITON_HOME
    export HYDRA_FULL_ERROR
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_PROCID --master_addr='$MASTER_NODE_IP' --master_port=29500 \
      -m verl.trainer.fsdp_sft_trainer \
      data.train_files=/path/to/your/data.parquet \
      data.val_files=null \
      data.multiturn.enable=true \
      data.multiturn.messages_key=conversations \
      data.chat_template=null \
      data.max_length=32768 \
      data.truncation=right \
      model.use_liger=true \
      model.partial_pretrain=/path/to/your/model \
      model.custom_chat_template='\''${oc.env:CUSTOM_TEMPLATE}'\'' \
      model.trust_remote_code=true \
      model.strategy=fsdp2 \
      data.micro_batch_size_per_gpu=1 \
      data.train_batch_size=1024 \
      optim.lr=2e-5 \
      optim.lr_scheduler=cosine \
      optim.warmup_steps_ratio=0.05 \
      optim.clip_grad=0.5 \
      trainer.default_local_dir=/path/to/your/output \
      trainer.project_name=k2-verl-sft \
      trainer.experiment_name=k2-0085000-verl-k2v2-sft-IFM_64_nodes \
      trainer.total_epochs=2 \
      trainer.logger=[console,wandb] \
      trainer.save_freq=200 \
      trainer.test_freq=-1 \
      ulysses_sequence_parallel_size=2 \
      use_remove_padding=true
  ' 