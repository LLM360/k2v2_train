# K2V2 SFT

Supervised Fine-Tuning scripts for K2V2. This stage comes after pre-training and before RL.

## Dependencies

This module uses our modified version of [verl](https://github.com/volcengine/verl) for distributed SFT training:

```bash
git clone https://github.com/Ber666/k2v2-sft.git
cd k2v2-sft
pip install -e .
pip install vllm>=0.8.2
```

## Data Preparation

See [`sft_data/`](sft_data/) for dataset preparation:
- Uses [TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts) as base dataset
- `create_sft_mix_3efforts.py` - create weighted SFT mixture
- `mix_3efforts.json` - dataset weights

## Training

1. Update paths in `training_scripts/`:
   - `data.train_files` - training data path
   - `model.partial_pretrain` - pre-trained checkpoint
   - `trainer.default_local_dir` - output directory

2. Run:
   ```bash
   sbatch training_scripts/train_k2_k2v2_sft.sh
   ```

## Checkpoint Conversion

```bash
bash scripts/merge_model.sh
```

## Structure

```
sft/
├── sft_data/           # Data preparation
├── training_scripts/   # SLURM training scripts
├── scripts/            # Checkpoint utilities
└── README.md
```
