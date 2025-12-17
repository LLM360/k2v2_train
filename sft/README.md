## Supervised Finetuning Instructions

You can simply obtain the [TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts) dataset and run the SFT script.

We further provide the weighting preprocessing script at `create_sft_mix_3effort.py`, if you want to change the dataset weights. The current weights ued in our data are specified in `mix_3efforts.json`. The final data will be output to `YOUR_OUTPUT_DIRECTORY` (feel free to change this in the script)
