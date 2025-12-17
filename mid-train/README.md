The training script uses context parallelism to support long-context training. Our parameters are only tested under H200 GPU setting, which may run out of memory for different accelerators.

The data can be obtained through [TxT360-Midas](https://huggingface.co/datasets/LLM360/TxT360-Midas), the datasets are organized as subsets, corresponding to the stages.