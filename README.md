## Requirements:

❗ **Hardware Requirement**:
- At least one **NVIDIA A40 48G GPU**

📦 **Python Dependencies**:
```bash
pip install torch transformers numpy tqdm
```
## Prepare

1. Download [meta-llama/Meta-Llama-3-8B · Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

2. Clone AlphaEdit:[AlphaEdit](https://github.com/jianghoucheng/AlphaEdit.git)

## Quick Start

1. Virtual Knowledge Prefix Construction

   ```python
   python prefix.py
   ```
   * input_file: Input raw prefixes (one per line)
   * output_file: Output filtered prefixes

2. Dynamic Prefix Selection
   ```python
   python construct.py
   ```
   * prefix_file: The prefixes filted by step 1
   * output_file1: Constructed train data 
   * output_file2: Constructed val data

3. Dual-channel knowledge edit
   1. Copy output_file1 of Step 2 into AlphaEdit/dsets
   ```bash
   cp ./datasets/train_data.json ./AlphaEdit/dsets/
   ```

   2. Add tarin_data in DS_DICT in AlphaEdit/experiments/evaluate.py and add TrainDataset
   ```python
   DS_DICT = {
      # ... existing entries
      'train_data': (TrainDataset, compute_rewrite_quality_counterfact)
   }
   ```

   2. Run Editing Process(Take Llama3-8B for example):
   ```
   python3 -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=train_data \
    --dataset_size_limit=100 \
    --num_edits=100 \
    --downstream_eval_steps=5
   ```
