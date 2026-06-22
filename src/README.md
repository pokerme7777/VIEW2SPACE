# VIEW2SPACE Codebase

This directory contains the official code released with VIEW2SPACE, including:

- public evaluation scripts for `view2space-v1`
- prompt and message-construction utilities used by the released pipeline
- training launchers and preprocessing templates for users who want to adapt the setup

## Repository Structure

```text
src/
  eval/
    run_all_subsets.sh
    test_qwen3vl.py
    evaluation_qwen3vl.py
  train/
    training_data_preprocessing.sh
    train_qwen3_vl.sh
    train_qwen3vl_trl_sft.py
  build_qwen3vl_train_jsonl.py
  testing_prompt.py
  environment.yml
  requirements.txt
  deepspeed_zero3.yaml
```

File summary:

- `eval/run_all_subsets.sh`: run inference and evaluation for `count`, `detect`, and `mcq`
- `eval/test_qwen3vl.py`: model inference entrypoint
- `eval/evaluation_qwen3vl.py`: prediction scoring script
- `build_qwen3vl_train_jsonl.py`: released message builder used by the eval and training pipeline
- `testing_prompt.py`: released prompt templates
- `train/training_data_preprocessing.sh`: template for converting your own data into training JSONL
- `train/train_qwen3_vl.sh`: template for launching training with your own paths

## Installation

The public evaluation code was validated with:

- Python `3.10`
- `torch==2.7.1`
- `transformers==4.57.0`
- `accelerate==1.12.0`
- `Pillow==12.0.0`
- `tqdm==4.67.1`

Clone the official repository first:

```bash
git clone https://github.com/pokerme7777/VIEW2SPACE.git
cd VIEW2SPACE
```

Create the recommended conda environment:

```bash
conda env create -f src/environment.yml
conda activate view2space
```

Or install manually:

```bash
conda create -n view2space python=3.10 -y
conda activate view2space
pip install -r src/requirements.txt
```

If you plan to use `--flash_attn`, install a compatible `flash-attn` build for
your CUDA / PyTorch stack separately.

## Public Dataset Format

The released evaluation scripts expect the dataset root to follow:

```text
view2space-v1/
  count/overall.jsonl
  detect/overall.jsonl
  mcq/overall.jsonl
  images/...
```

Each JSONL record is expected to contain:

- `q_idx`
- `q_type`
- `question`
- `options`
- `question_prompt`
- `answer`
- `image_paths`
- `supporting.draw_boxes`

`image_paths` must be relative to the dataset root, for example
`images/img_000123.png`.

## Evaluation Quick Start

Run all released subsets:

```bash
bash src/eval/run_all_subsets.sh \
  /path/to/model_dir \
  /path/to/view2space-v1 \
  /path/to/results
```

`run_all_subsets.sh` uses the released default inference settings:

- `--batch_size 16`
- `--num_workers 6`
- `--prefetch_factor 4`
- `--max_new_tokens 4096`
- `--dtype bf16`

Additional inference arguments can be appended after `RESULT_ROOT`.

Example with the released Hugging Face checkpoint:

```bash
bash src/eval/run_all_subsets.sh \
  Pokerme/view2space_4b \
  ./view2space-v1 \
  ./prediction
```

Run a single subset manually:

```bash
python src/eval/test_qwen3vl.py \
  --model_dir /path/to/model_dir \
  --in_overall_jsonl /path/to/view2space-v1/mcq/overall.jsonl \
  --scenes_root /path/to/view2space-v1 \
  --result_dir /path/to/results/mcq \
  --batch_size 16 \
  --num_workers 6 \
  --prefetch_factor 4 \
  --max_new_tokens 4096 \
  --dtype bf16
```

Then score the predictions:

```bash
python src/eval/evaluation_qwen3vl.py \
  --predictions_jsonl /path/to/results/mcq/predictions.jsonl \
  --task_mode mcq
```

For detection, if predictions are emitted in 0-1000 normalized coordinates,
add:

```bash
--scale_1000
```

## Training Utilities

Training-related files are provided as released templates rather than
turnkey scripts for every environment.

Before running them:

- edit the path variables at the top of `src/train/training_data_preprocessing.sh`
- edit the path variables at the top of `src/train/train_qwen3_vl.sh`
- keep the training arguments unchanged unless you intentionally want to tune them

## Notes

- Released public `q_type` values are `count`, `detect`, and `mcq`.
- `eval/test_qwen3vl.py` expects `--scenes_root` to point to the dataset root,
  not the `images/` subfolder.
- `run_all_subsets.sh` passes `--scenes_root "$DATASET_ROOT"` automatically.
