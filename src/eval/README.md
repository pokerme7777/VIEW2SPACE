# view2space-v1 Eval Code

This folder contains standalone evaluation scripts for the public `view2space-v1` release.

Public dataset:

- Hugging Face: `https://huggingface.co/datasets/Pokerme/view2space-v1`

Public checkpoint:

- Hugging Face: `https://huggingface.co/Pokerme/view2space_4b`

## Files

- `test_qwen3vl.py`: run model inference and write `predictions.jsonl`
- `evaluation_qwen3vl.py`: score `predictions.jsonl`
- `build_qwen3vl_train_jsonl.py`: build message-format JSONL from the public dataset format
- `testing_prompt.py`: prompt templates used by the evaluation scripts
- `run_all_subsets.sh`: convenience script to run `count`, `detect`, and `mcq`
- `requirements.txt`: validated Python package pins for the public eval code
- `environment.yml`: conda environment definition (`view2space`)

## Dataset Layout

These scripts expect the public dataset root to look like:

```text
view2space-v1/
  count/overall.jsonl
  detect/overall.jsonl
  mcq/overall.jsonl
  images/...
```

The JSONL files should contain:

- `q_idx`
- `q_type`
- `question`
- `options`
- `question_prompt`
- `answer`
- `image_paths`
- `supporting.draw_boxes`

`image_paths` must be relative to the dataset root, for example `images/img_000123.png`.

## Installation

Create the validated conda environment:

```bash
conda env create -f environment.yml
conda activate view2space
```

This environment mirrors the package versions validated in the internal `mmview`
runtime for the public eval scripts:

- Python `3.10`
- `torch==2.7.1` (`mmview` used the CUDA 12.6 build)
- `transformers==4.57.0`
- `accelerate==1.12.0`
- `Pillow==12.0.0`
- `tqdm==4.67.1`

If you prefer creating the environment manually, use:

```bash
conda create -n view2space python=3.10 -y
conda activate view2space
pip install -r requirements.txt
```

If you plan to run with `--flash_attn`, install a compatible `flash-attn` build
separately for your CUDA / PyTorch setup.

## Run All Subsets

```bash
bash run_all_subsets.sh \
  Pokerme/view2space_4b \
  /path/to/view2space-v1-release \
  /path/to/results
```

By default, `run_all_subsets.sh` uses the same core inference settings as the
legacy internal evaluation script:

- `--batch_size 16`
- `--num_workers 6`
- `--prefetch_factor 4`
- `--max_new_tokens 4096`
- `--dtype bf16`

You can still override any of them by appending extra CLI flags after
`RESULT_ROOT`.

This will:

1. run inference on `count`
2. run inference on `detect`
3. run inference on `mcq`
4. write `predictions.jsonl` and `evaluation_result.json` under the result directory

## Run One Subset

Inference:

```bash
python test_qwen3vl.py \
  --model_dir Pokerme/view2space_4b \
  --in_overall_jsonl /path/to/view2space-v1-release/mcq/overall.jsonl \
  --scenes_root /path/to/view2space-v1-release \
  --result_dir /path/to/results/mcq \
  --batch_size 16 \
  --num_workers 6 \
  --prefetch_factor 4 \
  --max_new_tokens 4096 \
  --dtype bf16
```

Evaluation:

```bash
python evaluation_qwen3vl.py \
  --predictions_jsonl /path/to/results/mcq/predictions.jsonl \
  --task_mode mcq
```

For detection, if predictions are emitted in 0-1000 normalized coordinates, add:

```bash
--scale_1000
```

## Notes

- Public `q_type` values are `count`, `detect`, and `mcq`.
- `test_qwen3vl.py` expects `--scenes_root` to point to the dataset root, not the `images/` subfolder.
- `run_all_subsets.sh` passes `--scenes_root "$DATASET_ROOT"` automatically.
- For the released public dataset, the prompt routing is performed by public `q_type` families (`count`, `detect`, `mcq`) instead of the older fine-grained internal subtype names.
