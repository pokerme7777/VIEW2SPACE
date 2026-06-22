#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference for Qwen3-VL with real box drawing during inference.

What it does:
- Read in_overall_jsonl (overall-format records)
- For each record: build Qwen-style messages (system+user) using build_messages_one()
- Materialize each image_path + bboxes item into a PIL image with boxes drawn
- Run batch generation with Qwen3-VL
- Save predictions to result_dir/predictions.jsonl
- Optionally save boxed images to result_dir/boxed_images/ for debugging

Notes:
- This script depends on src/build_qwen3vl_train_jsonl.py.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_qwen3vl_train_jsonl import build_messages_one

# -----------------------------
# Utilities
# -----------------------------
def batched(iterable, batch_size: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)




def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@lru_cache(maxsize=128)
def _load_rgb_cached(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def draw_boxes_on_pil(img: Image.Image, bboxes: List[List[float]], clamp_to_image: bool = True) -> Image.Image:
    if not bboxes:
        return img

    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)

    for box in bboxes:
        if not box or len(box) != 4:
            continue
        x, y, w, h = box

        if clamp_to_image:
            x = _clamp(x, 0, w_img - 1)
            y = _clamp(y, 0, h_img - 1)
            w = _clamp(w, 0, w_img - x)
            h = _clamp(h, 0, h_img - y)

        thick = max(1, int(round(min(w, h) * 0.0035))) if (w > 0 and h > 0) else 1

        x1, y1, x2, y2 = x, y, x + w, y + h
        for t in range(thick):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(0, 255, 0))

    return img

def merge_system_into_user(conv):
    if len(conv) >= 2 and conv[0]["role"] == "system" and conv[1]["role"] == "user":
        sys_text = "\n".join([x["text"] for x in conv[0]["content"] if x.get("type") == "text"])
        new_user_content = []
        # merge system prompt into user. Because molmo template does not support system prompt!
        if sys_text.strip():
            new_user_content.append({"type": "text", "text": sys_text.strip()})
        new_user_content.extend(conv[1]["content"])
        return [{"role": "user", "content": new_user_content}]
    return conv

def materialize_one_sample(
    msgs: List[Dict[str, Any]],
    prefer_cache: bool = False,
    clamp_to_image: bool = True,
    save_boxed_dir: Optional[str] = None,
    sample_id: str = "sample",
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:

    new_msgs: List[Dict[str, Any]] = []
    sample_images: List[Image.Image] = []
    boxed_counter = 0

    if save_boxed_dir:
        os.makedirs(save_boxed_dir, exist_ok=True)

    for m in msgs:
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError("message.content must be str or list")

        if role != "user":
            new_msgs.append({"role": role, "content": content})
            continue

        new_content = []
        for c in content:
            if not isinstance(c, dict):
                new_content.append(c)
                continue
            if c.get("type") != "image":
                new_content.append(c)
                continue

            # situation A: image_path + bboxes 
            if "image_path" in c:
                image_path = c["image_path"]
                bboxes = c.get("bboxes", []) or []

                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                base = _load_rgb_cached(image_path) if prefer_cache else Image.open(image_path).convert("RGB")

                if not bboxes:
                    boxed = base
                else:
                    base1 = base.copy() if prefer_cache else Image.open(image_path).convert("RGB")
                    boxed = draw_boxes_on_pil(base1, bboxes=bboxes, clamp_to_image=clamp_to_image)

                sample_images.append(boxed)
                if save_boxed_dir:
                    out_img = os.path.join(save_boxed_dir, f"{sample_id}_{boxed_counter}.jpg")
                    boxed.save(out_img, quality=95)
                    boxed_counter += 1

                new_content.append({"type": "image", "image": boxed})
                continue

            # situation B: already {"type":"image","image": ...} 
            if "image" in c:
                new_content.append(c)
                continue

            raise ValueError(f"Unsupported image item: {c}")

        new_msgs.append({"role": role, "content": new_content})

    return new_msgs, sample_images


class OverallJsonlDataset(Dataset):
    def __init__(
        self,
        in_overall_jsonl,
        scenes_root,
        done_qidx_set,
        CoT_option=True,
        prompt_style="default",
        is_molmo=False,
        pure_text=False,
    ):
        self.done_qidx_set = done_qidx_set
        self.in_overall_jsonl = in_overall_jsonl
        self.scenes_root = scenes_root
        self.CoT_option = CoT_option
        self.prompt_style = prompt_style
        self.is_molmo = is_molmo
        self.pure_text = pure_text

        self._lines = []
        with open(in_overall_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._lines.append(line)

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx):
        rec = json.loads(self._lines[idx])

        q_idx = rec.get("q_idx")
        if q_idx is not None and q_idx in self.done_qidx_set:
            return None # skip already done samples
        

        built = build_messages_one(
            rec,
            scenes_root=self.scenes_root,
            CoT_option=self.CoT_option,
            prompt_style=self.prompt_style,
            pure_text=self.pure_text,
        )
        msgs = built["messages"]
        msgs = msgs[:2] if len(msgs) >= 2 else msgs  # system + user

        if self.is_molmo:
            msgs = merge_system_into_user(msgs)

        new_msgs, sample_images = materialize_one_sample(
            msgs,
            prefer_cache=False,              # do not share cache across workers
            clamp_to_image=True,
            save_boxed_dir=None,
            sample_id=q_idx,
        )

        meta = {
            "q_idx": q_idx,
            "q_type": rec.get("q_type"),
            "question": rec.get("question"),
            "options": rec.get("options"),
            "answer": rec.get("answer"),
            "image_paths": rec.get("image_paths"),
            "supporting": rec.get("supporting"),
        }
        return new_msgs, sample_images, meta

def make_collate_fn(processor):
    def collate(batch):
        # filter skipped samples
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            print("[DEBUG] collate got empty batch (all samples invalid/None)")
            return None

        msgs_list, images_list, metas = zip(*batch)

        texts = [
            processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in msgs_list
        ]

        return texts, list(images_list), list(metas)
    return collate


def load_model_classes():
    try:
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise ImportError(
            "Failed to import Qwen vision-language model classes from `transformers`. "
            "Install a recent `transformers` release that includes Qwen3-VL and Qwen2.5-VL support."
        ) from exc

    return (
        AutoModelForImageTextToText,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

# -----------------------------
# Main inference
# -----------------------------
@torch.inference_mode()
def run_inference(args):
    print("[INFO]] Current version is for CoT:", {args.CoT_option})
    os.makedirs(args.result_dir, exist_ok=True)
    boxed_dir = os.path.join(args.result_dir, "boxed_images") if args.save_boxed_images else None

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("[INFO] Start loading model")
    (
        AutoModelForImageTextToText,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    ) = load_model_classes()
    model_dir_lower = args.model_dir.lower()
    model_type = None
    config_path = Path(args.model_dir) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_type = json.load(f).get("model_type")
        except Exception as e:
            print(f"[WARN] Failed to read model_type from {config_path}: {e}")

    # Prefer model_type from config for local checkpoints; fallback to path heuristics.
    is_molmo = (model_type == "molmo2") or ("molmo2" in model_dir_lower) or ("molmol2" in model_dir_lower)
    is_qwen3 = (model_type == "qwen3_vl") or ("qwen3" in model_dir_lower)
    is_qwen25 = (model_type == "qwen2_5_vl") or any(
        k in args.model_dir for k in ("MLL-Lab", "Qwen2.5", "RoboBrain2", "Diankun/Spatial", "qwen2.5")
    )

    if is_molmo:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    elif is_qwen3:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if args.flash_attn else None,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_dir)
    elif is_qwen25:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype=dtype, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_dir)
    else:
        raise ValueError(
            f"Cannot infer model family for model_dir={args.model_dir}, model_type={model_type}. "
            "Expected molmo2/qwen3_vl/qwen2_5_vl."
        )

    # with open(os.path.join(args.model_dir, "adapter_config.json"), "r") as f:
    #     config = json.load(f)

    # base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     config["base_model_name_or_path"],
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # processor = AutoProcessor.from_pretrained(args.model_dir)
    # model = PeftModel.from_pretrained(base_model, args.model_dir)

    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.eval()
    print("[Info] model merged and unloaded.")

    # ---------------------------------
    # Load existing predictions (resume)
    # ---------------------------------
    done_qidx = set()
    out_path = os.path.join(args.result_dir, "predictions.jsonl")

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "q_idx" in obj:
                        done_qidx.add(obj["q_idx"])
                except Exception:
                    continue

    print(f"[Resume] Found {len(done_qidx)} finished samples")

    dataset = OverallJsonlDataset(
        args.in_overall_jsonl,
        args.scenes_root,
        done_qidx,
        CoT_option=args.CoT_option,
        prompt_style=args.prompt_style,
        is_molmo=is_molmo,
        pure_text=args.pure_text,
    )
    # Add these CLI args if you want; otherwise defaults are fine.
    num_workers = getattr(args, "num_workers", 4)
    prefetch_factor = getattr(args, "prefetch_factor", 2)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=make_collate_fn(processor),
    )

    print("[Info] dataloader created, starting loop...")
    
    with open(out_path, "a", encoding="utf-8") as fout:
        print("[Info] got first batch")
        for batch in tqdm(dataloader):
            if batch is None:
                continue

            texts, images_by_sample, metas = batch
            processor_kwargs = {
                "text": texts,
                "padding": True,
                "return_tensors": "pt",
            }
            if any(sample_images for sample_images in images_by_sample):
                processor_kwargs["images"] = images_by_sample

            inputs = processor(**processor_kwargs)


            # move tensors to model device
            inputs = {
                k: v.to(model.device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            # 4) generate
            gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            # 5) trim prompt + decode
            in_ids = inputs["input_ids"]
            trimmed = [out[len(inp):] for inp, out in zip(in_ids, gen_ids)]
            preds = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # 6) write predictions
            for meta, pred in zip(metas, preds):
                fout.write(json.dumps({
                    **meta,
                    "prediction": pred,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())


    print(f"[OK] wrote -> {out_path}")
    if boxed_dir:
        print(f"[OK] boxed images -> {boxed_dir}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Folder with the trained model (or merged model).")
    ap.add_argument("--in_overall_jsonl", type=str, required=True, help="Input overall.jsonl for evaluation.")
    ap.add_argument("--scenes_root", type=str, required=True, help="Root dir to prefix relative image_paths.")
    ap.add_argument("--result_dir", type=str, required=True, help="Output folder for predictions.jsonl etc.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--no_CoT_option", dest="CoT_option", action="store_false", help="Disable Chain-of-Thought option prompts.")
    ap.set_defaults(CoT_option=True)
    ap.add_argument(
        "--prompt_style",
        choices=["default", "mindcube_cogmap"],
        default="default",
        help="Prompt template style to use for inference.",
    )


    ap.add_argument("--flash_attn", action="store_true", help="Use flash_attention_2 when supported.")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Compute dtype.")
    ap.add_argument("--prefer_cache", action="store_true", help="Cache image decoding with LRU and copy per use.")
    ap.add_argument("--no_clamp", action="store_true", help="Do not clamp bboxes to image bounds.")
    ap.add_argument("--save_boxed_images", action="store_true", help="Save boxed images to result_dir/boxed_images.")
    ap.add_argument("--pure_text", action="store_true", help="Run text-only evaluation without passing images.")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_inference(args)
