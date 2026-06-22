#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import torch
from datasets import Dataset, disable_caching 
from PIL import Image, ImageDraw
from dataclasses import dataclass

from peft import LoraConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info


disable_caching()


# -----------------------------
# 1) Load jsonl
# -----------------------------
def load_messages_jsonl(train_file: str) -> Dataset:
    rows = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError("Each line must contain 'messages'.")
            # rows.append({"messages": obj["messages"]})
            rows.append(obj)
    return Dataset.from_list(rows)


# -----------------------------
# 2) Return PIL after drawing box
# -----------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

@lru_cache(maxsize=64)
def _load_rgb_cached(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")

def resize_image_if_needed(img: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        return img

    width, height = img.size
    current_pixels = width * height
    if current_pixels <= max_pixels:
        return img

    scale = (max_pixels / float(current_pixels)) ** 0.5
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

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


# -----------------------------
# 3) Convert: image_path+bboxes -> PIL image , in message
#    Input sample: {"messages":[...]}
#    Output sample: {"messages":[...]} The image item becomes {"type":"image","image": PIL}
# -----------------------------
def build_materialize_images(
    prefer_cache: bool = False,
    clamp_to_image: bool = True,
    max_pixels: int = 1024 * 1024,
):
    def _transform(batch: Dict[str, Any]) -> Dict[str, Any]:
        all_mesgs_in_batch = batch["messages"]
        new_all_mesgs_in_batch = []
        images_by_sample = []

        for msgs in all_mesgs_in_batch:
            new_msgs = []
            sample_images = []

            for m in msgs:
                role = m.get("role")
                content = m.get("content")

                # if content is str or not user, keep as is
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

                    # situation A：image_path + bboxes
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

                        boxed = resize_image_if_needed(boxed, max_pixels=max_pixels)
                        
                        
                        # store image
                        sample_images.append(boxed)

                        # Replace with processor consumable format
                        new_item = {"type": "image", "image": boxed}
                        # new_item = {"type": "image", "image": boxed, "image_url": "pil"}

                        # If you need to keep other fields, you can copy them here
                        new_content.append(new_item)
                        continue

                    # situation B：if some samples are already {"type":"image","image": ...}
                    # It may be path/url/data-url/PIL, which is passed through as is
                    # if "image" in c:
                    #     new_content.append(c)
                    #     continue

                    raise ValueError(f"Unsupported image item: {c}")

                new_msgs.append({"role": role, "content": new_content})

            new_all_mesgs_in_batch.append(new_msgs)
            images_by_sample.append(sample_images)

        return {"messages":new_all_mesgs_in_batch, "images_by_sample": images_by_sample}
    return _transform

@dataclass
class Qwen3VLCollatorPerSampleImages:
    processor: Any
    max_length: int = 4096

    def __post_init__(self):
        tok = self.processor.tokenizer
        self._im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        # encode "assistant\n" as a whole to match actual tokenization
        # (BPE may merge differently than encoding pieces separately)
        assistant_nl_ids = tok.encode("assistant\n", add_special_tokens=False)
        # header = <|im_start|> + assistant\n
        self._assistant_header = [self._im_start_id] + assistant_nl_ids
        self._header_len = len(self._assistant_header)
        print(f"[CollatorInit] assistant header token ids: {self._assistant_header} "
              f"(len={self._header_len}), im_end_id={self._im_end_id}")

    def _make_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Only supervise assistant response tokens; mask everything else with -100."""
        labels = torch.full_like(input_ids, -100)
        pad_id = self.processor.tokenizer.pad_token_id

        for i in range(input_ids.shape[0]):
            seq = input_ids[i].tolist()
            found = False
            j = 0
            while j < len(seq):
                # look for <|im_start|>assistant\n
                if (seq[j] == self._im_start_id
                        and seq[j:j + self._header_len] == self._assistant_header):
                    found = True
                    # skip the header itself (keep masked)
                    content_start = j + self._header_len
                    # copy labels from content_start until <|im_end|> (inclusive)
                    k = content_start
                    while k < len(seq) and seq[k] != self._im_end_id:
                        if seq[k] != pad_id:
                            labels[i, k] = seq[k]
                        k += 1
                    if k < len(seq):          # include <|im_end|> so model learns to stop
                        labels[i, k] = seq[k]
                    j = k + 1
                    continue
                j += 1
            if not found:
                import warnings
                warnings.warn(
                    f"[_make_labels] sample {i}: no assistant header found! "
                    f"First 20 token ids: {seq[:20]}. All labels are -100, this sample contributes no loss."
                )
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images: List[List[Any]] = []   # each sample has different length of list

        for f in features:
            msgs = f["messages"]
            prompt = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(prompt)

            imgs = f.get("images_by_sample", []) or []
            images.append(imgs)

            # ----check ：len of placeholder == number of sample image ----
            image_inputs, video_inputs = process_vision_info(msgs)

            if len(image_inputs) != len(imgs):
                raise ValueError(
                    f"Placeholder/image mismatch: placeholders={len(image_inputs)}, images={len(imgs)}\n"
                    f"prompt(head)={prompt[:300]}"
                )

        batch = self.processor(
            text=texts,
            images=images,          # key：per-sample list，do not use flatten
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # batch["labels"] = self._make_labels(batch["input_ids"])
        batch["labels"] = batch["input_ids"].clone() # old stable version
        return batch


# -----------------------------
# 4) LoRA setting
# -----------------------------
def build_lora_config(rank: int, alpha: int, dropout: float, full: bool,
                      lora_vision: bool = False) -> LoraConfig:
    # --- LLM target modules ---
    if full:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    # --- Vision encoder target modules ---
    if lora_vision:
        target_modules += [
            "qkv",          # Qwen3VLVisionAttention.qkv  (fused Q/K/V)
            "linear_fc1",   # VisionMLP.linear_fc1 + Merger.linear_fc1
            "linear_fc2",   # VisionMLP.linear_fc2 + Merger.linear_fc2
        ]

    # https://medium.com/@ishaafsalman/fine-tuning-qwen-qwen3-vl-30b-a3b-moe-architecture-with-lora-2365359e870f
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_file", required=True, type=str, help="prepared jsonl (each line has {'messages':[...]})")
    ap.add_argument("--prefer_cache", action="store_true", help="LRU cache original images in memory")
    ap.add_argument("--no_clamp_to_image", action="store_true", help="do not clamp bbox to image boundary")
    ap.add_argument(
        "--max_image_pixels",
        type=int,
        default=800 * 800,
        help="resize images down before processor if width*height exceeds this limit; set <=0 to disable",
    )
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    ap.add_argument("--dataloader_pin_memory", action="store_true")
    ap.add_argument("--dataloader_persistent_workers", action="store_true")

    # model
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct", type=str)
    ap.add_argument("--attn_impl", default="flash_attention_2", type=str, choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--max_seq_length", default=4096, type=int)

    # train
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--num_train_epochs", default=2, type=int)
    ap.add_argument("--per_device_train_batch_size", default=1, type=int)
    ap.add_argument("--gradient_accumulation_steps", default=8, type=int)
    ap.add_argument("--learning_rate", default=1e-4, type=float)
    ap.add_argument("--warmup_ratio", default=0.03, type=float)
    ap.add_argument("--logging_steps", default=1, type=int)
    ap.add_argument("--save_strategy", default="epoch", type=str)
    ap.add_argument("--report_to", default="none", type=str)  # "wandb" / "none"
    ap.add_argument("--output_model_folder_subname", type=str, required=True, help="name of each different folder")

    # lora
    ap.add_argument("--enable_lora", action="store_true")
    ap.add_argument("--lora_rank", default=8, type=int)
    ap.add_argument("--lora_alpha", default=16, type=int)
    ap.add_argument("--lora_dropout", default=0.05, type=float)
    ap.add_argument("--lora_full_target", action="store_true")
    ap.add_argument("--lora_vision", action="store_true",
                    help="also apply LoRA to vision encoder & merger")

    args = ap.parse_args()

    full_output_dir = os.path.join(args.output_dir, args.output_model_folder_subname)
    os.makedirs(full_output_dir, exist_ok=True)

    # ---- 1) dataset: load jsonl -> materialize PIL images with boxes
    train_dataset = load_messages_jsonl(args.train_file)

    train_dataset = train_dataset.with_format("python")
    train_dataset.set_transform(
        build_materialize_images(
            prefer_cache=args.prefer_cache,
            clamp_to_image=not args.no_clamp_to_image,
            max_pixels=args.max_image_pixels,
        )
    )


    # ---- 2) model & processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(args.model_name)


    # ---- 3) LoRA config  (let SFTTrainer handle wrapping for DeepSpeed ZeRO-3 compatibility)
    peft_config = None
    if args.enable_lora:
        peft_config = build_lora_config(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            full=args.lora_full_target,
            lora_vision=args.lora_vision,
        )

    # ---- 4) SFT config (TRL)
    train_args = SFTConfig(
        output_dir=full_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        bf16=True,
        max_length=args.max_seq_length,  # trl    0.26.2 they updated this one in Sept 2025
        max_grad_norm=1.0,

        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},

        report_to=args.report_to,

        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
    )


    my_collator = Qwen3VLCollatorPerSampleImages(processor, max_length=args.max_seq_length)


    # ---- 5) TRL trainer
    # key：processing_class=processor. Using TRL processor to auto process for VLM（including PIL image）
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        processing_class=processor,    # previously we are using this one for many good version
        data_collator=my_collator,
        peft_config=peft_config,
    )

    # SFTTrainer calls get_peft_model() internally but does NOT call
    # enable_input_require_grads(). Without it, gradient checkpointing
    # with use_reentrant=True breaks the gradient chain (embeddings are
    # frozen by LoRA → embedding output has requires_grad=False →
    # checkpoint segments receive no-grad inputs → "Gradients will be None").
    if peft_config is not None:
        trainer.model.enable_input_require_grads()


    def get_latest_checkpoint(output_dir: str):
        ckpts = list(Path(output_dir).glob("checkpoint-*"))
        if not ckpts:
            return None

        def extract_step(p: Path):
            m = re.search(r"checkpoint-(\d+)", p.name)
            return int(m.group(1)) if m else -1

        ckpts = [(extract_step(p), p) for p in ckpts]
        ckpts = [x for x in ckpts if x[0] >= 0]
        if not ckpts:
            return None

        ckpts.sort(key=lambda x: x[0])
        return str(ckpts[-1][1])


    # resume (if exist checkpoint-*)
    latest_ckpt = get_latest_checkpoint(full_output_dir)
    if latest_ckpt:
        print("Resume checkpoint:", latest_ckpt)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(full_output_dir)


if __name__ == "__main__":
    main()
