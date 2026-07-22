#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 2: Assign tags per asset from the generated tag_library and update annotation.json.

Usage:
  python assign_tags_per_asset.py \
      --overview /path/to/preview_all.png \
      --mapping /path/to/mapping.jsonl \
      --annotation /path/to/annotation.json \
      --model gpt-4o \
      [--resume]

Notes:
- This script is idempotent and safe to resume if you pass --resume (it will skip objects with non-empty tags).
- The model MUST ONLY select from the provided tag_library.
"""

import argparse
import base64
import json
import mimetypes
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import AzureOpenAI
from tqdm import tqdm
from annotation_prompt.annotation1_prompt2_lightv2 import annotation1_prompt2

# ----------------------------- Data models -----------------------------

class MappingItem(BaseModel):
    asset_code: str
    dst: str
    preview_img: str


# ----------------------------- Utilities ------------------------------

def to_data_url(image_path: str) -> str:
    """Return a URL (http/https) as-is; return a base64 data URL for local paths."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_mapping(mapping_path: str) -> List[MappingItem]:
    items: List[MappingItem] = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            items.append(MappingItem(
                asset_code=raw["asset_code"],
                dst=raw["dst"],
                preview_img=raw.get("preview_img", "")
            ))
    return items


def index_objects_by_preview(annotation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build an index from preview_img -> object dict, so we can update tags in-place.
    The mapping.jsonl preview_img should match objects[*].preview_img from Part 1.
    """
    idx = {}
    for obj in annotation.get("objects", []):
        if obj.get("preview_img"):
            idx[obj["preview_img"]] = obj
    return idx


# ---------------------- Azure GPT-4o (shared client) -------------------

class GPTClient:
    """Thin wrapper around Azure Chat Completions to keep that block intact."""
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_URL"],
            api_version="2024-12-01-preview",
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=60.0
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=20),
           stop=stop_after_attempt(4),
           retry=retry_if_exception_type(Exception))
    def generate_tags1234_for_asset(self, image_data_url: str,
                            tag_library_0: list[str], comparison_keys: list[str]) -> dict[str, list[str]]:
        system_prompt = (annotation1_prompt2.strip())

        tools = [{
            "type": "function",
            "function": {
                "name": "return_tags",
                "description": "Generate Human-like visual tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags_1": {"type": "array","items": {"type": "string"},"description": "Base Name + ONE Feature."},
                        "tags_2": {"type": "array","items": {"type": "string"},"description": "Adaptive Unique Identifiers. One unique tag per asset."}
                    },
                    "required": ["tags_1", "tags_2"]
                }
            }
        }]
        lib0 = tag_library_0
        user_content = [
            {"type": "text", "text": "tag_library_0 are the general names or category tags for objects:"},
            {"type": "text", "text": json.dumps(lib0, ensure_ascii=False)},
            {"type": "text", "text": "comparison_keys are the key attributes to check to distinguish between objects:"},
            {"type": "text", "text": json.dumps(comparison_keys, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]

        # ------------------------ Call Azure Chat Completions ------------------------
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "return_tags"}}
        )
        # ---------------------------------------------------------------------------
        # print("Token usage:", resp.usage)
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if tool_calls:
            args_str = tool_calls[0].function.arguments
            obj = json.loads(args_str)
        else:
            # Fallback to raw JSON parsing
            raw = choice.message.get("content") or "{}"
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"tags_1": [], "tags_2": []}


        return {
            "tags_1": obj.get("tags_1", []),
            "tags_2": obj.get("tags_2", [])
        }


# ----------------------------- Main routine ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overview", required=True, help="Path/URL of the overview image (10-asset grid).")
    parser.add_argument("--mapping", required=True, help="Path to mapping.jsonl.")
    parser.add_argument("--annotation", required=True, help="Path to annotation.json (in/out).")
    parser.add_argument("--model", default="gpt-4o", help="Azure model name (default: gpt-4o).")
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip objects that already have non-empty tags.")
    args = parser.parse_args()

    # Load inputs
    with open(args.annotation, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    t_stage = time.monotonic()

    lib0 = annotation.get("tag_library_0", [])
    comparison_keys = annotation.get("comparison_keys", [])

    if not any([lib0]):
        legacy = annotation.get("tag_library", [])
        if not legacy:
            raise RuntimeError("annotation.json has no tag libraries. Run Part 1 first.")

    overview_data_url = to_data_url(args.overview)

    client = GPTClient(model=args.model)

    tag_library1234 = client.generate_tags1234_for_asset(overview_data_url, lib0, comparison_keys)

    annotation["tag_library_1"] = list(set(tag_library1234.get("tags_1", [])))
    annotation["tag_library_2"] = list(set(tag_library1234.get("tags_2", [])))
    annotation["tag_library_3"] = list(set(tag_library1234.get("tags_3", [])))
    annotation["tag_library_4"] = list(set(tag_library1234.get("tags_4", [])))

    # Save back
    tmp = args.annotation + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.annotation)

    total_elapsed = time.monotonic() - t_stage
    print(f"[OK] Updated {args.annotation} in {total_elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
