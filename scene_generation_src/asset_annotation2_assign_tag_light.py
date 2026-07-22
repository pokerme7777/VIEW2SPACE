#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 2: Assign tags per asset from the generated tag_library and update annotation.json.

Usage:
  python assign_tags_per_asset.py \
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
from typing import List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import AzureOpenAI
from tqdm import tqdm


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
    def pick_tags_for_asset(self, image_data_url: str,
                            tag_library_0: list[str],
                            tag_library_1: list[str],
                            tag_library_2: list[str],
                            tag_library_3: list[str],
                            tag_library_4: list[str]) -> dict[str, list[str]]:
        system_prompt = (
        "You will see one asset image and FIVE tag libraries grouped by combination level 0, 1, 2."
        "For EACH level, select ALL and ONLY tags that apply to this image. "
        "Never invent new tags; only choose from the provided list for that level. "
        "Return via function call with keys: tags_0, tags_1, tags_2"
        )

        tools = [{
            "type": "function",
            "function": {
                "name": "return_selected_tags",
                "description": "Return the selected subset of tags for this asset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags_0": {"type": "array","items": {"type": "string"},"description": "Subset chosen from tag_library_0 that apply."},
                        "tags_1": {"type": "array","items": {"type": "string"},"description": "Subset chosen from tag_library_1 that apply."},
                        "tags_2": {"type": "array","items": {"type": "string"},"description": "Subset chosen from tag_library_2 that apply."}
                    },
                    "required": ["tags_0","tags_1", "tags_2"]
                }
            }
        }]
        lib0, lib1, lib2 = tag_library_0, tag_library_1, tag_library_2
        user_content = [
            {"type": "text", "text": "Only select from these tag libraries by level:"},
            {"type": "text", "text": "tag_library_0:"},
            {"type": "text", "text": json.dumps(lib0, ensure_ascii=False)},
            {"type": "text", "text": "tag_library_1:"},
            {"type": "text", "text": json.dumps(lib1, ensure_ascii=False)},
            {"type": "text", "text": "tag_library_2:"},
            {"type": "text", "text": json.dumps(lib2, ensure_ascii=False)},
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
            tool_choice={"type": "function", "function": {"name": "return_selected_tags"}}
        )
        # ---------------------------------------------------------------------------

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
                obj = {"tags_0": [], "tags_1": [], "tags_2": [], "tags_3": [], "tags_4": []}

        def filter_level(arr, lib):
            libset = set(lib)
            out, seen = [], set()
            for t in arr or []:
                s = str(t).strip()
                if s and s in libset and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        return {
            "tags_0": filter_level(obj.get("tags_0", []), tag_library_0),
            "tags_1": filter_level(obj.get("tags_1", []), tag_library_1),
            "tags_2": filter_level(obj.get("tags_2", []), tag_library_2),
            "tags_3": filter_level(obj.get("tags_3", []), tag_library_3),
            "tags_4": filter_level(obj.get("tags_4", []), tag_library_4),
        }


# ----------------------------- Main routine ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, help="Path to mapping.jsonl.")
    parser.add_argument("--annotation", required=True, help="Path to annotation.json (in/out).")
    parser.add_argument("--model", default="gpt-4o", help="Azure model name (default: gpt-4o).")
    parser.add_argument("--resume", action="store_true",
                        help="If set, skip objects that already have non-empty tags.")
    args = parser.parse_args()

    # Load inputs
    with open(args.annotation, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    lib0 = annotation.get("tag_library_0", [])
    lib1 = annotation.get("tag_library_1", [])
    lib2 = annotation.get("tag_library_2", [])
    lib3 = annotation.get("tag_library_3", [])
    lib4 = annotation.get("tag_library_4", [])

    if not any([lib1, lib2, lib3, lib4]):
        legacy = annotation.get("tag_library", [])
        if not legacy:
            raise RuntimeError("annotation.json has no tag libraries. Run Part 1 first.")
        lib1, lib2, lib3, lib4 = legacy, [], [], []

    mapping_items = load_mapping(args.mapping)
    obj_index = index_objects_by_preview(annotation)

    client = GPTClient(model=args.model)

    updated = 0
    skipped = 0

    for mi in tqdm(mapping_items):
        obj = obj_index.get(mi.preview_img)
        if not obj:
            # If the object is missing (e.g., annotation scaffold changed), create it
            obj = {
                "asset_code": mi.asset_code,
                "preview_img": mi.preview_img,
                "tags_0": [],
                "tags_1": [],
                "tags_2": [],
                "tags_3": [],
                "tags_4": []
            }
            annotation.setdefault("objects", []).append(obj)

        if args.resume and any([
            obj.get("tags_0"),
            obj.get("tags_1"), obj.get("tags_2"),
            obj.get("tags_3"), obj.get("tags_4")
        ]):
            skipped += 1
            continue

        try:
            data_url = to_data_url(mi.preview_img)
        except Exception as e:
            print(f"[WARN] Cannot read preview for asset_code={mi.asset_code}: {e}")
            obj["tags"] = obj.get("tags", [])
            continue

        try:
            selected = client.pick_tags_for_asset(data_url, lib0,lib1, lib2, lib3, lib4)
        except Exception as e:
            print(f"[WARN] GPT selection failed for {obj.get('asset_code')}: {e}")
            selected = {}

        obj["tags_0"] = selected.get("tags_0", [])
        obj["tags_1"] = selected.get("tags_1", [])
        obj["tags_2"] = selected.get("tags_2", [])
        obj["tags_3"] = selected.get("tags_3", [])
        obj["tags_4"] = selected.get("tags_4", [])
        obj["flat_all"] = obj["tags_0"] + obj["tags_1"] + obj["tags_2"] + obj["tags_3"] + obj["tags_4"]
        updated += 1

    # Save back
    tmp = args.annotation + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.annotation)

    print(f"[OK] Updated {args.annotation}. Updated={updated}, Skipped={skipped}, Objects={len(annotation.get('objects', []))}.")


if __name__ == "__main__":
    main()
