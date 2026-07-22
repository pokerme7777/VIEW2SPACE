#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part 1: Generate a tag library from the overview image and initialize annotation.json.

Usage:
  python asset_annotation1_tag_library.py \
      --overview /path/to/preview_all.png \
      --mapping /path/to/mapping.jsonl \
      --out /path/to/annotation.json \
      --folder_name snowman \
      --model gpt-4o \
      --verbose \
      --http-debug

Environment:
- AZURE_OPENAI_URL
- OPENAI_API_KEY
"""

import argparse
import base64
import json
import mimetypes
import os
import time
import logging
from typing import List, Dict, Any

from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from tenacity import before_sleep_log
from openai import AzureOpenAI
from tqdm import tqdm
from annotation_prompt.annotation1_prompt_light import annotation1_prompt  # for reference only


# ----------------------------- Logging helpers -----------------------------

def setup_logging(verbose: bool = False, http_debug: bool = False) -> None:
    """Configure console logging and optional HTTP/OpenAI debug logs."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if http_debug:
        # OpenAI SDK / httpx debug (can be very verbose)
        os.environ["OPENAI_LOG"] = "debug"
        # httpx uses standard logging; enable at DEBUG for network traces
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)

logger = logging.getLogger("taglib")  # module-level logger


# ----------------------------- Data models -----------------------------

class MappingItem(BaseModel):
    """Single mapping item from mapping.jsonl."""
    asset_code: str
    dst: str
    preview_img: str


# ----------------------------- Utilities ------------------------------

def to_data_url(image_path: str) -> str:
    """Return a URL (http/https) as-is; return a base64 data URL for local paths."""
    # Logging for troubleshooting
    logger.info(f"Preparing data URL for image: {image_path}")
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
    """Read mapping.jsonl and return a list of MappingItem instances."""
    logger.info(f"Loading mapping from: {mapping_path}")
    items: List[MappingItem] = []
    count = 0
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
            count += 1
    logger.info(f"Loaded {count} mapping items.")
    return items


def asset_code_from_mapping_item(mi: MappingItem) -> str:
    """
    Derive a stable, human-readable asset_code.
    Priority:
      1) basename of 'dst' without extension (if present),
      2) fallback to 'asset_{asset_code}'.
    """
    base = os.path.splitext(os.path.basename(mi.dst))[0].strip()
    return base if base else f"asset_{mi.asset_code}"


# ---------------------- Azure GPT-4o (shared client) -------------------

class GPTClient:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_URL"],
            # api_version="2023-07-01-preview", old
            api_version="2025-01-01-preview",
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=60.0
        )
        self._logger = logging.getLogger("taglib.GPTClient")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logging.getLogger("taglib.retry"), logging.WARNING),
    )
    def generate_tag_library(self, overview_data_url: str) -> dict:
        """
        Ask GPT-4o to enumerate unambiguous tags from the overview image.
        Uses a function tool to guarantee strict JSON arguments.
        """
        system_prompt = annotation1_prompt.strip() 

        tools = [{
            "type": "function",
            "function": {
                "name": "return_tag_library",
                "description": "Analyze the overview image to generate base categories and variance analysis keys.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag_library_0": {"type": "array","items": {"type": "string"},"minItems": 3,"description": "List of short, unambiguous tags."},
                        "comparison_keys": {"type": "array","items": {"type": "string"},"minItems": 1,"description": "List of comparison keys defining dimensions of difference between assets."}
                    },
                    "required": ["tag_library_0", "comparison_keys"]
                }
            }
        }]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Enumerate the tag library for all assets visible."},
                {"type": "image_url", "image_url": {"url": overview_data_url}},
            ]},
        ]

        # --- Progress & timing ---
        self._logger.info("Calling Azure Chat Completions for tag library...")
        t0 = time.monotonic()

        # ------------------------ Call Azure Chat Completions ------------------------
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "return_tag_library"}},
        )

        # ---------------------------------------------------------------------------
        # print("Token usage:", resp.usage)
        elapsed = time.monotonic() - t0
        self._logger.info(f"Azure response received in {elapsed:.2f}s")

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
                obj = {"tag_library_0": [], "comparison_keys": []}


        return {
            "tag_library_0": obj.get("tag_library_0", []),
            "comparison_keys": obj.get("comparison_keys", [])
        }
        # # Fallback parsing if tool not used
        # if not tool_calls:
        #     raw = choice.message.get("content") or "{}"
        #     self._logger.debug(f"Raw content length: {len(raw)}")
        #     try:
        #         obj = json.loads(raw)
        #     except Exception:
        #         import re
        #         m = re.search(r"\{[\s\S]*\}", raw)
        #         obj = json.loads(m.group(0)) if m else {"tag_library": []}
        # else:
        #     args_str = tool_calls[0].function.arguments
        #     self._logger.debug(f"Tool arguments length: {len(args_str)}")
        #     obj = json.loads(args_str)

        # out = {}
        # for k in ["tag_library_0"]:
        #     arr = obj.get(k, [])
        #     uniq = []
        #     seen = set()
        #     for t in arr:
        #         s = str(t).strip()
        #         if s and s not in seen:
        #             seen.add(s)
        #             uniq.append(s)
        #     out[k] = uniq
        # return out


# ----------------------------- Main routine ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overview", required=True, help="Path/URL of the overview image (10-asset grid).")
    parser.add_argument("--mapping", required=True, help="Path to mapping.jsonl.")
    parser.add_argument("--out", required=True, help="Path to write annotation.json.")
    parser.add_argument("--folder_name", required=True, help='Folder name (e.g., "snowman").')
    parser.add_argument("--model", default="gpt-4o", help="Azure model name (default: gpt-4o).")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO-level logs.")
    parser.add_argument("--http-debug", action="store_true", help="Enable OpenAI/httpx debug logs (very verbose).")

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, http_debug=args.http_debug)

    logger.info("=== Stage 1: Prepare inputs ===")
    t_stage = time.monotonic()

    overview_data_url = to_data_url(args.overview)
    mapping_items = load_mapping(args.mapping)

    logger.info("=== Stage 2: Generate tag library ===")
    client = GPTClient(model=args.model)
    tag_library = client.generate_tag_library(overview_data_url)

    logger.info("=== Stage 3: Build objects array ===")
    objects = []
    for mi in tqdm(mapping_items, desc="Building objects"):  # progress bar
        objects.append({
            # "asset_code": asset_code_from_mapping_item(mi),
            "asset_code": mi.asset_code,
            "preview_img": mi.preview_img,
            "tags": []  # Part 1 initializes empty tags; Part 2 will fill them.
        })

    logger.info("=== Stage 4: Write annotation.json ===")
    annotation = {
        "folder_name": args.folder_name,
        **tag_library,
        "objects": objects
    }

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)

    total_elapsed = time.monotonic() - t_stage
    logger.info(f"[OK] Wrote {args.out} with {len(tag_library)} tags and {len(objects)} objects. Total={total_elapsed:.2f}s")
    print(f"[OK] Wrote {args.out} with {len(tag_library)} tags and {len(objects)} objects. Total={total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
