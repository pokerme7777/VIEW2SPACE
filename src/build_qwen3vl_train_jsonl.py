#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import testing_prompt as prompt_module


DEFAULT_PROMPT_STYLE = "default"
MINDCUBE_COGMAP_PROMPT_STYLE = "mindcube_cogmap"

def _as_coco_bbox(bbox_val) -> Optional[Tuple[float, float, float, float]]:
    """
    receive dict {"x","y","w","h"} or list/tuple [x,y,w,h], return (x,y,w,h) float; otherwise return None
    """
    if isinstance(bbox_val, dict):
        x, y, w, h = bbox_val.get("x"), bbox_val.get("y"), bbox_val.get("w"), bbox_val.get("h")
        if None in (x, y, w, h):
            return None
        return float(x), float(y), float(w), float(h)
    if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
        x, y, w, h = bbox_val
        return float(x), float(y), float(w), float(h)
    return None


def tool_and_prompt_for_qtype(q_type: str, CoT_option: bool, prompt_style: str = DEFAULT_PROMPT_STYLE) -> Tuple[dict, str]:
    # we dont need tool but need prompt
    if q_type in ["mcq"]:
        if prompt_style == MINDCUBE_COGMAP_PROMPT_STYLE:
            if CoT_option:
                return prompt_module.MCQ_testing_tool, prompt_module.MCQ_mindcube_cogmap_testing_prompt
            else:
                return prompt_module.MCQ_testing_tool, prompt_module.MCQ_mindcube_cogmap_direct_testing_prompt
        if CoT_option:
            return prompt_module.MCQ_testing_tool, prompt_module.MCQ_normal_testing_prompt
        # direct answer
        else: 
            return prompt_module.MCQ_testing_tool, prompt_module.MCQ_normal_direct_testing_prompt

    elif q_type in ["detect"]:
        if CoT_option:
            return prompt_module.detection_testing_tool, prompt_module.detection_normal_testing_prompt
        else: # direct answer
            return prompt_module.detection_testing_tool, prompt_module.detection_normal_direct_testing_prompt

    elif q_type in ["count"]:
        if CoT_option:
            return prompt_module.counting_testing_tool, prompt_module.counting_normal_testing_prompt
        else:
            return prompt_module.counting_testing_tool, prompt_module.counting_normal_direct_testing_prompt
    else:
        raise ValueError(f"Unknown question type: {q_type}")


def _lookup_bboxes(box_map, img_path):
    """Return a list of COCO-format bboxes for given image path.

    Supports:
    - single box: [x, y, w, h]
    - list of boxes: [[x, y, w, h], ...]
    - dict formats that U._as_coco_bbox can handle
    """
    if not isinstance(box_map, dict) or not box_map:
        return []

    # Try full relative path first, then fallback to file name.
    raw = box_map.get(img_path)

    if raw is None:
        raw = box_map.get(Path(img_path).name)

    if raw is None:
        return []

    bboxes = []

    def _add_one(b):
        bbox = _as_coco_bbox(b)
        if bbox is not None:
            bboxes.append(bbox)

    # single frame: dict or [x,y,w,h]
    if isinstance(raw, dict) or (
        isinstance(raw, (list, tuple))
        and len(raw) == 4
        and all(isinstance(v, (int, float)) for v in raw)
    ):
        _add_one(raw)
    # multiple frames：[[...], {...}, ...]
    elif isinstance(raw, (list, tuple)):
        for b in raw:
            _add_one(b)

    return bboxes


def build_messages_one(
    record: Dict[str, Any],
    scenes_root: str,
    include_view_prefix: bool = True,
    CoT_option: bool = True,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    pure_text: bool = False,
) -> Dict[str, Any]:
    """
    Input: overall.jsonl each record
    Output: {"messages":[...]} qwen3-vl + trl training formation (abs image_path+bboxes)
    """
    try:
        q_type = record["q_type"]
    except:
        q_type = record["q_idx"].split("_")[0]
    question = record["question"]
    options = record.get("options") or {}
    rel_image_paths = record.get("image_paths") or []
    draw_boxes_from_q = record.get("supporting", {}).get("draw_boxes")
    draw_boxes = record.get("draw_boxes", draw_boxes_from_q)

    # system prompt：reuse testing prompt for gpt-style training
    _, testing_prompt = tool_and_prompt_for_qtype(q_type, CoT_option, prompt_style=prompt_style)

    # if record.get("question_prompt"):
    #     testing_prompt += record.get("question_prompt")

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": question}]

    if options:
        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
        user_content.append({"type": "text", "text": f"Options:\n{options_str}"})
    
    # images：need to be absolute path with bounding bboxes
    if not pure_text:
        for i, rel_p in enumerate(rel_image_paths):
            abs_p = os.path.join(scenes_root, rel_p)

            #check file exist
            if not os.path.isfile(abs_p):
                raise FileNotFoundError(f"Image file not found: {abs_p}")

            bboxes = _lookup_bboxes(draw_boxes, rel_p)

            if include_view_prefix:
                user_content.append({"type": "text", "text": f"View {i+1}:"})

            # Notice：write absolute image_path + bboxes, do not need to draw now.
            img_item: Dict[str, Any] = {"type": "image", "image_path": abs_p}
            if bboxes:
                img_item["bboxes"] = bboxes
            else:
                img_item["bboxes"] = []  # make sure is empty.

            user_content.append(img_item)

    # assistant target：using gold answer to make them as str
    gold = record.get("supporting", {}).get("chain_of_thought")
    # gold = record.get("answer")
    # gold = f"<answer> {gold} </answer>"
    gold_text = json.dumps(gold, ensure_ascii=False) if isinstance(gold, (dict, list)) else str(gold)

    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": testing_prompt}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": gold_text}]},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_overall_jsonl", required=True, help="Input overall.jsonl (mapping format)")
    ap.add_argument("--out_train_jsonl", required=True, help="Output train jsonl (messages format)")
    ap.add_argument("--scenes_root", required=True, help="Absolute path to scenes root folder")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--prompt_style",
        choices=[DEFAULT_PROMPT_STYLE, MINDCUBE_COGMAP_PROMPT_STYLE],
        default=DEFAULT_PROMPT_STYLE,
        help="Prompt template style to use when building messages.",
    )
    ap.add_argument("--pure_text", action="store_true", help="Build text-only messages without image inputs.")
    args = ap.parse_args()

    n = 0
    with open(args.in_overall_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_train_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out = build_messages_one(
                rec,
                scenes_root=args.scenes_root,
                prompt_style=args.prompt_style,
                pure_text=args.pure_text,
            )
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"[OK] Wrote {n} samples -> {args.out_train_jsonl}")


if __name__ == "__main__":
    main()
