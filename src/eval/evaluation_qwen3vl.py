import argparse
import json
import re
import os
import ast
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict

CHOICES = {"A", "B", "C", "D"}

def extract_pred_choice(pred_text: str) -> Optional[str]:
    """
    english:
    Extract the predicted choice from the prediction text.
    returns 'A', 'B', 'C', or 'D' if found, else None.
    1) <answer>...</answer>
    2) "answer is X" / "correct answer is X" / "the correct answer is X" etc.
    3) "Therefore, the correct answer is B: pavilion" / "Therefore ... is B"
    4) fallback: take the last A-D found in the text
    """
    if not pred_text:
        return None

    # 1) <answer>...</answer>
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", pred_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2) "answer is X" / "correct answer is X" / "the correct answer is X", etc.
    m = re.search(
        r"\b(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*([A-D])\b",
        pred_text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # 3) "Therefore, the correct answer is B: pavilion" / "Therefore ... is B"
    m = re.search(r"\b([A-D])\s*[:\-]\s*", pred_text)  # cases like "B: pavilion"
    if m:
        return m.group(1).upper()

    # 4) final fallback: take the last A-D found in the text
    all_letters = re.findall(r"\b([A-D])\b", pred_text.upper())
    if all_letters:
        return all_letters[-1]

    return None


def evaluate_jsonl_mcq(path: str) -> Tuple[float, int, int, Dict[str, Dict[str, Any]]]:
    """
    Return `(acc, correct, total, per_type)`.
    """
    total = 0
    correct = 0
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: JSON decode error: {e}")
                print(line[:500])
                raise

            gt = str(obj.get("answer", "")).strip().upper()
            pred_text = obj.get("prediction", "")

            pred = extract_pred_choice(pred_text)

            q_idx = obj.get("q_idx", "")
            q_type = obj.get("q_type")
            if not q_type:
                q_type = q_idx.split("_")[0] if q_idx else "unknown"
            total += 1
            stats[q_type]["total"] += 1

            if pred == gt:
                correct += 1
                stats[q_type]["correct"] += 1

    overall_acc = correct / total if total > 0 else 0.0

    per_type = {}
    for q_type, s in stats.items():
        t = s["total"]
        c = s["correct"]
        per_type[q_type] = {"total": t, "correct": c, "accuracy": (c / t) if t > 0 else 0.0}

    return overall_acc, correct, total, per_type


def extract_pred_count(pred_text: str) -> Optional[int]:
    """
    counting:
    Extract an integer count from the prediction text.
    returns int if found, else None.

    Priority:
    1) <answer>...</answer>
    2) "answer is X" / "correct answer is X" / "the correct answer is X" etc.
    3) "Therefore, the correct answer is 3: ..." / "Therefore ... is 3"
    4) fallback: take the last integer found in the text
    """
    if not pred_text:
        return None

    # 1) <answer>...</answer>
    m = re.search(
        r"<answer>\s*(-?\d+)\s*</answer>",
        pred_text,
        flags=re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    # 2) "answer is X" / "correct answer is X" / "the correct answer is X"
    m = re.search(
        r"\b(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*(-?\d+)\b",
        pred_text,
        flags=re.IGNORECASE,
    )
    if m:
        return int(m.group(1))

    # 3) "Therefore, the correct answer is 3: ..." / "Therefore ... is 3"
    m = re.search(
        r"\b(-?\d+)\s*[:\-]\s*",
        pred_text,
    )
    if m:
        return int(m.group(1))

    # 4) fallback find last integer
    all_nums = re.findall(r"-?\d+", pred_text)
    if all_nums:
        return int(all_nums[-1])

    return None

def _safe_parse_list(obj_or_str) -> List:
    """
    Ground truth may already be a list, or it may be a string-form list.
    Empty strings and `None` are treated as `[]`.
    """
    if obj_or_str is None:
        return []
    if isinstance(obj_or_str, list):
        return obj_or_str
    s = str(obj_or_str).strip()
    if s == "" or s.lower() == "none":
        return []
    try:
        # Prefer ast.literal_eval because it is safer than eval.
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def sample_miou_matched(gt_boxes, pred_boxes):
    """
    mIoU definition:
    - IoU per sample
    """
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0

    matches = greedy_match_ious(gt_boxes, pred_boxes)

    # IoUs that satisfy the threshold
    valid_ious = [iou for iou in matches ]

    # Number of ground-truth boxes that are missed or below threshold
    num_missed = len(gt_boxes) - len(valid_ious)

    total_iou = sum(valid_ious) + 0.0 * num_missed
    return total_iou / len(gt_boxes)

def _extract_answer_payload(text: str) -> Optional[str]:
    """Return the string inside <answer>...</answer>, or None if not present."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None

def _to_text(pred_text: Any) -> str:
    """
    Normalize model output into a string.
    - If it's already a string: return as-is.
    - If it's bytes: decode.
    - Otherwise: convert to string representation (safe for regex scanning).
    """
    if pred_text is None:
        return ""
    if isinstance(pred_text, str):
        return pred_text
    if isinstance(pred_text, (bytes, bytearray)):
        try:
            return pred_text.decode("utf-8", errors="ignore")
        except Exception:
            return str(pred_text)
    return str(pred_text)

def _parse_pred_boxes(pred_text: str) -> List[List[float]]:
    """
    Parse predicted boxes for detection in exactly two accepted formats:

    A) Wrapped:
       <answer>[[x1, y1, x2, y2], [...]]</answer>

    B) Direct list:
       [[x1, y1, x2, y2], [...]]
    
    Any other format returns [].
    """
    pred_text = _to_text(pred_text).strip()
    if not pred_text:
        return []

    # Case A: use payload inside <answer>...</answer> if present
    payload = _extract_answer_payload(pred_text)

    # Case B: otherwise the whole text must be the list literal
    if payload is None:
        payload = pred_text.strip()

    payload = re.sub(r"\]\s*\[", "], [", payload)
    for _ in range(3):
        payload = re.sub(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", r"\1, \2", payload)


    # Parse list literal (Python/JSON-like lists are OK for literal_eval)
    try:
        v = ast.literal_eval(payload)
    except Exception:
        return []

    # Accept single box format
    if (
        isinstance(v, (list, tuple))
        and len(v) == 4
        and all(isinstance(x, (int, float)) for x in v)
    ):
        v = [list(v)]

    # Validate structure: list of 4-number boxes
    if not isinstance(v, list):
        return []

    out: List[List[float]] = []
    for item in v:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 4
            and all(isinstance(x, (int, float)) for x in item)
        ):
            out.append([float(x) for x in item])

    return out


def _clip_box_xyxy(box: List[float], w: float, h: float) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), w))
    x2 = max(0.0, min(float(x2), w))
    y1 = max(0.0, min(float(y1), h))
    y2 = max(0.0, min(float(y2), h))
    # Ensure x1<=x2 and y1<=y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def convert_gt_boxes_to_pixels(gt_boxes_norm: List[List[float]], width: int = 800, height: int = 600) -> List[List[float]]:
    """
    Ground-truth boxes are normalized xyxy coordinates in the 0-1 range.
    Convert them to pixels via x*width, y*height.
    """
    out = []
    for b in gt_boxes_norm or []:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        x1, y1, x2, y2 = b
        box = [float(x1) * width, float(y1) * height, float(x2) * width, float(y2) * height]
        out.append(_clip_box_xyxy(box, width, height))
    return out


def convert_pred_boxes_to_pixels(pred_boxes_0_1000: List[List[float]], width: int = 800, height: int = 600) -> List[List[float]]:
    """
    Predicted boxes are normalized xyxy coordinates scaled to 0-1000.
    Convert them to pixels via x/1000*width, y/1000*height.
    """
    out = []
    for b in pred_boxes_0_1000 or []:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        x1, y1, x2, y2 = b
        box = [float(x1) / 1000.0 * width, float(y1) / 1000.0 * height,
               float(x2) / 1000.0 * width, float(y2) / 1000.0 * height]
        out.append(_clip_box_xyxy(box, width, height))
    return out

def convert_pred_boxes_to_pixels01(pred_boxes_0_1: List[List[float]], width: int = 800, height: int = 600) -> List[List[float]]:
    """
    Predicted boxes are normalized xyxy coordinates in the 0-1 range.
    Convert them to pixels via x*width, y*height.
    """
    out = []
    for b in pred_boxes_0_1 or []:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        x1, y1, x2, y2 = b
        box = [float(x1)  * width, float(y1)  * height,
               float(x2)  * width, float(y2)  * height]
        out.append(_clip_box_xyxy(box, width, height))
    return out


def box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match_ious(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> List[float]:
    """
    Greedy matching: at each step, take the `(gt, pred)` pair with the
    largest IoU until one side is exhausted.
    Return the IoU list for matched pairs only, without unmatched zeros.
    """
    if not gt_boxes or not pred_boxes:
        return []

    gt_used = [False] * len(gt_boxes)
    pred_used = [False] * len(pred_boxes)

    ious = []
    while True:
        best = (-1.0, -1, -1)
        for i, g in enumerate(gt_boxes):
            if gt_used[i]:
                continue
            for j, p in enumerate(pred_boxes):
                if pred_used[j]:
                    continue
                iou = box_iou_xyxy(g, p)
                if iou > best[0]:
                    best = (iou, i, j)
        if best[1] == -1:
            break
        best_iou, gi, pj = best
        gt_used[gi] = True
        pred_used[pj] = True
        ious.append(best_iou)
        if all(gt_used) or all(pred_used):
            break
    return ious


def sample_miou(gt_boxes: List[List[float]], pred_boxes: List[List[float]]) -> float:
    """
    Per-sample mIoU definition:
    - both gt and pred are empty -> 1.0 (perfect agreement)
    - gt empty and pred non-empty -> 0.0
    - gt non-empty and pred empty -> 0.0
    - otherwise: for each gt box, take the maximum IoU over all predicted
      boxes, then average over gt boxes
      (more stable and robust when pred count differs from gt count)
    """
    if not gt_boxes and not pred_boxes:
        return 1.0
    if not gt_boxes and pred_boxes:
        return 0.0
    if gt_boxes and not pred_boxes:
        return 0.0

    per_gt = []
    for g in gt_boxes:
        best = 0.0
        for p in pred_boxes:
            best = max(best, box_iou_xyxy(g, p))
        per_gt.append(best)
    return sum(per_gt) / len(per_gt) if per_gt else 0.0


def accumulate_tp_fp_fn(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_thr: float
) -> Tuple[int, int, int]:
    """
    Match boxes at the given `iou_thr` and compute TP/FP/FN.
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes and pred_boxes:
        return 0, len(pred_boxes), 0
    if gt_boxes and not pred_boxes:
        return 0, 0, len(gt_boxes)

    # First perform greedy matching; matches above threshold are counted as TP.
    # Note: the greedy matcher itself is threshold-agnostic. The threshold is
    # only used when deciding whether a match counts as TP.
    matches = greedy_match_ious(gt_boxes, pred_boxes)
    tp = sum(1 for iou in matches if iou >= iou_thr)

    # Number of matched pairs = len(matches). Pairs below threshold contribute
    # both one FN and one FP because the gt was not correctly hit and the pred
    # is also a false alarm.
    matched_pairs = len(matches)
    low_iou_pairs = sum(1 for iou in matches if iou < iou_thr)

    # Ground-truth boxes unmatched by any prediction: len(gt)-matched_pairs -> FN
    # Predictions unmatched by any ground-truth box: len(pred)-matched_pairs -> FP
    fn = (len(gt_boxes) - matched_pairs) + low_iou_pairs
    fp = (len(pred_boxes) - matched_pairs) + low_iou_pairs

    return tp, fp, fn


def ap11_from_precision_recall(precision: float, recall: float) -> float:
    """
    Simplified VOC 2007 11-point AP:
    - when there is no confidence/ranking, the PR curve has only one working point `(recall, precision)`
    - after interpolation: precision is the working-point precision for `r<=recall`, otherwise 0
    """
    if precision < 0.0 or recall < 0.0:
        return 0.0
    recall = max(0.0, min(1.0, recall))
    precision = max(0.0, min(1.0, precision))
    # 11 points: 0.0, 0.1, ..., 1.0
    hits = int(recall * 10 + 1e-9) + 1  # number of points satisfying r<=recall
    hits = max(0, min(11, hits))
    return precision * (hits / 11.0)


def evaluate_jsonl_detection(
    path: str,
    width: int = 800,
    height: int = 600,
    iou_thresholds: Optional[List[float]] = None,
    scale_t=True,
) -> Dict[str, Any]:
    """
    Detection metrics:
    - mIoU averaged over samples
    - mAP (historically COCO-style thresholds; since predictions have no score,
      this script uses the simplified VOC 2007 single-point AP approximation)
    Output structure is aligned with the MCQ evaluator:
    {
      "overall": {...},
      "by_q_type": {
         "<type>": {...}
      },
      "note": ...
    }
    """
    # if iou_thresholds is None:
    #     iou_thresholds = [round(0.50 + 0.05 * k, 2) for k in range(10)]
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    # overall accumulators
    overall_total = 0
    overall_miou_sum = 0.0
    overall_thr_stats = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in iou_thresholds}

    # per-q_type accumulators
    per_type = defaultdict(lambda: {
        "total": 0,
        "miou_sum": 0.0,
        "thr_stats": {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in iou_thresholds},
    })

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            q_idx = obj.get("q_idx", "")
            q_type = obj.get("q_type", "")
            if not q_type:
                q_type = q_idx.split("_")[0] if q_idx else "unknown"

            gt_raw = obj.get("answer", "")
            pred_text = obj.get("prediction", "")
            # pred_text = random.sample([
            #     [460, 190, 580, 310],
            #     [450, 190, 570, 310],
            #     [470, 190, 590, 310],
            #     [460, 180, 580, 300],
            #     [450, 180, 570, 300],
            # ], random.randint(1,5))
            # pred_text = [[454, 223, 518, 249]]

            gt_boxes_norm = _safe_parse_list(gt_raw)
            pred_boxes_0_1000 = _parse_pred_boxes(pred_text)

            gt_boxes = convert_pred_boxes_to_pixels(gt_boxes_norm, width=width, height=height)
            if scale_t:
                pred_boxes = convert_pred_boxes_to_pixels(pred_boxes_0_1000, width=width, height=height)
            else:
                pred_boxes = convert_pred_boxes_to_pixels01(pred_boxes_0_1000, width=width, height=height)

            # overall
            overall_total += 1
            smiou = sample_miou(gt_boxes, pred_boxes)
            # smiou = sample_miou_matched(gt_boxes, pred_boxes)
            overall_miou_sum += smiou

            # per type
            per_type[q_type]["total"] += 1
            per_type[q_type]["miou_sum"] += smiou

            for thr in iou_thresholds:
                tp, fp, fn = accumulate_tp_fp_fn(gt_boxes, pred_boxes, iou_thr=thr)

                overall_thr_stats[thr]["tp"] += tp
                overall_thr_stats[thr]["fp"] += fp
                overall_thr_stats[thr]["fn"] += fn

                per_type[q_type]["thr_stats"][thr]["tp"] += tp
                per_type[q_type]["thr_stats"][thr]["fp"] += fp
                per_type[q_type]["thr_stats"][thr]["fn"] += fn

    def _summarize(total: int, miou_sum: float, thr_stats: Dict[float, Dict[str, int]]) -> Dict[str, Any]:
        miou = miou_sum / total if total > 0 else 0.0

        ap_by_thr = {}
        for thr, s in thr_stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            ap_by_thr[thr] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                # "ap11": ap11_from_precision_recall(precision, recall),
            }

        # mAP = sum(v["ap11"] for v in ap_by_thr.values()) / len(ap_by_thr) if ap_by_thr else 0.0
        f1_50 = ap_by_thr.get(0.50, {}).get("f1", 0.0)

        return {
            "total": total,
            "miou": miou,
            "f1@50": f1_50,
            "by_iou_threshold": ap_by_thr,
        }

    overall = _summarize(overall_total, overall_miou_sum, overall_thr_stats)

    by_q_type = {}
    for qt, s in per_type.items():
        by_q_type[qt] = _summarize(s["total"], s["miou_sum"], s["thr_stats"])

    return {
        "overall": overall,
        "by_q_type": by_q_type,
    }


def evaluate_jsonl_counting(
    path: str
) -> Dict[str, Any]:
    """
    counting metric:
    - ACC
    - MSE
    {
      "overall": {...},
      "by_q_type": {
         "<type>": {...}
      },
      "note": ...
    }
    """

    # overall accumulators
    overall_total = 0
    overall_correct = 0
    overall_sqerr_sum = 0.0
    overall_error_num = 0 

    overall_abs_err_sum = 0.0


    # per-q_type accumulators
    per_type = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "sqerr_sum": 0.0,
        "error_num": 0,
        "abs_err_sum": 0.0,
    })

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            q_idx = obj.get("q_idx", "")
            q_type = obj.get("q_type", "")
            if not q_type:
                q_type = q_idx.split("_")[0] if q_idx else "unknown"

            gt_raw = obj.get("answer", "")
            pred_text = obj.get("prediction", "")

            gt = gt_raw
            pred = extract_pred_count(str(pred_text))
            # pred = 2

            # overall
            overall_total += 1
            per_type[q_type]["total"] += 1

            if gt is None or pred is None:
                overall_error_num += 1
                per_type[q_type]["error_num"] += 1
                continue

            gt_i = int(round(gt))
            pred_i = int(round(pred))

            if pred_i == gt_i:
                overall_correct += 1
                per_type[q_type]["correct"] += 1

            err = (pred_i - gt_i)
            overall_sqerr_sum += float(err * err)
            per_type[q_type]["sqerr_sum"] += float(err * err)

            err = abs(pred_i - gt_i)
            overall_abs_err_sum += err
            per_type[q_type]["abs_err_sum"] += err

    def _summarize(total: int, correct: int, sqerr_sum: float, error_num: int, abs_err_sum: float) -> Dict[str, Any]:
        valid = max(total - error_num, 0)
        acc = (correct / valid) if valid > 0 else 0.0
        mse = (sqerr_sum / valid) if valid > 0 else 0.0
        mae = (abs_err_sum / valid) if valid > 0 else 0.0

        return {
            "total": total,
            "valid": valid,
            "error_num": error_num,
            "correct": correct,
            "acc": acc,
            "mse": mse,
            "mae": mae,
        }

    overall = _summarize(overall_total, overall_correct, overall_sqerr_sum, overall_error_num, overall_abs_err_sum)

    by_q_type = {}
    for qt, s in per_type.items():
        by_q_type[qt] = _summarize(s["total"], s["correct"], s["sqerr_sum"], s["error_num"], s["abs_err_sum"])

    return {
        "overall": overall,
        "by_q_type": by_q_type,
        "note": "ACC/MSE computed on valid samples only (valid = total - error_num). Counting treated as integer via round().",
    }
# -----------------------------
# Saving
# -----------------------------
def save_evaluation_result(output_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path

def build_argparser():
    ap = argparse.ArgumentParser(description="Evaluate prediction JSONL files for view2space-v1 public release.")
    ap.add_argument("--predictions_jsonl", required=True, help="Path to predictions.jsonl produced by test_qwen3vl.py.")
    ap.add_argument("--task_mode", choices=["mcq", "detect", "count"], required=True, help="Question family to evaluate.")
    ap.add_argument("--output_dir", default=None, help="Directory to write evaluation_result.json. Defaults to the predictions file directory.")
    ap.add_argument("--model_name", default=None, help="Optional model name recorded in evaluation_result.json.")
    ap.add_argument("--width", type=int, default=800, help="Image width used when converting 0-1000 detection outputs to pixels.")
    ap.add_argument("--height", type=int, default=600, help="Image height used when converting 0-1000 detection outputs to pixels.")
    ap.add_argument(
        "--scale_1000",
        action="store_true",
        help="Interpret detection predictions as 0-1000 normalized coordinates and convert them to pixels before scoring.",
    )
    return ap


def main():
    args = build_argparser().parse_args()
    jsonl_path = args.predictions_jsonl
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(jsonl_path))
    model_name = args.model_name or os.path.basename(os.path.dirname(os.path.abspath(jsonl_path)))

    result = {
        "model_name": model_name,
        "source_path": jsonl_path,
        "source_file": os.path.basename(jsonl_path),
        "task_mode": args.task_mode,
    }

    if args.task_mode == "mcq":
        overall_acc, correct, total, per_type = evaluate_jsonl_mcq(jsonl_path)
        print(f"[MCQ] Total samples: {total}")
        print(f"[MCQ] Correct: {correct}")
        print(f"[MCQ] Accuracy: {overall_acc:.4f}")
        print("\n[MCQ] Accuracy by q_type:")
        for q_type, stats in sorted(per_type.items(), key=lambda kv: kv[1]["total"], reverse=True):
            print(f"- {q_type}: acc={stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        result["mcq"] = {
            "overall": {"total": total, "correct": correct, "accuracy": overall_acc},
            "by_q_type": per_type,
        }

    elif args.task_mode == "detect":
        det_metrics = evaluate_jsonl_detection(
            jsonl_path,
            width=args.width,
            height=args.height,
            scale_t=args.scale_1000,
        )
        print("\n[Det][Overall] mIoU:", f"{det_metrics['overall']['miou']:.4f}")
        print("[Det][Overall] F1@50:", f"{det_metrics['overall']['f1@50']:.4f}")
        print("\n[Det] Metrics by q_type:")
        for q_type, stats in sorted(det_metrics["by_q_type"].items(), key=lambda kv: kv[1]["total"], reverse=True):
            print(f"- {q_type}: mIoU={stats['miou']:.4f}, F1@50={stats['f1@50']:.4f} (n={stats['total']})")
        result["detection"] = det_metrics

    else:
        count_metrics = evaluate_jsonl_counting(jsonl_path)
        print("[Count][Overall] ACC:", f"{count_metrics['overall']['acc']:.4f}")
        print("[Count][Overall] MSE:", f"{count_metrics['overall']['mse']:.4f}")
        print("[Count][Overall] MAE:", f"{count_metrics['overall']['mae']:.4f}")
        print("\n[Count] Metrics by q_type:")
        for q_type, stats in sorted(count_metrics["by_q_type"].items(), key=lambda kv: kv[1]["total"], reverse=True):
            print(
                f"- {q_type}: ACC={stats['acc']:.4f}, MSE={stats['mse']:.4f}, "
                f"MAE={stats['mae']:.4f} (n={stats['total']}), ErrorNum={stats['error_num']}"
            )
        result["count"] = count_metrics

    output_path = save_evaluation_result(output_dir=output_dir, payload=result)
    print(f"\nSaved evaluation result to: {output_path}")


if __name__ == "__main__":
    main()
