# vllm_unattended_direct.py
# Uses vLLM Python API directly → works on Windows with vllm 0.11.0

import cv2
import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import time
import json
import re
import torch

# ------------------- vLLM IMPORTS -------------------
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.transformers_utils.tokenizer import get_tokenizer
# ----------------------------------------------------

# ------------------- CONFIG -------------------
EVALUATION_MODE = True
VIDEO_FOLDER = "test_videos"
ANNOTATIONS_FOLDER = "test_annotations"
RESULTS_CSV_PATH = "vllm_direct_results.csv"
IOU_THRESHOLD = 0.5

OUT_W, OUT_H = 800, 600

# ------------------- PROMPT (same logic) -------------------
PROMPT = """
<OD>Analyze the airport frame for unattended luggage.
Rules:
- Luggage: handbag, backpack, suitcase
- "unattended" = no person within 1.5× depth-aware radius for ≥2 seconds
- "ABANDONED!" = alone in large radius for >1 second
Return ONLY JSON:
{
  "unattended": [{"bbox": [x1, y1, x2, y2], "label": "suitcase"}],
  "potential": [{"bbox": [x1, y1, x2, y2], "label": "handbag"}]
}
No extra text.
"""

# ------------------- LOAD vLLM MODEL ONCE -------------------
print("Loading Florence-2-large via vLLM (this may take 10–20 sec)...")
llm = LLM(
    model="microsoft/Florence-2-large",
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enforce_eager=True  # Critical for Windows
)
tokenizer = get_tokenizer(llm.llm_engine.tokenizer)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
    stop_token_ids=[tokenizer.eos_token_id]
)
print("Model loaded!")

# ------------------- IMAGE → PROMPT -------------------
def encode_image_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return "data:image/jpeg;base64," + buffer.tobytes().hex()

def vllm_infer(frame):
    try:
        image_url = encode_image_base64(frame)
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": PROMPT}
            ]}
        ]
        # Build fake request to reuse OpenAI logic
        request = ChatCompletionRequest(model="microsoft/Florence-2-large", messages=messages)
        outputs = llm.generate(prompt_token_ids=None, sampling_params=sampling_params, request=request)
        text = outputs[0].outputs[0].text

        # Extract JSON
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return [], []
        data = json.loads(m.group())
        return data.get("unattended", []), data.get("potential", [])
    except Exception as e:
        print(f"vLLM error: {e}")
        return [], []

# ------------------- EVAL HELPERS (unchanged) -------------------
def load_gt(xml_path):
    gt, w, h = {}, None, None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('.//original_size') or root.find('.//meta//size')
        if size is not None:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
        for track in root.findall(".//track"):
            if track.get('label') != 'unattended': continue
            for box in track.findall('box'):
                f = int(box.get('frame'))
                b = [int(float(box.get('xtl'))), int(float(box.get('ytl'))),
                     int(float(box.get('xbr'))), int(float(box.get('ybr')))]
                gt.setdefault(f, []).append(b)
    except Exception as e:
        print(f"XML error: {e}")
    return gt, w, h

def scale_box(b, sx, sy):
    return [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]

def iou(a, b):
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0: return 0
    return inter / ((x2-x1)*(y2-y1) + (X2-X1)*(Y2-Y1) - inter)

def evaluate(pred, gt, thr):
    tp = fp = fn = iou_sum = 0
    matched = [False] * len(gt)
    for p in pred:
        best_iou = 0
        best_idx = -1
        for i, g in enumerate(gt):
            if matched[i]: continue
            cur = iou(p, g)
            if cur > best_iou:
                best_iou = cur
                best_idx = i
        if best_iou >= thr:
            tp += 1
            iou_sum += best_iou
            matched[best_idx] = True
        else:
            fp += 1
    fn = len(gt) - sum(matched)
    return tp, fp, fn, iou_sum

def write_csv(data):
    exists = os.path.isfile(RESULTS_CSV_PATH)
    with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=data.keys())
        if not exists: w.writeheader()
        w.writerow(data)

# ------------------- MAIN LOOP -------------------
if __name__ == "__main__":
    videos = []
    for f in sorted(os.listdir(VIDEO_FOLDER)):
        if not f.lower().endswith('.mp4'): continue
        vp = os.path.join(VIDEO_FOLDER, f)
        xp = os.path.join(ANNOTATIONS_FOLDER, Path(f).stem + ".xml")
        if os.path.exists(xp):
            videos.append({"path": vp, "file": f, "xml": xp})
        else:
            print(f"No XML for {f}")

    for info in videos:
        cap = cv2.VideoCapture(info["path"])
        if not cap.isOpened(): continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out_path = f"annotated_videos/{Path(info['path']).stem}_vllm_direct.mp4"
        os.makedirs("annotated_videos", exist_ok=True)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (OUT_W, OUT_H))

        eval_active = EVALUATION_MODE and os.path.exists(info["xml"])
        gt_by_frame, orig_w, orig_h = ({}, None, None)
        scale_x = scale_y = 1.0
        if eval_active:
            gt_by_frame, orig_w, orig_h = load_gt(info["xml"])
            if orig_w:
                scale_x, scale_y = OUT_W/orig_w, OUT_H/orig_h

        frame_idx = 0
        results = []
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=info["file"])

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (OUT_W, OUT_H))

            start = time.time()
            unattended, potential = vllm_infer(frame)
            infer_ms = (time.time() - start) * 1000

            # Draw
            for box in unattended:
                x1, y1, x2, y2 = map(int, box["bbox"])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, "ABANDONED!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            for box in potential:
                x1, y1, x2, y2 = map(int, box["bbox"])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)
                cv2.putText(frame, "Potential", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

            # Eval
            if eval_active:
                pred = [b["bbox"] for b in unattended]
                gt_raw = gt_by_frame.get(frame_idx, [])
                gt_scaled = [scale_box(b, scale_x, scale_y) for b in gt_raw]
                tp, fp, fn, iou_sum = evaluate(pred, gt_scaled, IOU_THRESHOLD)
                results.append({"tp": tp, "fp": fp, "fn": fn, "iou": iou_sum/tp if tp else 0, "ms": infer_ms})
                for b in gt_scaled:
                    cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,255,255), 2)
                    cv2.putText(frame, "GT", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                frame_idx += 1

            cv2.imshow("vLLM Direct Unattended", frame)
            out.write(frame)
            pbar.update(1)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release(); out.release(); cv2.destroyAllWindows(); pbar.close()

        if eval_active and results:
            tp = sum(r["tp"] for r in results)
            fp = sum(r["fp"] for r in results)
            fn = sum(r["fn"] for r in results)
            prec = tp/(tp+fp) if (tp+fp) else 0
            rec = tp/(tp+fn) if (tp+fn) else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            iou = sum(r["iou"]*r["tp"] for r in results)/tp if tp else 0
            inf = sum(r["ms"] for r in results)/len(results)
            write_csv({
                "video": info["file"],
                "tp": tp, "fp": fp, "fn": fn,
                "precision": f"{prec:.4f}", "recall": f"{rec:.4f}", "f1": f"{f1:.4f}",
                "avg_iou": f"{iou:.4f}", "avg_ms": f"{inf:.1f}"
            })

        print(f"Done → {out_path}")

    print(f"\nAll results in {RESULTS_CSV_PATH}")