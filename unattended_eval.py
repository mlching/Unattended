import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time
import os
from collections import deque
import xml.etree.ElementTree as ET
import csv
from pathlib import Path

# --- NEW: BATCH EVALUATION CONFIGURATION ---
# !! UPDATED FOR YOUR FOLDER STRUCTURE !!
EVALUATION_MODE = True  # Set to True to run evaluation
VIDEO_FOLDER = "test_videos"      # Folder with 1.mp4, 2.mp4, ...
ANNOTATIONS_FOLDER = "test_annotations"  # Folder with 1.xml, 2.xml, ...
IOU_THRESHOLD = 0.5
RESULTS_CSV_PATH = "./unattended_results_summary.csv"

# Define all models you want to test
MODELS_TO_EVALUATE = [
    {
        'name': 'yolov8s',
        'path': './weights/yolov8s.pt'
    },
    {
        'name': 'yolov8m',
        'path': './weights/yolov8m.pt'
    },
    {
        'name': 'yolo11s',
        'path': './weights/yolo11s.pt'
    }
]

# --- AUTO-GENERATE VIDEO LIST FROM test_videos FOLDER ---
VIDEOS_TO_PROCESS = []
for video_file in sorted(os.listdir(VIDEO_FOLDER)):
    if video_file.lower().endswith('.mp4'):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        video_id = Path(video_file).stem  # e.g., "1", "2"
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{video_id}.xml")
        
        if os.path.exists(annotation_path):
            VIDEOS_TO_PROCESS.append({
                'path': video_path,
                'filename': video_file,
                'annotation_xml': annotation_path  # NEW: individual XML per video
            })
        else:
            print(f"Warning: No annotation found for {video_file} at {annotation_path}. Skipping.")
# --- END CONFIGURATION ---


# --- MODIFIED: LOAD INDIVIDUAL CVAT XML PER VIDEO ---
def load_cvat_annotations(xml_path):
    """
    Parses a single CVAT XML file (one video per file).
    Returns: gt_boxes_by_frame {frame_idx: [[x1,y1,x2,y2], ...]}, original_w, original_h
    """
    print(f"Loading annotations from '{xml_path}'...")
    gt_boxes_by_frame = {}
    original_w, original_h = None, None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get original size
        size_tag = root.find('.//original_size')
        if size_tag is not None:
            original_w = int(size_tag.find('width').text)
            original_h = int(size_tag.find('height').text)
        else:
            meta = root.find('.//meta')
            if meta:
                size = meta.find('.//size')
                if size is not None:
                    original_w = int(size.find('width').text)
                    original_h = int(size.find('height').text)

        if original_w is None:
            print("Warning: Could not find original size. Using default 1920x1080.")
            original_w, original_h = 1920, 1080

        # Look for tracks with label "unattended" (or change to your label)
        label_name = 'unattended'  # CHANGE IF YOUR LABEL IS DIFFERENT (e.g., "abandoned")
        found_tracks = 0

        for track in root.findall(".//track"):
            if track.get('label') != label_name:
                continue
            found_tracks += 1
            for box in track.findall('box'):
                frame_num = int(box.get('frame'))
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                box_coords = [int(xtl), int(ytl), int(xbr), int(ybr)]
                
                if frame_num not in gt_boxes_by_frame:
                    gt_boxes_by_frame[frame_num] = []
                gt_boxes_by_frame[frame_num].append(box_coords)
        
        print(f"Loaded {found_tracks} tracks from {xml_path}")
        return gt_boxes_by_frame, original_w, original_h

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return {}, None, None

# --- KEEP ALL OTHER HELPER FUNCTIONS UNCHANGED ---
def scale_gt_box(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea

def evaluate_frame(pred_boxes, gt_boxes, iou_threshold):
    tp = fp = fn = 0
    iou_sum = 0.0
    
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, 0.0
    
    if not pred_boxes:
        return 0, 0, len(gt_boxes), 0.0
        
    if not gt_boxes:
        return 0, len(pred_boxes), 0, 0.0

    matched_gt = [False] * len(gt_boxes)
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            if matched_gt[i]: continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            iou_sum += best_iou
            if best_gt_idx != -1:
                matched_gt[best_gt_idx] = True
        else:
            fp += 1
            
    fn = len(gt_boxes) - sum(matched_gt)
    return tp, fp, fn, iou_sum

def write_summary_to_csv(csv_path, summary_data):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_data)

def write_frame_data_to_csv(csv_path, all_frame_results, model_name, video_name):
    if not all_frame_results: return
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['model_name', 'video_name'] + list(all_frame_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_frame_results:
            row_data = row.copy()
            row_data['model_name'] = model_name
            row_data['video_name'] = video_name
            writer.writerow(row_data)

# --- ORIGINAL HELPER FUNCTIONS (unchanged) ---
color_mapping = {
    0: (255, 0, 255),
    1: (0, 255, 0),
}

def calculate_distance_squared(coord1, coord2):
    return (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2

def has_movement(positions, threshold, frame_number=5):
    if len(positions) < frame_number:
        return True
    pts = np.array(positions)
    avg = pts.mean(axis=0)
    for p in pts:
        if calculate_distance_squared(p, avg) >= threshold**2:
            return True
    return False

# --- MAIN PROCESSING FUNCTION (only small changes) ---
def run_evaluation_on_video(model, model_name_for_csv, video_path, video_filename, annotation_xml):
    print(f"Processing '{video_filename}' with model '{model_name_for_csv}'")
    
    out_w, out_h = 800, 600

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_stem = Path(video_path).stem
    output_path = f"./annotated_videos/{video_stem}_{model_name_for_csv}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Reset state
    person_coords = {}; suitcase_coords = {}; suitcase_positions_history = {}
    suitcase_timestamps = {}; suitcase_initial_people = {}; connections = {}
    abandoned_suitcases = set(); moving_suitcases = set()

    movement_threshold = 10
    abandonment_time = 1
    radius = 150
    frame_number = 10

    os.makedirs('screenshots', exist_ok=True)
    os.makedirs('started_moving', exist_ok=True)

    # --- EVALUATION SETUP (now uses individual XML) ---
    all_frame_results = []
    frame_idx = 0
    current_eval_mode = EVALUATION_MODE

    if current_eval_mode:
        print("--- RUNNING IN EVALUATION MODE ---")
        gt_annotations, original_w, original_h = load_cvat_annotations(annotation_xml)
        
        if original_w is None:
            print("Disabling evaluation due to missing annotations.")
            current_eval_mode = False
            scale_x = scale_y = 1.0
        else:
            scale_x = out_w / original_w
            scale_y = out_h / original_h
            print(f"Scaling: {original_w}x{original_h} → {out_w}x{out_h} (x{scale_x:.3f}, y{scale_y:.3f})")

    # --- MAIN LOOP (unchanged except evaluation part) ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.resize(frame, (out_w, out_h))
        eval_start_time = time.time()
        results_list = list(model.track([frame], persist=True, classes=[0, 24, 26, 28], stream=True))
        eval_inference_time = (time.time() - eval_start_time) * 1000

        annotator = Annotator(frame)
        r = results_list[0] if results_list else None
        frame_boxes = r.boxes if r else None

        current_person_ids = set()
        current_suitcase_ids = set()

        if frame_boxes:
            for box in frame_boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                c = int(box.cls)
                color = color_mapping.get(c, (255, 0, 0))
                class_name = model.names[c]
                object_id = int(box.id[0]) if box.id is not None else -1
                label_text = f"{class_name} ({object_id})"
                annotator.box_label(b, label_text, color=color)

                if c == 0:
                    person_coords[object_id] = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                    current_person_ids.add(object_id)
                elif c in [24, 26, 28]:
                    suitcase_coords[object_id] = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                    current_suitcase_ids.add(object_id)

                    if object_id not in suitcase_positions_history:
                        suitcase_positions_history[object_id] = deque(maxlen=frame_number)
                    suitcase_positions_history[object_id].append(suitcase_coords[object_id])

                    # movement logic...
                    current_time = time.time()
                    if object_id in suitcase_timestamps:
                        prev_center, last_moved_time, was_moving = suitcase_timestamps[object_id]
                        is_moving_now = has_movement(suitcase_positions_history[object_id], movement_threshold, frame_number)
                        if is_moving_now:
                            suitcase_timestamps[object_id] = (suitcase_coords[object_id], current_time, True)
                            cv2.putText(frame, "Moving", (b[0], b[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            if object_id in abandoned_suitcases:
                                abandoned_suitcases.discard(object_id)
                        else:
                            suitcase_timestamps[object_id] = (prev_center, last_moved_time, False)
                            cv2.putText(frame, "Not Moving", (b[0], b[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        suitcase_timestamps[object_id] = (suitcase_coords[object_id], current_time, True)
                        nearby_people = {pid for pid, pcoord in person_coords.items() 
                                       if calculate_distance_squared(pcoord, suitcase_coords[object_id]) < radius ** 2}
                        suitcase_initial_people[object_id] = nearby_people
                        connections[object_id] = nearby_people

        # --- [Rest of your original logic unchanged] ---
        # Cleanup disappeared objects
        for pid in set(person_coords.keys()) - current_person_ids:
            person_coords.pop(pid, None)
            for s in connections.values():
                s.discard(pid)

        for sid in set(suitcase_coords.keys()) - current_suitcase_ids:
            suitcase_coords.pop(sid, None)
            suitcase_timestamps.pop(sid, None)
            suitcase_initial_people.pop(sid, None)
            connections.pop(sid, None)
            suitcase_positions_history.pop(sid, None)

        # Update connections
        for sid in list(connections.keys()):
            connections[sid] = {pid for pid in connections[sid] 
                              if pid in person_coords and 
                                 calculate_distance_squared(person_coords[pid], suitcase_coords[sid]) < radius ** 2}
            if not connections[sid]:
                del connections[sid]

        # Draw lines
        for sid, pids in connections.items():
            sc = suitcase_coords.get(sid)
            if not sc: continue
            for pid in pids:
                pc = person_coords.get(pid)
                if pc:
                    cv2.line(frame, (int(pc[0]), int(pc[1])), (int(sc[0]), int(sc[1])), (255, 0, 255), 2)

        # Abandonment logic
        current_time = time.time()
        for sid, (center, last_moved, is_moving) in list(suitcase_timestamps.items()):
            if sid not in suitcase_coords: continue
            has_owner = len(connections.get(sid, set())) > 0
            if not is_moving and not has_owner:
                elapsed = current_time - last_moved
                if elapsed > abandonment_time and sid not in abandoned_suitcases:
                    abandoned_suitcases.add(sid)
                    b = next((box.xyxy[0].cpu().numpy().astype(int) for box in frame_boxes 
                             if box.id is not None and int(box.id[0]) == sid), None)
                    if b is not None:
                        cv2.putText(frame, "ABANDONED!", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        cv2.imwrite(f"screenshots/{video_stem}_abandoned_{sid}_{int(current_time)}.png", frame)
                    print(f"ALERT: Luggage {sid} is UNATTENDED!")
            else:
                abandoned_suitcases.discard(sid)

        # Draw persistent ABANDONED
        for sid in abandoned_suitcases:
            if sid in suitcase_coords:
                b = next((box.xyxy[0].cpu().numpy().astype(int) for box in frame_boxes 
                         if box.id is not None and int(box.id[0]) == sid), None)
                if b is not None:
                    cv2.putText(frame, "ABANDONED!", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # --- EVALUATION (updated to use per-video XML) ---
        if current_eval_mode:
            current_predicted_boxes = []
            if frame_boxes:
                for sid in abandoned_suitcases:
                    b = next((box.xyxy[0].cpu().numpy().astype(int) for box in frame_boxes 
                             if box.id is not None and int(box.id[0]) == sid), None)
                    if b is not None:
                        current_predicted_boxes.append(b)

            raw_gt = gt_annotations.get(frame_idx, [])
            scaled_gt = [scale_gt_box(box, scale_x, scale_y) for box in raw_gt]

            tp, fp, fn, iou_sum = evaluate_frame(current_predicted_boxes, scaled_gt, IOU_THRESHOLD)
            tn = 1 if not scaled_gt and not current_predicted_boxes else 0
            avg_iou = iou_sum / tp if tp > 0 else 0.0

            all_frame_results.append({
                'frame': frame_idx, 'tp': tp, 'fp': fp, 'fn': fn, 'tn_frame': tn,
                'avg_iou': avg_iou, 'inference_time_ms': eval_inference_time,
                'gt_count': len(scaled_gt), 'pred_count': len(current_predicted_boxes)
            })

            # Draw GT boxes
            for box in scaled_gt:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                cv2.putText(frame, "GT", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            frame_idx += 1

        # Show & save
        annotated = annotator.result()
        cv2.imshow("YOLOv8", annotated)
        out.write(annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); out.release(); cv2.destroyAllWindows()

    # Save results
    if current_eval_mode and all_frame_results:
        total_tp = sum(r['tp'] for r in all_frame_results)
        total_fp = sum(r['fp'] for r in all_frame_results)
        total_fn = sum(r['fn'] for r in all_frame_results)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou_all = sum(r['avg_iou'] * r['tp'] for r in all_frame_results) / total_tp if total_tp > 0 else 0
        avg_inf = sum(r['inference_time_ms'] for r in all_frame_results) / len(all_frame_results)

        summary = {
            'model_name': model_name_for_csv, 'video_name': video_filename,
            'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
            'precision': f"{precision:.4f}", 'recall': f"{recall:.4f}", 'f1_score': f"{f1:.4f}",
            'overall_avg_iou': f"{avg_iou_all:.4f}", 'avg_inference_ms': f"{avg_inf:.2f}",
            'total_frames': len(all_frame_results), 'iou_threshold': IOU_THRESHOLD
        }
        write_summary_to_csv(RESULTS_CSV_PATH, summary)
        #write_frame_data_to_csv(f"./{video_stem}_{model_name_for_csv}_per_frame.csv", all_frame_results, model_name_for_csv, video_filename)

    print(f"Done → {output_path}")
    print(f'metrics -> {RESULTS_CSV_PATH}')


# --- MAIN ---
def main():
    if not EVALUATION_MODE:
        print("Set EVALUATION_MODE = True")
        return

    for model_info in MODELS_TO_EVALUATE:
        print(f"\nLoading model: {model_info['name']}")
        try:
            model = YOLO(model_info['path'])
            model_name = model_info['name']
        except Exception as e:
            print(f"Failed to load {model_info['path']}: {e}")
            continue

        for video_info in VIDEOS_TO_PROCESS:
            print(f"\n→ Processing {video_info['filename']}")
            run_evaluation_on_video(
                model, model_name,
                video_info['path'],
                video_info['filename'],
                video_info['annotation_xml']
            )

    print("\nALL DONE!")

if __name__ == "__main__":
    main()