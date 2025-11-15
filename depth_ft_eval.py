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

# ---------------------------
#       EVALUATION & BATCH CONFIGURATION (from unattended_eval.py)
# ---------------------------
EVALUATION_MODE = True  # Set to True to run evaluation
VIDEO_FOLDER = "test_videos"      # Folder with 1.mp4, 2.mp4, ...
ANNOTATIONS_FOLDER = "test_annotations"  # Folder with 1.xml, 2.xml, ...
IOU_THRESHOLD = 0.5
RESULTS_CSV_PATH = "./depth_ft_results_summary.csv"

MODELS_TO_EVALUATE = [
    {
        'name': 'custom',
        'path': './weights/CCTV-Korzo_Dusserdorf.pt'
    }
    # Add more models if you want to run them
]

# ---------------------------
#       DEPTH-SPECIFIC CONFIG (from original depth.py)
# ---------------------------
# Toggle features
draw_radius = True
draw_connections_flag = True
use_depth_radius = True  # Scale radius by depth map

# Base radius for proximity checks and drawing circles (used if depth is disabled or invalid)
default_fallback_radius = 5
half_radius_multiplier = 0.5  # For the 'new connection' logic

# Depth Map Settings
depth_map_path = "./weights/dusslerdorf_aiport_2024_26_02_12_20pm_first_frame_depth.npy"
depth_map_scale_factor = 2.0
depth_coefficient = 500.0
power_factor = 0.6
visual_max_radius_limit = 1000

# Movement & abandonment settings (adapted)
movement_threshold = 30
abandonment_time = 0.01  # seconds stationary before flagged as 'potentially abandoned'
definite_abandonment_time = 0.01  # seconds alone and stationary for definite abandonment
definite_abandonment_radius_multiplier = 1.5
frame_history = 30  # history length for movement detection
new_connection_time_threshold = 1

# Misc
out_w, out_h = 800, 600

# ---------------------------
#       Helper functions (merged)
# ---------------------------
def calculate_distance_squared(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def has_movement(pos_deque, thr_sq, history_len):
    if len(pos_deque) < history_len:
        return True  # Consider it moving if not enough history
    start = pos_deque[0]
    for p in list(pos_deque)[1:history_len]:
        if calculate_distance_squared(p, start) >= thr_sq:
            return True
    return False

def get_depth_based_radius(cx, cy, depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius_val):
    current_radius = default_fallback_radius_val
    circle_color = (0, 255, 255)  # Default yellow-cyan

    if depth_map is not None and 0 <= cx < w and 0 <= cy < h:
        d = float(depth_map[cy, cx])
        if d > 0:
            try:
                denominator = d**power_factor
                if denominator == 0:
                    r = visual_max_radius_limit
                else:
                    r = int(depth_coefficient / denominator)

                if r > visual_max_radius_limit:
                    current_radius = visual_max_radius_limit
                    circle_color = (0, 0, 255)  # Red for max radius
                else:
                    current_radius = r
            except (OverflowError, ZeroDivisionError):
                current_radius = visual_max_radius_limit
                circle_color = (0, 0, 255)  # Red for error/max
        else:
            current_radius = default_fallback_radius_val
            circle_color = (128, 128, 128)  # Grey for invalid depth
    else:
        current_radius = default_fallback_radius_val
        circle_color = (128, 128, 128)  # Grey for out of bounds/no depth map

    return max(1, current_radius), circle_color  # Ensure radius is at least 1 for drawing

# Evaluation helpers (from unattended_eval.py)
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
            if matched_gt[i]:
                continue
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

# ---------------------------
#       Video list generation (unchanged)
# ---------------------------
VIDEOS_TO_PROCESS = []
for video_file in sorted(os.listdir(VIDEO_FOLDER)):
    if video_file.lower().endswith('.mp4'):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        video_id = Path(video_file).stem
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, f"{video_id}.xml")

        if os.path.exists(annotation_path):
            VIDEOS_TO_PROCESS.append({
                'path': video_path,
                'filename': video_file,
                'annotation_xml': annotation_path
            })
        else:
            print(f"Warning: No annotation found for {video_file} at {annotation_path}. Skipping.")

# ---------------------------
#       Main processing (merge of unattended_eval main loop + depth logic)
# ---------------------------
def run_evaluation_on_video(model, model_name_for_csv, video_path, video_filename, annotation_xml):
    print(f"Processing '{video_filename}' with model '{model_name_for_csv}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_stem = Path(video_path).stem
    output_path = f"./annotated_videos/{video_stem}_{model_name_for_csv}_annotated.mp4"
    os.makedirs('./annotated_videos', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Reset state (use depth.py structures and names)
    person_coords = {}
    person_history = {}
    person_info = {}

    suitcase_coords = {}
    suitcase_history = {}
    suitcase_info = {}
    suitcase_init_people = {}
    connections = {}

    potential_abandoned_timer = {}
    definite_abandoned_set = set()

    potential_new_connections_timer = {}

    movement_threshold_sq = movement_threshold ** 2

    os.makedirs('screenshots', exist_ok=True)
    os.makedirs('started_moving', exist_ok=True)

    # Load depth map (if enabled)
    depth_map = None
    if use_depth_radius:
        try:
            loaded_depth_map = np.load(depth_map_path)
            depth_map = cv2.resize(loaded_depth_map, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            depth_map = depth_map * depth_map_scale_factor
            print(f"Depth map loaded for video {video_filename}, resized to ({out_w}, {out_h}).")
        except Exception as e:
            print(f"Error loading depth map: {e}. Disabling depth-based radii for this run.")
            depth_map = None

    # Evaluation setup
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

    # Main loop adapted from depth.py but matching unattended_eval structure
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (out_w, out_h))
        t_now = time.time()

        eval_start_time = time.time()
        # use stream=True to speed up; persist behaviour is handled by results objects
        results_list = list(model.track([frame], persist=True, classes=[0, 1], stream=True))
        eval_inference_time = (time.time() - eval_start_time) * 1000

        annotator = Annotator(frame)
        r = results_list[0] if results_list else None
        frame_boxes = r.boxes if r else None

        current_persons = set()
        current_suitcases = set()

        # Process detections using depth.py logic
        if frame_boxes is not None:
            for box in frame_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls)
                obj_id = int(box.id[0]) if box.id is not None else -1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                name = model.names.get(cls, f"Class {cls}")
                color = (255, 0, 255) if cls != 0 else (0, 255, 0)
                label = f"{name} ({obj_id})"
                annotator.box_label((x1, y1, x2, y2), label, color=color)

                # Luggage classes
                if cls == 0:
                    suitcase_coords[obj_id] = (cx, cy)
                    current_suitcases.add(obj_id)

                    if obj_id not in suitcase_history:
                        suitcase_history[obj_id] = deque(maxlen=frame_history)
                        suitcase_info[obj_id] = ((cx, cy), t_now, True)

                        initial_associated_people = set()
                        for pid, p_coord in person_coords.items():
                            person_rad, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                            if calculate_distance_squared(p_coord, (cx, cy)) < person_rad ** 2:
                                initial_associated_people.add(pid)
                        suitcase_init_people[obj_id] = initial_associated_people
                        connections[obj_id] = initial_associated_people.copy()
                        print(f"Luggage {obj_id} first seen. Initial owner(s): {suitcase_init_people[obj_id]}")

                    else:
                        prev_c, ts_luggage_last_moved, was_moving_prev_frame = suitcase_info.get(obj_id, ((cx, cy), t_now, True))
                        suitcase_history[obj_id].append((cx, cy))
                        luggage_is_moving = has_movement(suitcase_history[obj_id], movement_threshold_sq, frame_history)

                        if luggage_is_moving:
                            if not was_moving_prev_frame:
                                cv2.imwrite(f"started_moving/luggage_{obj_id}_started_moving_{int(t_now)}.png", frame)
                                print(f"Luggage {obj_id} STARTED MOVING at {time.ctime(t_now)}. RESETTING ABANDONMENT TIMERS/FLAGS.")

                                potential_abandoned_timer.pop(obj_id, None)
                                definite_abandoned_set.discard(obj_id)
                                potential_new_connections_timer.pop(obj_id, None)

                                current_owners_on_move = set()
                                for pid, p_coord in person_coords.items():
                                    person_rad, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                                    if calculate_distance_squared(p_coord, (cx, cy)) < person_rad ** 2:
                                        current_owners_on_move.add(pid)
                                suitcase_init_people[obj_id] = current_owners_on_move
                                connections[obj_id] = current_owners_on_move.copy()
                                if current_owners_on_move:
                                    print(f"Luggage {obj_id} re-assigned owner(s) on move: {current_owners_on_move}")
                                else:
                                    print(f"Luggage {obj_id} moving with no clear owner nearby.")

                            suitcase_info[obj_id] = ((cx, cy), t_now, True)
                            cv2.putText(frame, "Moving", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            suitcase_info[obj_id] = ((cx, cy), ts_luggage_last_moved, False)
                            cv2.putText(frame, "Not Moving", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # PERSON LOGIC
                elif cls == 1:
                    person_coords[obj_id] = (cx, cy)
                    current_persons.add(obj_id)

                    if obj_id not in person_history:
                        person_history[obj_id] = deque(maxlen=frame_history)
                        person_info[obj_id] = ((cx, cy), t_now, True)
                    person_history[obj_id].append((cx, cy))

                    prev_p_c, prev_p_ts, prev_p_mov = person_info.get(obj_id, ((cx, cy), t_now, True))
                    person_is_moving = has_movement(person_history[obj_id], movement_threshold_sq, frame_history)
                    person_info[obj_id] = ((cx, cy), t_now if person_is_moving else prev_p_ts, person_is_moving)

                    if draw_radius:
                        current_person_radius, circle_color_person = get_depth_based_radius(cx, cy, depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                        cv2.circle(frame, (cx, cy), current_person_radius, circle_color_person, 2)
                        cv2.putText(frame, f"R:{current_person_radius}", (cx + 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        half_person_radius = int(current_person_radius * half_radius_multiplier)
                        cv2.circle(frame, (cx, cy), half_person_radius, (0, 128, 255), 1)
                        cv2.putText(frame, f"R/2:{half_person_radius}", (cx + 20, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1)

        # Cleanup missing objects (people)
        for pid in list(person_coords.keys()):
            if pid not in current_persons:
                person_coords.pop(pid, None)
                person_history.pop(pid, None)
                person_info.pop(pid, None)
                for s_id in connections:
                    connections[s_id].discard(pid)
                for s_id in potential_new_connections_timer:
                    potential_new_connections_timer[s_id].pop(pid, None)

        # Cleanup missing suitcases
        for sid in list(suitcase_coords.keys()):
            if sid not in current_suitcases:
                suitcase_coords.pop(sid, None)
                suitcase_history.pop(sid, None)
                suitcase_info.pop(sid, None)
                suitcase_init_people.pop(sid, None)
                connections.pop(sid, None)
                potential_abandoned_timer.pop(sid, None)
                definite_abandoned_set.discard(sid)
                potential_new_connections_timer.pop(sid, None)

        # Update connections for stationary luggage
        for sid in list(suitcase_coords.keys()):
            if sid in suitcase_coords:
                sc = suitcase_coords[sid]
                suitcase_current_info = suitcase_info.get(sid)
                luggage_is_moving = suitcase_current_info[2] if suitcase_current_info else False

                if not luggage_is_moving:
                    connected_people_for_this_luggage = connections.get(sid, set()).copy()
                    for pid in list(connected_people_for_this_luggage):
                        if pid in person_coords:
                            p_coord = person_coords[pid]
                            person_rad_for_disc, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                            if calculate_distance_squared(p_coord, sc) >= person_rad_for_disc ** 2:
                                connections[sid].discard(pid)
                                print(f"Luggage {sid}: Person {pid} DISCONNECTED (moved out of radius).")
                        else:
                            connections[sid].discard(pid)
                            print(f"Luggage {sid}: Person {pid} DISCONNECTED (person not detected).")

                    if sid not in potential_new_connections_timer:
                        potential_new_connections_timer[sid] = {}

                    for pid, p_coord in person_coords.items():
                        person_current_radius, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                        half_person_radius_sq = int(person_current_radius * half_radius_multiplier) ** 2

                        if pid not in connections.get(sid, set()) and calculate_distance_squared(p_coord, sc) < half_person_radius_sq:
                            if pid not in potential_new_connections_timer[sid]:
                                potential_new_connections_timer[sid][pid] = t_now
                            else:
                                time_in_half_radius = t_now - potential_new_connections_timer[sid][pid]
                                if time_in_half_radius > new_connection_time_threshold:
                                    connections.setdefault(sid, set()).add(pid)
                                    if not suitcase_init_people.get(sid):
                                        suitcase_init_people.setdefault(sid, set()).add(pid)
                                    print(f"Luggage {sid}: Person {pid} now connected after {new_connection_time_threshold}s in half_radius!")
                                    cv2.imwrite(f"screenshots/luggage_{sid}_new_connection_person_{pid}_{int(t_now)}.png", frame)
                                    potential_abandoned_timer.pop(sid, None)
                                    definite_abandoned_set.discard(sid)
                                    potential_new_connections_timer[sid].pop(pid, None)
                        else:
                            if pid in potential_new_connections_timer[sid]:
                                potential_new_connections_timer[sid].pop(pid, None)

                    # Clean up timers for people who disappeared or left zone
                    people_to_remove_from_timers = []
                    for pid_timed in list(potential_new_connections_timer.get(sid, {})):
                        person_coords_exists = pid_timed in person_coords
                        if person_coords_exists:
                            p_coord_timed = person_coords[pid_timed]
                            person_current_radius_timed, _ = get_depth_based_radius(p_coord_timed[0], p_coord_timed[1], depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                            half_person_radius_sq_timed = int(person_current_radius_timed * half_radius_multiplier) ** 2
                        else:
                            half_person_radius_sq_timed = 0

                        if not person_coords_exists or calculate_distance_squared(person_coords.get(pid_timed, (0,0)), sc) >= half_person_radius_sq_timed or pid_timed in connections.get(sid, set()):
                            people_to_remove_from_timers.append(pid_timed)
                    for pid_to_remove in people_to_remove_from_timers:
                        potential_new_connections_timer[sid].pop(pid_to_remove, None)

        # Draw lines
        if draw_connections_flag:
            for sid, pids_connected in connections.items():
                if sid in suitcase_coords:
                    luggage_center = suitcase_coords[sid]
                    for pid in pids_connected:
                        if pid in person_coords:
                            person_center = person_coords[pid]
                            cv2.line(frame, person_center, luggage_center, (255, 0, 255), 2)
                            dist = np.sqrt(calculate_distance_squared(person_center, luggage_center))
                            text_pos = ((person_center[0] + luggage_center[0]) // 2,
                                        (person_center[1] + luggage_center[1]) // 2 - 10)
                            cv2.putText(frame, f"{dist:.0f}px", text_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            if sid in potential_new_connections_timer and pid in potential_new_connections_timer[sid]:
                                time_left = new_connection_time_threshold - (t_now - potential_new_connections_timer[sid][pid])
                                if time_left > 0:
                                    cv2.putText(frame, f"New Connect in {int(time_left)}s", (person_center[0] - 50, person_center[1] - 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Abandonment check (depth-aware)
        for sid, (luggage_center_pos, ts_luggage_last_moved, luggage_is_moving) in list(suitcase_info.items()):
            if luggage_is_moving:
                if sid in potential_abandoned_timer:
                    potential_abandoned_timer.pop(sid, None)
                if sid in definite_abandoned_set:
                    definite_abandoned_set.discard(sid)
                continue

            current_people_connected_to_luggage = connections.get(sid, set())
            is_luggage_unattended = not current_people_connected_to_luggage

            definite_abandonment_radius_for_this_luggage = default_fallback_radius * definite_abandonment_radius_multiplier
            if use_depth_radius and depth_map is not None and sid in suitcase_coords:
                lc_x, lc_y = suitcase_coords[sid]
                luggage_base_radius, _ = get_depth_based_radius(lc_x, lc_y, depth_map, out_h, out_w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                definite_abandonment_radius_for_this_luggage = int(luggage_base_radius * definite_abandonment_radius_multiplier)
                definite_abandonment_radius_for_this_luggage = min(definite_abandonment_radius_for_this_luggage, visual_max_radius_limit)
                definite_abandonment_radius_for_this_luggage = max(1, definite_abandonment_radius_for_this_luggage)

            is_anyone_within_definite_radius = False
            if definite_abandonment_radius_for_this_luggage > 0 and sid in suitcase_coords:
                lc_x, lc_y = suitcase_coords[sid]
                for pid, p_coord in person_coords.items():
                    if calculate_distance_squared(p_coord, (lc_x, lc_y)) < definite_abandonment_radius_for_this_luggage ** 2:
                        is_anyone_within_definite_radius = True
                        break

            if is_luggage_unattended:
                time_stationary_unattended = t_now - ts_luggage_last_moved
                if time_stationary_unattended > abandonment_time:
                    if sid not in potential_abandoned_timer:
                        potential_abandoned_timer[sid] = t_now
                        cv2.imwrite(f"screenshots/luggage_{sid}_potential_abandoned_{int(t_now)}.png", frame)
                        print(f"Luggage {sid} is now potentially abandoned (stationary & unattended). Definite timer starts now.")

                    if not is_anyone_within_definite_radius:
                        time_alone_and_stationary_for_definite_check = t_now - potential_abandoned_timer[sid]
                        if time_alone_and_stationary_for_definite_check > definite_abandonment_time:
                            definite_abandoned_set.add(sid)
                            cv2.putText(frame, "ABANDONED!", (luggage_center_pos[0], luggage_center_pos[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                            print(f"Luggage {sid} is NOW DEFINITELY ABANDONED.")
                        else:
                            remaining_time_for_definite = definite_abandonment_time - time_alone_and_stationary_for_definite_check
                            cv2.putText(frame, f"Potential Aban. ({int(remaining_time_for_definite)}s)",
                                        (luggage_center_pos[0], luggage_center_pos[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    else:
                        if sid in definite_abandoned_set:
                            definite_abandoned_set.discard(sid)
                            print(f"Luggage {sid} reverted from DEFINITELY ABANDONED (someone walked by/nearby).")
                        potential_abandoned_timer[sid] = t_now
                        cv2.putText(frame, "Potentially Abandoned", (luggage_center_pos[0], luggage_center_pos[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    potential_abandoned_timer.pop(sid, None)
                    definite_abandoned_set.discard(sid)
            else:
                potential_abandoned_timer.pop(sid, None)
                definite_abandoned_set.discard(sid)

        # Draw persistent ABANDONED (for visualization)
        for sid in definite_abandoned_set:
            if sid in suitcase_coords:
                # find box in current frame for that id
                b = None
                if frame_boxes is not None:
                    b = next((box.xyxy[0].cpu().numpy().astype(int) for box in frame_boxes if box.id is not None and int(box.id[0]) == sid), None)
                if b is not None:
                    cv2.putText(frame, "ABANDONED!", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # --- EVALUATION ---
        if current_eval_mode:
            current_predicted_boxes = []
            if frame_boxes:
                for sid in definite_abandoned_set:
                    b = next((box.xyxy[0].cpu().numpy().astype(int) for box in frame_boxes if box.id is not None and int(box.id[0]) == sid), None)
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

    cap.release()
    out.release()
    cv2.destroyAllWindows()

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

    print(f"Done → {output_path}")
    print(f'metrics -> {RESULTS_CSV_PATH}')


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
