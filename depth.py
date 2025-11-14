import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time
import os
from collections import deque

# ---------------------------
#       CONFIG & HYPERPARAMETERS
# ---------------------------

# Toggle features
draw_radius = True
draw_connections_flag = True
use_depth_radius = True # Scale radius by depth map

# Base radius for proximity checks and drawing circles (used if depth is disabled or invalid)
# THIS WILL BE OVERRIDDEN IF use_depth_radius IS TRUE FOR PERSON RADII
default_fallback_radius = 250
half_radius_multiplier = 0.5 # For the 'new connection' logic (e.g., 0.5 means half the calculated radius)


# Depth Map Settings
depth_map_path = "./weights/dusslerdorf_aiport_2024_26_02_12_20pm_first_frame_depth.npy"
depth_map_scale_factor = 2.0 # Scaling factor for the loaded depth map values
depth_coefficient = 1000.0 # A larger value will result in larger radii overall. Tune this!
power_factor = 0.6 # A value between 0.5 and 1.0 (e.g., 0.7, 0.8, 0.9)
                         # Smaller power_factor makes the radius grow faster for closer objects.
visual_max_radius_limit = 1000 # Define a reasonable max radius for visualization purposes

# Paths
model_path = "./weights/yolov8s.pt"
#model_path = "./weights/CCTV-Korzo_Dusserdorf.pt"
#video_path = "/mnt/SamsungSSD/Prtljaga/Datasets/dataset_IITP20/IITP20_Datset_02/Abandoned_cases/medium_cases/vid_00027_cutted.mp4"
#video_path = "/mnt/SamsungSSD/Prtljaga/Datasets/dataset_IITP20/IITP20_Datset_01/Abandoned_cases/hard_cases/vid_00069.mp4" 
video_path = "/C:/Users/admin/ABODA/cropped/video1.mp4"
#video_path = "C:/Users/admin/Downloads/abandoned_3.mp4"
#video_path = "C:/Users/admin/Downloads/abandoned_2.mp4"


# Movement & abandonment settings (from second script, modified for quick testing as in third)
movement_threshold = 15
abandonment_time = 2 # seconds stationary before flagged as 'potentially abandoned'
definite_abandonment_time = 1 # seconds alone and stationary for definite abandonment
definite_abandonment_radius_multiplier = 1.5
frame_history = 30 # From second script (larger history for smoother motion detection)

# Connection logic
new_connection_time_threshold = 1

# Color mapping: {0: 'luggage', 1: 'person'}
color_mapping = {
    0: (255, 0, 255),   # Purple
    1: (0, 255, 0),  # Green
}

# ---------------------------
#       HELPER FUNCTIONS
# ---------------------------

def calculate_distance_squared(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def has_movement(pos_deque, thr_sq, history_len):
    if len(pos_deque) < history_len:
        return True # Consider it moving if not enough history
    start = pos_deque[0]
    for p in list(pos_deque)[1:history_len]:
        if calculate_distance_squared(p, start) >= thr_sq:
            return True
    return False

def get_depth_based_radius(cx, cy, depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius_val):
    current_radius = default_fallback_radius_val
    circle_color = (0, 255, 255) # Default yellow-cyan

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
                    circle_color = (0, 0, 255) # Red for max radius
                else:
                    current_radius = r
            except (OverflowError, ZeroDivisionError):
                current_radius = visual_max_radius_limit
                circle_color = (0, 0, 255) # Red for error/max
        else:
            current_radius = default_fallback_radius_val
            circle_color = (128, 128, 128) # Grey for invalid depth
    else:
        current_radius = default_fallback_radius_val
        circle_color = (128, 128, 128) # Grey for out of bounds/no depth map

    return max(1, current_radius), circle_color # Ensure radius is at least 1 for drawing

# ---------------------------
#       INITIALIZATION
# ---------------------------

# Load depth map
depth_map = None
if use_depth_radius:
    try:
        # Load the depth map and resize it to match the video frame resolution (800x600)
        loaded_depth_map = np.load(depth_map_path)
        # Assuming the video frames are resized to (800, 600) in the loop,
        # ensure the depth map matches this resolution for accurate lookup.
        # Use INTER_LINEAR or INTER_AREA for continuous data, INTER_NEAREST if it's categorical.
        depth_map = cv2.resize(loaded_depth_map, (800, 600), interpolation=cv2.INTER_LINEAR)
        depth_map = depth_map * depth_map_scale_factor
        print(f"Depth map loaded, resized to (800, 600), and scaled by {depth_map_scale_factor}x.")
        print("\n--- Depth Map Properties ---")
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth map data type: {depth_map.dtype}")
        print(f"Depth map min value (after scaling): {np.min(depth_map)}")
        print(f"Depth map max value (after scaling): {np.max(depth_map)}")
        print("----------------------------\n")
    except FileNotFoundError:
        print(f"Error: Depth map not found at {depth_map_path}. Disabling depth radius.")
        use_depth_radius = False
    except Exception as e:
        print(f"Error loading or resizing depth map: {e}. Disabling depth radius.")
        use_depth_radius = False

model = YOLO(model_path)
print("Model Class Names:", model.names)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0
out_w, out_h = 800, 600
output_path = f'{video_path.split(".")[0]}_annotated.mp4'
print(f"output_path: {output_path}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
print(f"Recording → {output_path} (press 'q' to stop early)")

person_coords = {}
person_history = {}
person_info = {} # Stores ((cx,cy), timestamp_last_moved, is_moving)

suitcase_coords = {}
suitcase_history = {}
suitcase_info = {} # Stores ((cx,cy), timestamp_last_moved, is_moving)
suitcase_init_people = {} # Stores initial people associated when suitcase first appears
connections = {} # {suitcase_id: {person_id, ...}} Current connections based on proximity

potential_abandoned_timer = {} # {suitcase_id: timestamp_started_potential_abandonment}
definite_abandoned_set = set() # Set of suitcase IDs that are definitively abandoned

potential_new_connections_timer = {} # {suitcase_id: {person_id: timestamp_entered_half_radius}}

os.makedirs("screenshots", exist_ok=True)
os.makedirs("started_moving", exist_ok=True)

movement_threshold_sq = movement_threshold**2

# ---------------------------
#       MAIN LOOP
# ---------------------------
print("\nStarting video processing...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    t_now = time.time()

    frame = cv2.resize(frame, (800, 600))
    h, w = frame.shape[:2]

    results = model.track([frame], persist=True, classes=[0, 24, 26, 28])
    annotator = Annotator(frame)

    current_persons = set()
    current_suitcases = set()

    for res in results:
        if res.boxes and res.boxes.id is not None:
            for box in res.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls)
                obj_id = int(box.id[0])

                cx, cy = (x1+x2)//2, (y1+y2)//2

                name = model.names.get(cls, f"Class {cls}")
                color = color_mapping.get(cls, (255,0,0))
                label = f"{name} ({obj_id})"
                annotator.box_label((x1,y1,x2,y2), label, color=color)

                # --- LUGGAGE LOGIC ---
                if cls == 28 or cls == 24 or cls == 26: # luggage
                    suitcase_coords[obj_id] = (cx,cy)
                    current_suitcases.add(obj_id)

                    if obj_id not in suitcase_history:
                        suitcase_history[obj_id] = deque(maxlen=frame_history)
                        suitcase_info[obj_id] = ((cx,cy), t_now, True)

                        initial_associated_people = set()
                        # When a suitcase first appears, connect it to any nearby person
                        # using their current calculated radius.
                        for pid, p_coord in person_coords.items():
                            # Get the person's current depth-based radius for connection check
                            person_rad, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                            if calculate_distance_squared(p_coord, (cx,cy)) < person_rad**2:
                                initial_associated_people.add(pid)
                        suitcase_init_people[obj_id] = initial_associated_people
                        connections[obj_id] = initial_associated_people.copy()
                        print(f"Luggage {obj_id} first seen. Initial owner(s): {suitcase_init_people[obj_id]}")

                    else: # Luggage is already being tracked
                        prev_c, ts_luggage_last_moved, was_moving_prev_frame = suitcase_info[obj_id]
                        suitcase_history[obj_id].append((cx,cy))
                        luggage_is_moving = has_movement(suitcase_history[obj_id], movement_threshold_sq, frame_history)

                        if luggage_is_moving:
                            if not was_moving_prev_frame:
                                cv2.imwrite(f"started_moving/luggage_{obj_id}_started_moving_{int(t_now)}.png", frame)
                                print(f"Luggage {obj_id} STARTED MOVING at {time.ctime(t_now)}. RESETTING ABANDONMENT TIMERS/FLAGS.")

                                if obj_id in potential_abandoned_timer:
                                    potential_abandoned_timer.pop(obj_id)
                                if obj_id in definite_abandoned_set:
                                    definite_abandoned_set.discard(obj_id)

                                potential_new_connections_timer.pop(obj_id, None) # Also reset new connection timers for this luggage

                                # When luggage starts moving, its "owner" becomes whoever is currently closest
                                current_owners_on_move = set()
                                for pid, p_coord in person_coords.items():
                                    person_rad, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                                    if calculate_distance_squared(p_coord, (cx,cy)) < person_rad**2:
                                        current_owners_on_move.add(pid)
                                suitcase_init_people[obj_id] = current_owners_on_move
                                connections[obj_id] = current_owners_on_move.copy()
                                if current_owners_on_move:
                                    print(f"Luggage {obj_id} re-assigned owner(s) on move: {current_owners_on_move}")
                                else:
                                    print(f"Luggage {obj_id} moving with no clear owner nearby.")

                            suitcase_info[obj_id] = ((cx,cy), t_now, True)
                            cv2.putText(frame, "Moving", (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                        else: # Luggage is stationary
                            suitcase_info[obj_id] = ((cx,cy), ts_luggage_last_moved, False)
                            cv2.putText(frame, "Not Moving", (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                # --- PERSON LOGIC ---
                elif cls == 0: # person
                    person_coords[obj_id] = (cx,cy)
                    current_persons.add(obj_id)

                    if obj_id not in person_history:
                        person_history[obj_id] = deque(maxlen=frame_history)
                        person_info[obj_id] = ((cx,cy), t_now, True)
                    person_history[obj_id].append((cx,cy))

                    prev_p_c, prev_p_ts, prev_p_mov = person_info[obj_id]
                    person_is_moving = has_movement(person_history[obj_id], movement_threshold_sq, frame_history)
                    person_info[obj_id] = ((cx,cy), t_now if person_is_moving else prev_p_ts, person_is_moving)

                    if draw_radius:
                        # Get depth-based radius for this person
                        current_person_radius, circle_color_person = get_depth_based_radius(
                            cx, cy, depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius
                        )
                        cv2.circle(frame, (cx,cy), current_person_radius, circle_color_person, 2)
                        cv2.putText(frame, f"R:{current_person_radius}", (cx + 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        # Draw the half-radius for new connection logic (still based on depth radius)
                        half_person_radius = int(current_person_radius * half_radius_multiplier)
                        cv2.circle(frame, (cx,cy), half_person_radius, (0, 128, 255), 1)
                        cv2.putText(frame, f"R/2:{half_person_radius}", (cx + 20, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 1)


    # --- Cleanup missing objects ---
    for pid in list(person_coords.keys()):
        if pid not in current_persons:
            person_coords.pop(pid)
            person_history.pop(pid, None)
            person_info.pop(pid, None)
            # Remove this person from any suitcase connections and new connection timers
            for s_id in connections:
                connections[s_id].discard(pid)
            for s_id in potential_new_connections_timer:
                potential_new_connections_timer[s_id].pop(pid, None)


    for sid in list(suitcase_coords.keys()):
        if sid not in current_suitcases:
            suitcase_coords.pop(sid)
            suitcase_history.pop(sid, None)
            suitcase_info.pop(sid, None)
            suitcase_init_people.pop(sid, None)
            connections.pop(sid, None)
            # Clean up abandonment and new connection timers for missing suitcase
            if sid in potential_abandoned_timer:
                potential_abandoned_timer.pop(sid)
            if sid in definite_abandoned_set:
                definite_abandoned_set.discard(sid)
            potential_new_connections_timer.pop(sid, None)

    # --- Update current connections (ONLY DISCONNECT FOR STATIONARY LUGGAGE) ---
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
                        # Disconnect if person moves out of their current depth-based radius
                        person_rad_for_disc, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                        if calculate_distance_squared(p_coord, sc) >= person_rad_for_disc**2:
                            connections[sid].discard(pid)
                            print(f"Luggage {sid}: Person {pid} DISCONNECTED (moved out of radius).")
                    else:
                        # Person no longer detected
                        connections[sid].discard(pid)
                        print(f"Luggage {sid}: Person {pid} DISCONNECTED (person not detected).")


                # NEW LOGIC: Check for new connections based on half_radius time
                if sid not in potential_new_connections_timer:
                    potential_new_connections_timer[sid] = {}

                for pid, p_coord in person_coords.items():
                    person_current_radius, _ = get_depth_based_radius(p_coord[0], p_coord[1], depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                    half_person_radius_sq = int(person_current_radius * half_radius_multiplier)**2

                    if pid not in connections.get(sid, set()) and \
                       calculate_distance_squared(p_coord, sc) < half_person_radius_sq:

                        if pid not in potential_new_connections_timer[sid]:
                            potential_new_connections_timer[sid][pid] = t_now
                            # print(f"Luggage {sid}: Person {pid} entered half_radius. Timer started.")
                        else:
                            time_in_half_radius = t_now - potential_new_connections_timer[sid][pid]
                            if time_in_half_radius > new_connection_time_threshold:
                                connections.setdefault(sid, set()).add(pid)
                                # If the suitcase didn't have initial people or they all left,
                                # and a new person connects, consider them a new initial owner.
                                if not suitcase_init_people.get(sid):
                                    suitcase_init_people.setdefault(sid, set()).add(pid)
                                print(f"Luggage {sid}: Person {pid} now connected after {new_connection_time_threshold}s in half_radius!")
                                cv2.imwrite(f"screenshots/luggage_{sid}_new_connection_person_{pid}_{int(t_now)}.png", frame)

                                # If a new connection is made, reset abandonment timers for this luggage
                                if sid in potential_abandoned_timer:
                                    potential_abandoned_timer.pop(sid)
                                if sid in definite_abandoned_set:
                                    definite_abandoned_set.discard(sid)

                                potential_new_connections_timer[sid].pop(pid) # Remove timer as connection is established
                    else:
                        # If person is outside half_radius or already connected, remove their timer
                        if pid in potential_new_connections_timer[sid]:
                            # print(f"Luggage {sid}: Person {pid} left half_radius or got connected. Timer reset.")
                            potential_new_connections_timer[sid].pop(pid)

                # Clean up timers for people who disappeared or are no longer valid for connection
                people_to_remove_from_timers = []
                for pid_timed in potential_new_connections_timer.get(sid, {}):
                    person_coords_exists = pid_timed in person_coords
                    if person_coords_exists:
                        p_coord_timed = person_coords[pid_timed]
                        person_current_radius_timed, _ = get_depth_based_radius(p_coord_timed[0], p_coord_timed[1], depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
                        half_person_radius_sq_timed = int(person_current_radius_timed * half_radius_multiplier)**2
                    else:
                        half_person_radius_sq_timed = 0 # Forces removal if person is gone

                    if not person_coords_exists or \
                       calculate_distance_squared(person_coords[pid_timed], sc) >= half_person_radius_sq_timed or \
                       pid_timed in connections.get(sid, set()): # If they became connected by other means or left zone
                        people_to_remove_from_timers.append(pid_timed)
                for pid_to_remove in people_to_remove_from_timers:
                    potential_new_connections_timer[sid].pop(pid_to_remove)


    # --- Draw lines between connected people and luggage ---
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

                        # Show new connection timer countdown if applicable
                        if sid in potential_new_connections_timer and pid in potential_new_connections_timer[sid]:
                            time_left = new_connection_time_threshold - (t_now - potential_new_connections_timer[sid][pid])
                            if time_left > 0:
                                cv2.putText(frame, f"New Connect in {int(time_left)}s", (person_center[0] - 50, person_center[1] - 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- Abandonment Check Logic ---
    for sid, (luggage_center_pos, ts_luggage_last_moved, luggage_is_moving) in list(suitcase_info.items()):
        if luggage_is_moving:
            # If luggage is moving, it cannot be abandoned. Reset timers.
            if sid in potential_abandoned_timer:
                potential_abandoned_timer.pop(sid)
            if sid in definite_abandoned_set:
                definite_abandoned_set.discard(sid)
            continue # Skip to next suitcase

        current_people_connected_to_luggage = connections.get(sid, set())
        is_luggage_unattended = not current_people_connected_to_luggage

        # Calculate the definite abandonment radius for this luggage based on its depth
        definite_abandonment_radius_for_this_luggage = default_fallback_radius * definite_abandonment_radius_multiplier
        if use_depth_radius and depth_map is not None and sid in suitcase_coords:
            lc_x, lc_y = suitcase_coords[sid]
            # Get depth-based radius for the luggage itself
            luggage_base_radius, _ = get_depth_based_radius(lc_x, lc_y, depth_map, h, w, depth_coefficient, power_factor, visual_max_radius_limit, default_fallback_radius)
            definite_abandonment_radius_for_this_luggage = int(luggage_base_radius * definite_abandonment_radius_multiplier)
            definite_abandonment_radius_for_this_luggage = min(definite_abandonment_radius_for_this_luggage, visual_max_radius_limit)
            definite_abandonment_radius_for_this_luggage = max(1, definite_abandonment_radius_for_this_luggage) # Ensure at least 1

        # Check if ANY person is within the calculated definite abandonment radius (even if not 'connected')
        is_anyone_within_definite_radius = False
        if definite_abandonment_radius_for_this_luggage > 0 and sid in suitcase_coords:
            lc_x, lc_y = suitcase_coords[sid]
            for pid, p_coord in person_coords.items():
                if calculate_distance_squared(p_coord, (lc_x, lc_y)) < definite_abandonment_radius_for_this_luggage**2:
                    is_anyone_within_definite_radius = True
                    break

        if is_luggage_unattended:
            time_stationary_unattended = t_now - ts_luggage_last_moved

            if time_stationary_unattended > abandonment_time:
                # Potential abandonment state
                if sid not in potential_abandoned_timer:
                    potential_abandoned_timer[sid] = t_now
                    cv2.imwrite(f"screenshots/luggage_{sid}_potential_abandoned_{int(t_now)}.png", frame)
                    print(f"Luggage {sid} is now potentially abandoned (stationary & unattended). Definite timer starts now.")

                # Definite abandonment check
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
                    # Luggage is unattended and stationary, but someone is within the definite abandonment radius.
                    # This means it's 'Potentially Abandoned' but cannot become 'Definitely Abandoned'.
                    # If it was previously definite, revert it.
                    if sid in definite_abandoned_set:
                        definite_abandoned_set.discard(sid)
                        print(f"Luggage {sid} reverted from DEFINITELY ABANDONED (someone walked by/nearby).")
                    # Reset the potential timer to require re-triggering the definite abandonment check
                    # (i.e., someone needs to leave the large radius for a consistent time).
                    potential_abandoned_timer[sid] = t_now
                    cv2.putText(frame, "Potentially Abandoned", (luggage_center_pos[0], luggage_center_pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                # Not yet in potential abandonment phase (time_stationary_unattended <= abandonment_time)
                # Ensure timers are reset if they somehow started prematurely.
                if sid in potential_abandoned_timer:
                    potential_abandoned_timer.pop(sid)
                if sid in definite_abandoned_set:
                    definite_abandoned_set.discard(sid)
        else:
            # Luggage is not unattended (someone is 'connected'). Reset abandonment timers.
            if sid in potential_abandoned_timer:
                potential_abandoned_timer.pop(sid)
            if sid in definite_abandoned_set:
                definite_abandoned_set.discard(sid)

    annotated_frame = annotator.result()
    cv2.imshow("YOLOv8", annotated_frame)   # <-- live preview
    out.write(annotated_frame)   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("\nVideo processing finished. Resources released.")
print(f"Video saved successfully as '{output_path}'")