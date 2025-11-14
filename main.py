import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time
import os
from collections import deque


color_mapping = {
    0: (255, 0, 255),   # Red
    1: (0, 255, 0),  # Green
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

#model = YOLO('/mnt/SamsungSSD/Prtljaga/korzo_model.pt')
#model = YOLO('/mnt/SamsungSSD/Prtljaga/yolov8_models/trained_model/weights/best.pt')
#model = YOLO('./weights/yolov8s.pt')
#model = YOLO("./weights/CCTV-Korzo_Dusserdorf.pt")
model = YOLO("./weights/yolov8s.pt")

print(model.names)



video_path = "/C:/Users/admin/ABODA/cropped/video1.mp4"
#video_path = "C:/Users/admin/Downloads/abandoned_2.mp4"
#video_path = "./videos/Unattended.mp4"

cap = cv2.VideoCapture(video_path)
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
suitcase_coords = {}
suitcase_positions_history = {}  
suitcase_timestamps = {}
suitcase_initial_people = {}
connections = {}
abandoned_suitcases = set()
moving_suitcases = set()

movement_threshold = 10  # pixel
abandonment_time = 1  # Seconds
radius = 150  # radius
frame_number = 10

if not os.path.exists('screenshots'):
    os.makedirs('screenshots')
if not os.path.exists('started_moving'):
    os.makedirs('started_moving')

def print_info():
    print("Current suitcase coordinates:", suitcase_coords)
    print("Current person coordinates:", person_coords)
    print("Suitcase timestamps:", suitcase_timestamps)
    print("Connections:", connections)
    print("Abandoned suitcases:", abandoned_suitcases)
    print("Moving suitcases:", moving_suitcases)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("video over")
        break

    current_time = time.time()

    frame = cv2.resize(frame, (800, 600))

    results = model.track([frame], persist=True, classes=[0, 24, 26, 28])

    annotator = Annotator(frame)

    current_person_ids = set()
    current_suitcase_ids = set()

    for r in results:
        boxes = r.boxes
        for idx, box in enumerate(boxes):
            b = box.xyxy[0].cpu().numpy().astype(int)  
            c = int(box.cls)
            color = color_mapping.get(c, (255, 0, 0))  
            class_name = model.names[c]
            object_id = int(box.id[0]) if box.id else idx
            label_text = f"{class_name} ({object_id})"
            annotator.box_label(b, label_text, color=color)

            # update coordinate
            if c == 0:
                person_coords[object_id] = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                current_person_ids.add(object_id)
            elif c == 28 or c == 24 or c == 26:
                suitcase_coords[object_id] = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
                current_suitcase_ids.add(object_id)

                if object_id not in suitcase_positions_history:
                    suitcase_positions_history[object_id] = deque(maxlen=frame_number)
                suitcase_positions_history[object_id].append(suitcase_coords[object_id])

                # update timestamp

                if object_id in suitcase_timestamps:
                    prev_center, timestamp, was_moving = suitcase_timestamps[object_id]
                    if not has_movement(suitcase_positions_history[object_id], movement_threshold, frame_number):
                        suitcase_timestamps[object_id] = (prev_center, timestamp, was_moving)
                        print(f"Suitcase {object_id} not moving.")
                        cv2.putText(frame, "Not Moving", (int(b[0]), int(b[3]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        if not was_moving:
                            moving_suitcases.add(object_id)
                            screenshot_path = os.path.join('started_moving', f'screenshot_{object_id}.png')
                            cv2.imwrite(screenshot_path, frame)
                            print(f"Screenshot saved for suitcase {object_id} at {screenshot_path}. Suitcase started moving again.")
                            #re
                            nearby_people = {pid for pid, pcoord in person_coords.items() if calculate_distance_squared(pcoord, suitcase_coords[object_id]) < radius ** 2}
                            suitcase_initial_people[object_id] = nearby_people
                            connections[object_id] = nearby_people  
                            print(f"Suitcase {object_id} reinitialized with people: {nearby_people}")

                        suitcase_timestamps[object_id] = (suitcase_coords[object_id], current_time, True)
                        cv2.putText(frame, "Moving", (int(b[0]), int(b[3]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    suitcase_timestamps[object_id] = (suitcase_coords[object_id], current_time, True)
                    nearby_people = {pid for pid, pcoord in person_coords.items() if calculate_distance_squared(pcoord, suitcase_coords[object_id]) < radius ** 2}
                    suitcase_initial_people[object_id] = nearby_people
                    print(f"Suitcase {object_id} initially tracked people: {nearby_people}")
                    connections[object_id] = nearby_people  

    person_ids_to_remove = set(person_coords.keys()) - current_person_ids
    suitcase_ids_to_remove = set(suitcase_coords.keys()) - current_suitcase_ids

    for person_id in person_ids_to_remove:
        person_coords.pop(person_id, None)
        for suitcase_id, persons in connections.items():
            if person_id in persons:
                persons.remove(person_id)

    for suitcase_id in suitcase_ids_to_remove:
        suitcase_coords.pop(suitcase_id, None)
        suitcase_timestamps.pop(suitcase_id, None)
        suitcase_initial_people.pop(suitcase_id, None)
        connections.pop(suitcase_id, None)
        suitcase_positions_history.pop(suitcase_id, None)

    # remove links if out of radius
    for suitcase_id in list(connections.keys()):
        connections[suitcase_id] = {person_id for person_id in connections[suitcase_id] if person_id in person_coords and calculate_distance_squared(person_coords[person_id], suitcase_coords[suitcase_id]) < radius ** 2}
        if not connections[suitcase_id]:
            del connections[suitcase_id]

    # draw connections
    for suitcase_id, person_ids in connections.items():
        suitcase_coord = suitcase_coords.get(suitcase_id)
        for person_id in person_ids:
            person_coord = person_coords.get(person_id)
            if person_coord and suitcase_coord:
                person_coord_int = (int(person_coord[0]), int(person_coord[1]))
                suitcase_coord_int = (int(suitcase_coord[0]), int(suitcase_coord[1]))
                cv2.line(frame, person_coord_int, suitcase_coord_int, (255, 0, 255), 2)  
                distance_squared = calculate_distance_squared(person_coord, suitcase_coord)
                distance = np.sqrt(distance_squared)
                distance_text = f"{distance:.2f} units"
                cv2.putText(frame, distance_text, person_coord_int, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                print(f"Person {person_id} is {distance:.2f} units away from suitcase {suitcase_id}.")
    # abandoned
    for suitcase_id, (suitcase_center, timestamp, was_moving) in suitcase_timestamps.items():
        if current_time - timestamp > abandonment_time:
            tracked_people = suitcase_initial_people.get(suitcase_id, set())
            tracked_people = {pid for pid in tracked_people if pid in person_coords and calculate_distance_squared(person_coords[pid], suitcase_coords[suitcase_id]) < radius ** 2}
            suitcase_initial_people[suitcase_id] = tracked_people

            if not tracked_people:  
                b = next((box.xyxy[0].cpu().numpy().astype(int) for r in results for box in r.boxes if box.id and int(box.id[0]) == suitcase_id), None)
                if b is not None:
                    cv2.putText(frame, f"Suitcase {suitcase_id} left alone and potentially abandoned.", (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    print(f"Suitcase {suitcase_id} left alone and potentially abandoned.")

                    if suitcase_id not in abandoned_suitcases:
                        abandoned_suitcases.add(suitcase_id)
                        screenshot_path = os.path.join('screenshots', f'screenshot_{suitcase_id}.png')
                        cv2.imwrite(screenshot_path, frame)
                        print(f"Screenshot saved for suitcase {suitcase_id} at {screenshot_path}.")

    annotated_frame = annotator.result()
    cv2.imshow("YOLOv8", annotated_frame)   # <-- live preview
    out.write(annotated_frame)                     # <-- save frame

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print_info()

else: 
    print("finish")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved successfully as '{output_path}'")
