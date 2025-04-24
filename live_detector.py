# pls install ultralytics and dependencies: pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO
import logging
import os
import numpy as np
import time

# --- config ---
CAMERA_INDEX = 0                   # camera index (0 for default webcam)
MODEL_NAME = "yolov8n.pt"           # YOLOv8 model variant (try 'yolov8s.pt' if the code fails) 
"""
options are: n, s, m, 1, x. increasing order of size/accuracy BUT slower!! 
if you want to test different model variants change the letter after 'yolov8' to the letter you want.
the default is n, it's the fastest varient BUT the least accurate. 
the tested variants are n and s.
if you face any issues please contact me!
"""
CONFIDENCE_THRESHOLD = 0.5         # detection confidence threshold (the less the more likely it will make mistakes)
CLASSES_TO_DETECT = [0]            # classes to detect (0 for person)
WARNING_THRESHOLD = 10             # number of people to trigger a warning
FRAME_SKIP = 1                     # process every Nth frame (1 = process all)
DISPLAY_WINDOW_NAME = "Live Crowd Detection - Error Handling"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Functions ---

def load_model(model_name: str) -> YOLO | None:
    """Loads the YOLOv8 model."""
    try:
        if not os.path.exists(model_name):
             logging.warning(f"⚠️ Model file '{model_name}' not found. Attempting to download...")
        model = YOLO(model_name)
        logging.info(f"✅ Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load model '{model_name}': {e}")
        return None

# --- main test loop ---
def process_live_feed_with_error_handling(
    camera_index: int,
    model: YOLO,
    conf_threshold: float = 0.5,
    classes: list[int] | None = None,
    frame_skip: int = 1,
    warning_threshold: int = 10
):
    """
    here it processes the camera live feed
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error(f"❌ Failed to open camera with index: {camera_index}")
        return

    logging.info("✅ Camera opened successfully. Starting live feed processing...")
    logging.info("ℹ️ Press 'q' to quit.")

    frame_num = 0
    processed_frame_count = 0
    results = None # initialize results outside the loop

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ Failed to grab frame from camera. Exiting.")
                break

            annotated_frame = frame.copy() # start with a fresh copy each time

            # process only every Nth frame if frame_skip > 1
            if frame_num % frame_skip == 0:
                processed_frame_count += 1
                try:
                    """
                    prediction fails sometimes esp when the target is moving so I placed it 
                    inside a try block to skip frames where it fails to predict instead of crashing 
                    """
                    results = model.predict(source=frame, conf=conf_threshold, classes=classes, verbose=False)
                    logging.info(f"--- Frame {frame_num}: Prediction successful.")

                    # --- manual drawing block (only if prediction succeeds) ---
                    count = 0
                    person_class_id = 0
                    color = (0, 255, 0) # green -> good! no warnings
                    warn_color = (0, 0, 255) # red -> crowded! warning
                    thickness = 2

                    if len(results) > 0 and results[0].boxes is not None:
                        if len(results[0].boxes) > 0:
                           try:
                               boxes = results[0].boxes.xyxy.cpu().numpy()
                               detected_classes = results[0].boxes.cls.cpu().numpy()
                               confidences = results[0].boxes.conf.cpu().numpy()

                               for i in range(len(boxes)):
                                   x1, y1, x2, y2 = map(int, boxes[i])
                                   current_class_id = int(detected_classes[i])
                                   current_conf = confidences[i]
                                   label = f"{model.names[current_class_id]} {current_conf:.2f}"
                                   cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                                   cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                                   if current_class_id == person_class_id:
                                       count += 1
                           except Exception as e_draw:
                               # this error shouldn't happen if predict succeeded, but good practice :)
                               logging.error(f"❌ Error accessing detection results for drawing on frame {frame_num}: {e_draw}")
                        else:
                            # no boxes detected -> count remains 0
                             logging.info(f"--- Frame {frame_num}: No objects detected.")
                             pass # explicitly do nothing if no boxes


                    # add overall count and warning text (uses count from successful predictions)
                    text_color = warn_color if count > warning_threshold else color
                    cv2.putText(annotated_frame, f"People Count: {count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    if count > warning_threshold:
                        cv2.putText(annotated_frame, "WARNING: High Crowd Count!", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, warn_color, 2, cv2.LINE_AA)
                    # --- end manual drawing block ---


                except Exception as predict_e:
                    # --- this block executes if model.predict() fails -> if the target is moving a lot ---
                    logging.warning(f"⚠️ Frame {frame_num}: Skipping frame due to error during prediction: {predict_e}")
                    cv2.putText(annotated_frame, "Processing Error", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    # continue # optional: uncomment if you don't want to show frames with errors at all

            # display the frame (either successfully annotated or with error message)
            cv2.imshow(DISPLAY_WINDOW_NAME, annotated_frame)

            frame_num += 1

            # heck for quit key ('q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("ℹ️ 'q' pressed, quitting.")
                break

    except Exception as e:
        # log other potential errors outside the main loop/predict try-except
        logging.error(f"❌ An error occurred outside the main processing loop: {e}")
    finally:
        # release the resources
        if cap.isOpened():
            cap.release()
            logging.info("✅ Camera released.")
        cv2.destroyAllWindows()
        logging.info("✅ Display windows closed.")


# --- main ---
if __name__ == "__main__":
    model = load_model(MODEL_NAME)
    if model:
        try:
            process_live_feed_with_error_handling(
                camera_index=CAMERA_INDEX,
                model=model,
                conf_threshold=CONFIDENCE_THRESHOLD,
                classes=CLASSES_TO_DETECT,
                frame_skip=FRAME_SKIP,
                warning_threshold=WARNING_THRESHOLD
            )
        except Exception as main_exception:
             logging.error(f"❌ An error occurred in the main execution block: {main_exception}")
             cv2.destroyAllWindows()
    else:
        logging.error("❌ Model could not be loaded. Exiting.")