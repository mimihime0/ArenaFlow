# ArenaFlow

## Description

This project is a Python prototype designed for real-time person detection and counting using a live camera feed (a webcam was used for testing). It uses the YOLOv8 object detection model via the `ultralytics` library and uses OpenCV for video capture and display.

The script counts the number of people detected in each frame and displays a visual warning if the count exceeds a configurable threshold (mentioned later). 

Developed for the SCAI Hackathon, this script is a core component of our "ArenaFlow" concept, aiming to revolutionize crowd safety. This real-time detection prototype showcases the potential of AI in creating smarter, safer public spaces.

## Features 

* Real-time person detection from a connected camera.
* Live counting of detected people per frame.
* Visual display of the camera feed with bounding boxes around detected people.
* Configurable warning threshold for high crowd counts (displays text on screen).
* Choice of YOLOv8 model variant (defaults to `yolov8n` for speed).
* Adjustable confidence threshold for detections.
* Option to process every Nth frame (`FRAME_SKIP`) for performance tuning.
* Error handling around model prediction to prevent crashes on difficult frames.

## Requirements

* Python 3.9+
* OpenCV (`opencv-python`)
* Ultralytics YOLOv8 (`ultralytics`)
* NumPy (`numpy`)

You can install the required Python packages using pip:

```
pip install opencv-python ultralytics numpy
```
(Note: OpenCV might require additional system libraries on some OS versions. The Ultralytics library will automatically download the specified YOLOv8 model file if needed 👍🏻.)

## Configuration

You can adjust the script's behavior by modifying the constants at the top of the `live_detector.py` file:

* `CAMERA_INDEX = 0`
    * Specifies which camera connected to your system to use. `0` is usually the default built-in webcam. Try `1`, `2`, etc., if you have multiple cameras.

* `MODEL_NAME = "yolov8n.pt"`
    * Determines which YOLOv8 model variant to use for detection.
    * **Options are:** `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` (representing Nano, Small, Medium, Large, Extra-Large).
    * Models increase in size and accuracy (n -> s -> m -> l -> x) **BUT** they also become significantly slower!!
    * If you want to test different model variants, change the letter after 'yolov8' (e.g., `"yolov8s.pt"`). The library will download the model automatically if needed.
    * **Default is 'n':** This was chosen because it's the fastest variant, best suited for real-time processing on standard hardware (like for this prototype), but it is the least accurate.
    * **Tested variants:** 'n', 's' and 'm' were primarily tested during development.

* `CONFIDENCE_THRESHOLD = 0.5`
    * The minimum confidence score (from 0.0 to 1.0) the model needs to have for a detection to be counted.
    * A value of `0.5` means the model must be at least 50% sure.
    * **Lowering this value** (e.g., to `0.3` or `0.4`) might help detect more people (especially those far away or partially hidden) **BUT** increases the chance of false detections. Adjust based on your needs.

* `CLASSES_TO_DETECT = [0]`
    * Filters the detections to specific object classes. YOLOv8 trained on the COCO dataset uses `0` for the 'person' class.
    * Leave this at `[0]` to *only* count people!!!

* `WARNING_THRESHOLD = 10`
    * The number of detected people above which the "WARNING: High Crowd Count!" message will appear on the screen in red text. Adjust this value based on the capacity or safety limit you want to simulate.

* `FRAME_SKIP = 1`
    * Setting this to `1` processes every single frame from the camera (most resource-intensive and the default).
    * Setting it to `2` processes every other frame, `3` processes every third frame, and so on.
    * **Increasing this value** can significantly improve performance (reduce lag) if your device struggles to process every frame in real-time, **BUT** the count and display will update less frequently.

* `DISPLAY_WINDOW_NAME = "Live Crowd Detection - Error Handling"`
    * Sets the title for the output window displayed by OpenCV.

*If you face any issues please contact me!*

# Usage
1. Ensure you have the required libraries installed (see Requirements).

2. Make sure your camera is connected and accessible.

3. Run the script from your terminal:
```
py live_detector.py
```
4. An OpenCV window will open displaying the camera feed with detections.

5. Press the 'q' key while the display window is active to quit the script.

# NOTES
The script includes a ```try...except``` block around the ```model.predict()``` call within the main processing loop. This was added because testing revealed that the prediction function could sometimes fail on specific frames from the live feed (potentially due to motion blur or complex scenes), causing the program to crash. This means that during moments of heavy movement or complex scenes that trigger the error, the people count displayed might temporarily freeze or be inaccurate because those frames are being skipped. This is a trade-off for stability in the prototype 😥. 
