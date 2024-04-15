# ANIMAL-INTRUSION-DETECTION-SYSTEM-WITH-YOLO-V8

README.md:

# Wildlife Animal Detection using YOLOv8

This repository contains scripts for real-time wildlife animal detection using YOLOv8, a state-of-the-art object detection algorithm. By leveraging computer vision techniques, this project aims to contribute to wildlife conservation efforts by enabling the detection and monitoring of various animal species.

## Requirements
- Python 3.x
- OpenCV
- Ultralytics YOLO package
  
## Dataset
- https://universe.roboflow.com/machine-train-ur3hn/animals-detection-bsbbi.

## Installation
1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Run the Python script `animal.py` to start real-time detection from your webcam.
    ```
    python animal.py
    ```
2. Press the 'Esc' key to exit the detection loop.

## Customization
- You can modify the list of `classnames` in the script to include or exclude specific animal classes according to your needs.
- Replace the provided `best.pt` model with your own trained YOLOv8 model for customized detection.

## Acknowledgments
- The YOLOv8 model implementation is based on Ultralytics YOLO package.
- Sample code adapted from [Ultralytics YOLO documentation](https://github.com/ultralytics/ultralytics).

- ## Code Explanation

This Python script enables real-time wildlife animal detection using YOLOv8, an advanced object detection algorithm. Below is a breakdown of its functionality:

1. **Imports**: The script imports necessary libraries:
   - `ultralytics.YOLO`: Interface for YOLOv8 models.
   - `cv2`: OpenCV library for computer vision tasks.
   - `math`: Python math library for mathematical operations.

2. **Webcam Initialization**: 
   - `cv2.VideoCapture(0)`: Opens the default webcam (index 0) for real-time video capture.

3. **Model Loading**:
   - `YOLO('best.pt')`: Loads a pre-trained YOLOv8 model from the file 'best.pt'.

4. **Class Names**: 
   - `classnames`: A list containing names of wildlife animal classes that the model can detect.

5. **Frame Processing Loop**:
   - `while True:`: Loops indefinitely until the 'Esc' key is pressed.

6. **Inference with YOLOv5**:
   - `result = model(frame, stream=True)`: Detects objects in the current frame using the YOLOv8 model with real-time streaming.

7. **Bounding Boxes and Display**:
   - Draws bounding boxes around detected objects and displays class labels with confidence percentages.

8. **Frame Display**:
   - `cv2.imshow('Animal Detection', frame)`: Displays the processed frame with bounding boxes and labels.

9. **Loop Termination and Resource Release**:
   - Breaks the loop when the 'Esc' key is pressed.
   - Releases the webcam and closes the OpenCV window.

This script offers a straightforward approach to detect wildlife animals in real-time, aiding in various conservation and monitoring efforts.
10.**Animal detection**:

-Alert is sent through siren when Animal is detected.
-Twilio API is integrated to send the messages to the rangers.


