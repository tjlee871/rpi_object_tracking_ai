import argparse
import sys
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

import RPi.GPIO as GPIO

no_detect_counter = 0
x_delta = 0
stepper_seq = 0
IN1 = 12
IN2 = 16
IN3 = 20
IN4 = 21
LED1 = 19
LED2 = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(LED1, GPIO.OUT)
GPIO.setup(LED2, GPIO.OUT)

last_detections = []

def track_x_delta():
    global stepper_seq
    if x_delta > 5:
        step_forward(stepper_seq)
        stepper_seq += 1
        GPIO.output(LED1, 1)
        GPIO.output(LED2, 1)
    elif x_delta < -5:
        step_backward(stepper_seq)
        stepper_seq += 1
        GPIO.output(LED1, 1)
        GPIO.output(LED2, 1)
    else:
        GPIO.output(LED1, 0)
        GPIO.output(LED2, 0)
    if stepper_seq >= 4:
        stepper_seq = 0

def step_forward(seq):
    if seq == 0:
        GPIO.output(IN1, 1)
        GPIO.output(IN2, 0)
        GPIO.output(IN3, 0)
        GPIO.output(IN4, 1)
    elif seq == 1:
        GPIO.output(IN1, 1)
        GPIO.output(IN2, 1)
        GPIO.output(IN3, 0)
        GPIO.output(IN4, 0)
    elif seq == 2:
        GPIO.output(IN1, 0)
        GPIO.output(IN2, 1)
        GPIO.output(IN3, 1)
        GPIO.output(IN4, 0)
    elif seq == 3:
        GPIO.output(IN1, 0)
        GPIO.output(IN2, 0)
        GPIO.output(IN3, 1)
        GPIO.output(IN4, 1)

def step_backward(seq):
    if seq == 0:
        GPIO.output(IN1, 0)
        GPIO.output(IN2, 0)
        GPIO.output(IN3, 1)
        GPIO.output(IN4, 1)
    elif seq == 1:
        GPIO.output(IN1, 0)
        GPIO.output(IN2, 1)
        GPIO.output(IN3, 1)
        GPIO.output(IN4, 0)
    elif seq == 2:
        GPIO.output(IN1, 1)
        GPIO.output(IN2, 1)
        GPIO.output(IN3, 0)
        GPIO.output(IN4, 0)
    elif seq == 3:
        GPIO.output(IN1, 1)
        GPIO.output(IN2, 0)
        GPIO.output(IN3, 0)
        GPIO.output(IN4, 1)

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        last_detections = None
        return last_detections

    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    if bbox_normalization:
        boxes = boxes / input_h

    if bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]
    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if (score > threshold) and (category < 26)  # Filter out irrelevant category
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    global x_delta
    global no_detect_counter
    
    if detections is None:
        no_detect_counter += 1
        if no_detect_counter > 9:
            no_detect_counter = 0
            x_delta = 0
        return
    
    max_score = 0
    for detection in detections:
        if detection.conf >= max_score:
            max_score = detection.conf
    
    labels = get_labels()
    
    with MappedArray(request, stream) as m:
        for detection in detections:
            # Focus only on highest confidence detection
            if detection.conf < max_score:
                continue
            x, y, w, h = detection.box
            no_detect_counter = 0
            
            # Find x delta from the center
            x_delta = x + (w >> 1) - 320
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f}), x_delta: {x_delta}"
            
            # Draw text on top of the background
            cv2.putText(m.array, label, (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, default=True, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="xy",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Detection threshold")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    
    last_results = None
    picam2.pre_callback = draw_detections
    
    try:
        while True:
            last_results = parse_detections(picam2.capture_metadata())
            track_x_delta()
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt: closing the program")
        
    finally:
        GPIO.cleanup()
