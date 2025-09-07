# Object Tracking with Raspberry Pi & AI Camera (IMX500)

## Introduction

This project enables real-time object tracking using a **Raspberry Pi Zero 2 W** paired with the **Sony IMX500 AI Camera**. Since the Raspberry Pi Zero 2 W lacks the processing power required for on-device AI inference, the IMX500 sensor-capable of edge AI processing-is utilized to handle object detection tasks efficiently.

The AI model used is **YOLOv11n**, trained on the **COCO dataset**, enabling the system to identify and track multiple object classes.

### Key Features

- Lightweight, low-power object tracking solution
- Edge AI with no cloud dependency
- Ideal for embedded or remote applications

---

## Potential Applications

- Smart surveillance systems
- Object-following camera setups
- Automated pan/tilt (PTZ) camera systems
- DIY robotics and AI projects

---

## Hardware Requirements

| Component                 | Description                               |
|---------------------------|-------------------------------------------|
| Raspberry Pi Zero 2 W     | Main SBC (Single Board Computer)          |
| Raspberry Pi AI Camera    | Sony IMX500 sensor for AI inference       |
| 5V DC Stepper Motor       | For panning motion                        |
| ULN2003                   | Stepper motor driver                      |
| 2x LED Lights *(optional)*| For stepper motor movement indicator      |
| DC Power Supply           | 5V 1.5A output minimum                    |

---

## Software Dependencies

Ensure your Raspberry Pi is running Python 3 and has the following libraries installed:

- [RPi.GPIO](https://pypi.org/project/RPi.GPIO/)
- [Picamera2](https://github.com/raspberrypi/picamera2)
- [OpenCV](https://github.com/opencv/opencv-python)
- [NumPy](https://github.com/numpy/numpy)
- [IMX500 Models](https://github.com/raspberrypi/imx500-models)
