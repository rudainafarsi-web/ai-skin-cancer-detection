# AI-Based Skin Cancer Detection Using Deep Learning and Dermoscopic Imaging

An academic embedded AI prototype developed as a Bachelor of Engineering (Honours) graduation project in Electronics and Telecommunications Engineering at Middle East College, Oman.

## Academic Context

- **Project type:** Individual Bachelor’s Graduation Project
- **Completion:** June 2026
- **Institution:** Middle East College, Oman
- **Supervisor:** Dr. Shaik Asif Hussain

## Problem Statement

Access to specialist skin screening can be limited in remote areas. This project explored a portable embedded AI prototype that supports preliminary screening of dermoscopic skin lesion images while keeping final diagnosis with qualified medical specialists.

## System Overview

The prototype uses a fine-tuned MobileNetV2 model to classify dermoscopic images into benign or malignant categories. The model is deployed locally on an Orange Pi 4 Pro using TensorFlow Lite.

## Hardware

- Orange Pi 4 Pro with 6 GB RAM
- High-resolution USB camera
- 7-inch HDMI display
- MicroSD storage
- 5V power supply
- Custom acrylic enclosure

## Software

- Python
- TensorFlow
- Keras
- TensorFlow Lite
- OpenCV
- NumPy
- PyQt5
- Firebase Realtime Database
- Linux

## Workflow

1. Capture a dermoscopic image using a USB camera.
2. Resize the image to 224 × 224 × 3.
3. Normalize pixel values.
4. Process the image using MobileNetV2.
5. Run local inference with TensorFlow Lite on the Orange Pi 4 Pro.
6. Display the prediction, confidence score, and visual heatmap in the graphical interface.
7. Send monitoring results to Firebase for Android application monitoring.

## Dataset and Model

- **Dataset:** ISIC dermoscopic image dataset
- **Selected images:** 7,211
- **Split:** 75% training and 25% testing and validation
- **Classes:** Benign and malignant
- **Input size:** 224 × 224 × 3
- **Architecture:** MobileNetV2 with transfer learning
- **Fine-tuning:** Top 20 layers
- **Batch size:** 32
- **Total epochs:** 25
- **Optimizer:** Adam
- **Loss function:** Binary cross-entropy
- **Deployment model:** Float32 TensorFlow Lite model

## Verified Results

| Metric | Result |
|---|---:|
| Simulation accuracy | 90.43% |
| Hardware accuracy | 89.92% |
| Classification error rate | 9.57% |
| Simulation inference latency | 227 ms |
| Orange Pi inference latency | 248 ms |
| Binary cross-entropy loss | 0.227 |

The project received the **Best Project Poster Award — Spring 2026** from the Department of Computing & Electronics Engineering at Middle East College.

## Repository Contents

This repository currently includes the dataset preparation and image augmentation portion of the training workflow.

The full dataset, Firebase credentials, patient information, private configuration files, and model files are intentionally not included.


## Image Gallery

### Embedded AI Prototype

![Embedded AI prototype with Orange Pi, USB camera, display, and custom enclosure](images/ai-prototype.jpg)

## Running the Code

The current `training.py` file prepares training and validation image generators.

The full model definition, training execution, TensorFlow Lite conversion, and deployment files are not included in this repository yet.

## Medical Disclaimer

This project is an academic preliminary screening and decision-support prototype. It does not replace dermatologist diagnosis, clinical examination, or biopsy.

## Privacy and Security

This repository does not include patient information, Firebase credentials, database URLs, API keys, environment files, private source files, trained model files, or the full ISIC dataset.
