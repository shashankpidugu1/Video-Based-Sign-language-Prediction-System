# Video-Based Sign Language Translation System

A lightweight, real-time American Sign Language (ASL) translation system that enables seamless communication between sign language users and non-signers. Unlike conventional deep learning-based solutions that require extensive computational resources and large datasets, this project leverages traditional machine learning techniques for efficient performance on low-end devices.

The system uses a webcam to capture hand gestures in real time and employs MediaPipe for hand landmark detection. Extracted landmarks are classified using a Random Forest Classifier, providing accurate gesture recognition with minimal computational overhead. The model also supports personalized training, allowing users to create custom datasets and adapt the system to their unique signing styles for improved accuracy.

Recognized ASL gestures are converted into text and speech output, making communication more accessible and inclusive for deaf and mute individuals. Designed with a focus on efficiency, adaptability, and real-time performance, this project demonstrates an effective alternative to resource-intensive deep learning approaches while maintaining high recognition accuracy.

## Key Features

* Real-time ASL gesture recognition using a webcam
* Hand landmark detection powered by MediaPipe
* Lightweight Random Forest-based classification
* Personalized model training for custom gesture adaptation
* Text and speech output generation
* Low computational requirements, suitable for resource-constrained devices
* Enhanced accessibility and inclusive communication

## Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn (Random Forest Classifier)
* Text-to-Speech (TTS) Engine

This project aims to bridge the communication gap between sign language users and the wider community through an efficient, accessible, and user-friendly translation system.
