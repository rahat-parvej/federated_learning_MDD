# EEG-Based Major Depressive Disorder (MDD) Detection using Federated Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Django 4.x](https://img.shields.io/badge/Django-4.x-092E20.svg)](https://www.djangoproject.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![DOI](https://img.shields.io/badge/DOI-10.XXXXX/XXXXX-blue)](https://doi.org/) -->

A privacy-preserving web-based federated learning system for detecting Major Depressive Disorder (MDD) using EEG signals. This framework enables multiple healthcare institutions to collaboratively train machine learning models without sharing sensitive patient data.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Contact](#contact)

## 🌟 Overview

This repository implements a federated learning framework for EEG-based detection of Major Depressive Disorder (MDD). The system addresses critical challenges in healthcare AI:

- **Data Privacy**: Raw EEG data never leaves the client institutions
- **Collaborative Learning**: Multiple hospitals can improve models together
- **Clinical Interpretability**: Spectral entropy analysis provides biomarkers
- **Real-World Deployment**: Web-based interface for practical use

**Novel Contributions:**
- First federated learning implementation for EEG-based MDD detection
- Novel loss-weighted aggregation for non-IID data distributions
- Four FL aggregation strategies for comprehensive comparison
- Web-based Django platform with client-server architecture
- Integration of spectral entropy for clinical decision support

## 🚀 Key Features

### Privacy Protection
- Raw EEG data remains on client devices
- Only model updates (weights) are shared
- Encrypted client-server communication
- No central data storage

### Multiple Aggregation Strategies
1. **Proposed Method**: Loss-weighted aggregation
2. **FedAvg**: Sample-weighted standard federated averaging
3. **Simple Averaging**: Equal weights for all clients
4. **FedNova**: Normalized by local computation steps

### EEG Processing Pipeline
- Gamma band (30-100 Hz) extraction
- 15-second epoch segmentation
- Spectral entropy calculation
- Z-score normalization
- 20-channel selection (10-20 system)

### Technical Stack
- **Backend**: Django 4.x, TensorFlow 2.x, MNE-Python
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: Keras with 1D CNN architecture
- **Data Processing**: NumPy, SciPy, scikit-learn

### Dataset
**Public EEG Dataset**
This project uses the publicly available EEG dataset from:

Mumtaz, W., Xia, L., Ali, S. S. A., Yasin, M. A. M., Husaini, M., & Malik, A. S. (2017). _Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD)_. Biomedical Signal Processing and Control, 31, 108-115.

**Dataset Specifications:**

Subjects: 30 Healthy Controls, 34 MDD Patients

Age: HC = 40.33 ± 12.86, MDD = 38.23 ± 15.64

Recording: 5-minute EEG, 256 Hz sampling rate

Electrodes: 19 channels (10-20 system)

Conditions: Eyes closed, eyes open, P300 task

Format: EDF files
