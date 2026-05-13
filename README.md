# 🚗 Autonomous Driving Perception System

> Real-time multi-task perception pipeline for autonomous vehicles — Lane Detection, Object Detection, Multi-Object Tracking, and Depth Estimation.
> Built to demonstrate Tesla Autopilot-style computer vision engineering.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Overview

This project implements a full autonomous driving perception stack that mirrors the core vision systems used in production self-driving vehicles. Given a driving video or camera feed, the system simultaneously:

- Detects and classifies objects (cars, pedestrians, cyclists, trucks)
- Tracks detected objects across frames with unique IDs
- Identifies lane boundaries and road geometry
- Estimates per-object depth/distance from a monocular camera
- Serves real-time results via a FastAPI inference endpoint

---

## 🏗️ Architecture

```
Input Video / Camera Stream
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  ← Frame extraction, resize, normalize
│   (OpenCV)          │
└────────┬────────────┘
         │
   ┌─────┴──────┐
   ▼            ▼
┌──────────┐  ┌──────────────┐
│  Lane    │  │   Object     │
│Detection │  │  Detection   │
│(PolyFit) │  │  (YOLOv8)    │
└──────────┘  └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   Tracking   │
              │  (ByteTrack) │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │    Depth     │
              │  Estimation  │
              │   (MiDaS)    │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  FastAPI     │
              │  Endpoint    │ ← JSON output + annotated frame
              └──────────────┘
```

---

## 📁 Project Structure

```
autonomous-driving-perception/
├── src/
│   ├── detection/          # YOLOv8 object detection module
│   ├── tracking/           # ByteTrack multi-object tracking
│   ├── lane/               # Lane detection & polynomial fitting
│   └── depth/              # MiDaS monocular depth estimation
├── models/                 # Pretrained & fine-tuned weights
├── data/                   # Dataset samples (BDD100K / nuScenes)
├── configs/                # Model & pipeline configuration YAMLs
├── notebooks/              # EDA, training, evaluation notebooks
├── api/                    # FastAPI inference server
├── scripts/                # Training, evaluation, export scripts
├── tests/                  # Unit and integration tests
└── outputs/                # Annotated videos and inference results
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | ByteTrack / DeepSORT |
| Lane Detection | OpenCV + Polynomial Fitting |
| Depth Estimation | MiDaS (Intel) |
| Training Framework | PyTorch 2.x |
| Inference API | FastAPI |
| Containerization | Docker |
| Experiment Tracking | Weights & Biases |
| Dataset | BDD100K / nuScenes |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
CUDA 11.8+ (GPU recommended)
Docker (optional)
```

### Installation
```bash
git clone https://github.com/rajkumar-prog/autonomous-driving-perception.git
cd autonomous-driving-perception
pip install -r requirements.txt
```

### Run Inference on a Video
```bash
python scripts/run_inference.py --input data/sample_drive.mp4 --output outputs/result.mp4
```

### Start FastAPI Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker
```bash
docker build -t autopilot-perception .
docker run -p 8000:8000 autopilot-perception
```

---

## 📊 Datasets

| Dataset | Size | Download |
|---|---|---|
| BDD100K | 100K driving videos | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) |
| nuScenes | 1000 driving scenes | [nuscenes.org](https://nuscenes.org) |
| KITTI | Stereo + LiDAR data | [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti) |

---

## 📈 Results

| Task | Metric | Score |
|---|---|---|
| Object Detection | mAP@0.5 | TBD |
| Lane Detection | Accuracy | TBD |
| Tracking | MOTA | TBD |
| Depth Estimation | AbsRel Error | TBD |

---

## 🔮 Roadmap

- [ ] YOLOv8 baseline object detection
- [ ] Lane detection with polynomial fitting
- [ ] ByteTrack multi-object tracking integration
- [ ] MiDaS depth estimation
- [ ] FastAPI inference server
- [ ] ONNX export for edge deployment (30 FPS target)
- [ ] Docker containerization
- [ ] W&B experiment tracking dashboard
- [ ] TensorRT optimization for real-time inference

---

## 👤 Author

**Raj Kumar Satya**
- GitHub: [@rajkumar-prog](https://github.com/rajkumar-prog)
- LinkedIn: [Raj Kumar Satya](https://linkedin.com/in/rajkumarsatya)
- Email: rajkumarsatya65@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
