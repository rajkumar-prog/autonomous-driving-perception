# ⚡ NeurOcc — 4D Neuromorphic Occupancy Forecasting

> The first open-source system that combines **event cameras** with **4D voxel occupancy prediction** to forecast the future state of a driving scene — bridging neuromorphic sensing and autonomous vehicle world modeling.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-SNN-purple)](https://github.com/freemlwang/spikingjelly)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 Why This Doesn't Exist Yet

Every autonomous driving perception system today uses **frame cameras** (30fps). A child running onto a road takes ~80ms — invisible at 30fps. **Event cameras** capture brightness changes at **microsecond resolution**, firing asynchronously per-pixel.

Existing work:
- Event cameras → object **detection** (present state only)
- Frame cameras → 3D occupancy (present state only)

**What NeurOcc does that nobody has:**
> Fuses event camera streams with RGB frames through a **Spiking Neural Network**, then predicts **4D voxel occupancy** — which 3D grid cells will be occupied at `t+1s`, `t+2s`, `t+3s` in the future.

This is the perception-prediction stack Tesla's world model team is racing toward. No open repo does it.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     SENSOR INPUTS                       │
│  Event Camera Stream          RGB Frame Cameras         │
│  (μs-resolution events)       (standard 30fps)          │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────┐
│  SNN Event Encoder   │    │  CNN Frame Encoder      │
│  (SpikingJelly)      │    │  (ResNet-50 backbone)   │
│  Spike trains →      │    │  Spatial features →     │
│  temporal features   │    │  voxel feature volume   │
└──────────┬───────────┘    └────────────┬────────────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
           ┌─────────────────────┐
           │   Hybrid Fusion     │
           │   Cross-attention   │
           │   event × frame     │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │  4D Occupancy       │
           │  Transformer        │
           │                     │
           │  Input: fused BEV   │
           │  Output: voxel grid │
           │  [X, Y, Z, T]       │
           │  T = {+1s,+2s,+3s}  │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │  Collision Risk     │
           │  Scorer             │
           │  per future frame   │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │  FastAPI Endpoint   │
           │  + 3D Visualizer    │
           └─────────────────────┘
```

---

## 📁 Project Structure

```
neurocc-4d-occupancy-forecasting/
├── src/
│   ├── event_processing/     # Event stream parsing, voxel-grid conversion
│   ├── snn_encoder/          # Spiking Neural Network encoder (SpikingJelly)
│   ├── fusion/               # Cross-attention event × frame fusion module
│   ├── occupancy_predictor/  # 4D Transformer for future voxel prediction
│   ├── risk_scorer/          # Per-timestep collision risk computation
│   └── visualizer/           # BEV + 3D voxel visualization (Open3D)
├── models/
│   ├── snn/                  # SNN weights & configs
│   ├── occupancy/            # 4D occupancy transformer weights
│   └── fusion/               # Fusion module weights
├── data/
│   ├── dsec/                 # DSEC event-camera driving dataset
│   ├── nuscenes/             # nuScenes 3D occupancy labels
│   ├── processed/            # Preprocessed voxel grids
│   └── samples/              # Quick-start sample sequences
├── api/                      # FastAPI inference server
├── configs/                  # YAML configs for all modules
├── notebooks/                # EDA, training, evaluation notebooks
├── scripts/                  # Training, evaluation, ONNX export scripts
└── tests/                    # Unit + integration tests
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Event Encoding | SpikingJelly (SNN) | Native spike-train processing, energy-efficient |
| Frame Encoding | PyTorch + ResNet-50 | Proven spatial feature extraction |
| Fusion | Custom Cross-Attention | Aligns event timing with frame features |
| 4D Prediction | Custom Transformer | Temporal + spatial attention across voxels |
| Visualization | Open3D + Matplotlib | 3D voxel rendering |
| Inference API | FastAPI | Production serving |
| Experiment Tracking | W&B | Full training observability |
| Primary Dataset | DSEC (event cam driving) | Real driving + event camera ground truth |

---

## 🚀 Getting Started

```bash
git clone https://github.com/rajkumar-prog/autonomous-driving-perception.git
cd autonomous-driving-perception
pip install -r requirements.txt
```

### Download DSEC Dataset
```bash
python scripts/download_dsec.py --split train --output data/dsec/
```

### Preprocess Events to Voxel Grids
```bash
python scripts/preprocess_events.py --input data/dsec/ --output data/processed/
```

### Train the Full Pipeline
```bash
python scripts/train.py --config configs/neurocc_full.yaml --wandb
```

### Run 4D Occupancy Inference
```bash
python scripts/infer.py \
  --events data/samples/sequence_01/events.h5 \
  --frames data/samples/sequence_01/frames/ \
  --future-steps 3 \
  --visualize
```

### Start FastAPI Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 📊 Datasets

| Dataset | Content | Link |
|---|---|---|
| DSEC | Driving sequences with event cameras (Zurich) | [dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch) |
| nuScenes | 3D occupancy annotations | [nuscenes.org](https://nuscenes.org) |
| Occ3D | Dense 3D occupancy labels | [GitHub](https://github.com/FANG-MING/occupancy-for-nuscenes) |

---

## 🔮 Roadmap

- [ ] Event stream to voxel grid preprocessing pipeline
- [ ] SNN encoder with SpikingJelly (membrane potential dynamics)
- [ ] ResNet-50 frame encoder with BEV projection
- [ ] Cross-attention event x frame fusion module
- [ ] 4D Occupancy Transformer (predict t+1s, t+2s, t+3s)
- [ ] Collision risk scoring per predicted timestep
- [ ] ONNX export targeting 30 FPS on edge hardware
- [ ] W&B training dashboard with voxel visualizations
- [ ] FastAPI inference endpoint
- [ ] Docker deployment

---

## 📄 Related Work

- [Autonomous Driving with SNNs (NeurIPS 2024)](https://arxiv.org/abs/2405.19687) — detection only, no future prediction
- [EAS-SNN](https://ieeexplore.ieee.org/) — event-based detection, no world model
- [Tesla FSD Neural Occupancy](https://tesla.com/AI) — proprietary, not open

**NeurOcc is the first open system connecting event cameras + SNNs + 4D future occupancy prediction.**

---

## 👤 Author

**Raj Kumar Satya** — AI/ML Engineer | W&B • Possible Finance • Razorpay
- GitHub: [@rajkumar-prog](https://github.com/rajkumar-prog)
- Email: rajkumarsatya65@gmail.com
