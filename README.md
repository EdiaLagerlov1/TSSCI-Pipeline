# TSSCI: Time-Series Super Classifier Images for Human Pose Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edialagerlov1/TSSCI-Pipeline/blob/main/TSSCI_Complete_Pipeline.ipynb)

Complete pipeline for converting human movement videos into TSSCI (Time-Series Super Classifier Images) format for exercise classification and analysis.

## 🎯 Overview

This project implements a complete system that:
- 📹 Extracts human skeleton poses from video using **MediaPipe**
- 🔄 Transforms skeleton data to **TSSCI format**
- 🖼️ Generates **TSSCI images** suitable for CNN processing
- 🎬 Creates **interactive visualizations** with dual-view playback

## 🌟 Features

- **Skeleton Extraction**: MediaPipe Pose detection (33 landmarks)
- **TSSCI Transformation**: Converts temporal sequences to 2D images
- **Interactive Players**: Real-time skeleton animation with controls
- **Dual-View Display**: TSSCI image with synchronized skeleton
- **Export Capabilities**: GIF animations and visualizations
- **Google Colab Ready**: Complete notebook for cloud execution

## 📊 Demo

### Input → Output Pipeline

```
Video (240 frames @ 30fps)
    ↓
MediaPipe Pose Detection (33 landmarks)
    ↓
TSSCI Transformation (49 keypoints)
    ↓
TSSCI Image (49×49×2)
    ↓
Interactive Visualization
```

### Sample Output

![TSSCI Visualization](tssci_output/tssci_visualization.png)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Upload your video file
3. Run all cells
4. Download your results

### Option 2: Local Installation

#### Prerequisites

- Python 3.8+
- macOS, Linux, or Windows

#### Installation

```bash
# Clone repository
git clone https://github.com/edialagerlov1/TSSCI-Pipeline.git
cd TSSCI-Pipeline

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv tssci-env

# Activate environment
source tssci-env/bin/activate  # macOS/Linux
# or
tssci-env\Scripts\activate      # Windows

# Install dependencies
uv pip install opencv-python mediapipe numpy matplotlib pillow

# Download MediaPipe model (29.2 MB)
curl -L -o pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

## 📖 Usage

### Step 1: Generate TSSCI from Video

```bash
python3 generate_tssci.py
```

**Output:**
- `tssci_output/skeleton_data.npy` - Raw MediaPipe skeleton data
- `tssci_output/tssci_data.npy` - TSSCI transformed data
- `tssci_output/tssci_image.npy` - Final TSSCI image
- `tssci_output/tssci_visualization.png` - Static visualization

### Step 2: Create Animations (Optional)

```bash
python3 animate_tssci.py
```

**Output:**
- `tssci_output/skeleton_animation_full.gif` - Full animation (6 MB)
- `tssci_output/skeleton_animation_short.gif` - Short preview (2 MB)

### Step 3: Launch Interactive Player

```bash
python3 tssci_player.py
```

Opens interactive window with skeleton animation and playback controls.

### Step 4: Launch Dual-View Player

```bash
python3 tssci_dual_player.py
```

Opens dual-view window showing TSSCI image with scanning line + synchronized skeleton.

## 🎮 Player Controls

| Button | Action |
|--------|--------|
| **Play/Pause** | Toggle animation |
| **Stop** | Reset to frame 0 |
| **< Prev** | Previous frame |
| **Next >** | Next frame |
| **Slower** | Decrease speed (-5 fps) |
| **Faster** | Increase speed (+5 fps) |
| **Export GIF** | Save animation |
| **Slider** | Jump to any frame |

## 📁 Project Structure

```
L44/
├── input video/
│   └── WhatsApp Video 2026-03-14 at 20.10.47.mp4
├── tssci_output/
│   ├── skeleton_data.npy          # Raw MediaPipe data
│   ├── tssci_data.npy             # TSSCI transformed
│   ├── tssci_image.npy            # TSSCI image
│   ├── tssci_visualization.png    # Visualization
│   ├── skeleton_animation_full.gif
│   └── skeleton_animation_short.gif
├── generate_tssci.py              # Main generator
├── animate_tssci.py               # Animation exporter
├── tssci_player.py                # Interactive player
├── tssci_dual_player.py           # Dual-view player
├── pose_landmarker_heavy.task     # MediaPipe model (29.2 MB)
├── TSSCI_Complete_Pipeline.ipynb  # Google Colab notebook
├── PRD.md                         # Product Requirements Document
└── README.md                      # This file
```

## 🔬 Technical Details

### What is TSSCI?

**TSSCI (Time-Series Super Classifier Images)** converts temporal skeleton sequences into 2D images that can be processed by CNNs.

**Key Advantages:**
- Enables use of powerful pre-trained CNNs (EfficientNet, ResNet)
- Compact representation of temporal patterns
- Faster inference than recurrent networks
- Preserves spatial and temporal relationships

### TSSCI Transformation

1. **MediaPipe → OpenPose**: Convert 33 landmarks to 25 keypoints
2. **TSSCI Rearrangement**: Apply specific 49-point pattern
3. **Normalization**: Scale coordinates to [0, 1]
4. **Image Sampling**: Sample 49 frames from sequence
5. **Image Generation**: Create 49×49×2 image (X and Y channels)

### TSSCI Pattern

```python
[1,2,3,4,3,2,1,0,15,17,15,0,16,18,16,0,1,5,6,7,6,5,1,8,
 12,13,14,19,20,19,14,21,14,13,12,8,9,10,11,24,11,22,23,22,11,10,9,8,1]
```

This pattern traces: head → shoulders → arms → torso → legs → feet

## 📊 Performance

| Metric | Value |
|--------|-------|
| Processing Speed | ~30 fps |
| TSSCI Generation | ~2 seconds |
| Total Pipeline | ~10 seconds |
| Memory Usage | <500 MB |
| Output Size | ~7 MB total |

## 🛠️ Dependencies

```
opencv-python==4.13.0.92
mediapipe==0.10.32
numpy==2.4.3
matplotlib==3.10.8
pillow==12.1.1
```

## 🎓 Academic Context

Based on research by **Yoram Segal** (Ben-Gurion University):

**Paper:** "Using EfficientNet-B7 (CNN), Variational Auto Encoder (VAE) and Siamese Twins' Networks to Evaluate Human Exercises as Super Objects in A TSSCI Images"

- 📄 [Research Paper](https://www.preprints.org/manuscript/202304.1268/v1)
- 🎥 [Video Explanation](https://youtu.be/QQf-pyQw8Wc)
- 💻 [Original Repository](https://github.com/yoramse/TSSCI)

## 🚧 Future Enhancements

### Short-term
- [ ] Exercise classification with EfficientNet-B7
- [ ] Batch processing for multiple videos
- [ ] Model training pipeline

### Medium-term
- [ ] VAE for synthetic movement generation
- [ ] Siamese network for trainer comparison
- [ ] Web interface

### Long-term
- [ ] Mobile app with real-time feedback
- [ ] Multi-person pose detection
- [ ] 3D pose estimation

## 📝 Documentation

- **[PRD.md](PRD.md)** - Comprehensive Product Requirements Document
- **[TSSCI_Complete_Pipeline.ipynb](TSSCI_Complete_Pipeline.ipynb)** - Google Colab notebook

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational purposes. Please cite the original research paper when using this work.

## 🙏 Acknowledgments

- **Yoram Segal** - Original TSSCI research
- **Google MediaPipe** - Pose detection library
- **Ben-Gurion University** - Research institution

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for remote physiotherapy and exercise analysis**

