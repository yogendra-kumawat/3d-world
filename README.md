# 🤲 Tactile Vision Aid

<div align="center">

**Monocular Depth → Vibrotactile Pin-Matrix Display System for the Visually Impaired**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)]()
[![STM32](https://img.shields.io/badge/STM32-03234B?style=for-the-badge&logo=stmicroelectronics&logoColor=white)]()
[![YOLO](https://img.shields.io/badge/YOLOv8-FF6600?style=for-the-badge)]()
[![DepthAnything](https://img.shields.io/badge/Depth_Anything_V2-black?style=for-the-badge)]()

</div>

---

## 🎯 What This Does

A blind person wears a camera. The system sees the world through it, and translates the scene into a **16 × 8 grid of physical pins** — each pin rises to a height representing how far away that part of the scene is. By **touching the pin matrix with their fingers**, the user builds a mental 3D model of the space around them in real time.

The system also narrates detected objects aloud via **synthesized speech** so the user knows *what* is in the scene, not just *where* things are.

```
Camera Feed
    │
    ▼
┌─────────────────────────────────────────────┐
│           Vision Pipeline (Python)          │
│                                             │
│  Depth Anything V2 ──► Metric depth map     │
│  YOLOv8            ──► Object bounding boxes│
│  IRLS Ground Plane ──► Floor mask           │
│  Tactile Encoder   ──► 16×8 height matrix   │
│  Speech Engine     ──► Audio narration      │
└──────────────┬──────────────────────────────┘
               │  18-byte UART packets
               ▼
┌─────────────────────────────────────────────┐
│         STM32 Firmware (HAL / C)            │
│  Parse height bytes → drive pin actuators   │
└─────────────────────────────────────────────┘
               │
               ▼
         🤲 Tactile Pin Matrix
        (user touches to perceive scene)
```

---

## 🖥️Led based hardware  & UI

> Live depth map + pin matrix simulator — what the user would feel on the physical device:

<img width="1000" height="469" alt="Simulation Screenshot" src="https://github.com/user-attachments/assets/d5a1472e-5ed9-4a30-b5d1-38e99ea1ded8" />

The left panel shows the **original camera feed**. The right panel shows the **depth heatmap** (red = close, blue = far). The bottom panels show the **pin matrix output** (tactile grid) and the **top-down bird's-eye view** (BEV) of the detected scene.

---

## 🧠 System Architecture

### Three Operating Modes

The system has three selectable modes depending on the user's need:

| Mode | Description | Best For |
|------|-------------|----------|
| **Walk** | Bird's-eye view (BEV) — obstacles shown from above | Navigating through a space |
| **Scene** | Front-facing depth feel — full depth map as pins | Understanding surroundings |
| **Identifier** | YOLO objects placed in 3D — labelled objects only | Identifying specific objects |

---

### 1. 🔍 Depth Estimation (`depth/`)

Uses **Depth Anything V2** (small / base / large) to compute a per-pixel metric depth map from a single RGB camera frame. No stereo camera needed.

- Supports `small`, `base`, `large` model sizes — trade speed for accuracy
- Indoor / Outdoor domain selection adjusts metric calibration
- Output: full-resolution float32 depth map (metres per pixel)

---

### 2. 📦 Object Detection (`detection/`)

Runs **YOLOv8** on every frame to identify and bound objects in the scene.

- Used in **Identifier mode** to place detected objects in 3D space on the pin matrix
- Object names are passed to the **audio module** for speech narration
- Bounding box + depth value → 3D position of each object

---

### 3. 🗺️ Geometry Processing (`geometry/`)

The most complex module — turns raw depth into a navigable tactile representation:

**Ground-Plane Extraction (IRLS)**
- Applies **Iteratively Reweighted Least Squares** with strict sanity gates to fit a ground plane
- A **radial subtraction formula** then masks the floor directly underfoot while preserving the path ahead
- Result: only obstacles and scene structure remain — the flat ground is ignored so it doesn't saturate the pin matrix

**Tactile Encoding**
- Depth map is **min-pooled** into a 16 × 8 grid
- Values are quantized into **8 discrete height levels** (0 = far/low pin, 7 = close/high pin)
- In **Walk mode**: top-down BEV projection
- In **Scene mode**: direct front-facing depth
- In **Identifier mode**: YOLO-gated — only detected objects raise pins

---

### 4. 🤲 Tactile Output (`tactile/`)

Packages the 16 × 8 matrix into a compact **18-byte serial packet** for the STM32:

```
Packet format: b"A09909909909911111"
               │ └──────────────────── zero-padded height strings (0–7 per cell)
               └── header byte 'A'
```

Each chunk encodes one row of the 16-cell grid. The STM32 parses the height values and maps them directly to the physical pin actuator array.

---

### 5. 🔊 Audio Narration (`audio/`)

A speech synthesis module that:
- Receives detected object labels from YOLO
- Announces objects and their estimated distance aloud
- Runs asynchronously so it never blocks the vision pipeline

---

### 6. 🖥️ User Interface (`ui/`)

A live simulation dashboard (see screenshot above) with:
- Real-time camera feed and depth heatmap side by side
- **Pin matrix 3D visualizer** (Matplotlib 3D bars — slow but detailed, or fast 2D mode)
- **Top-down BEV** overlay with floor (gray) and obstacles (colour)
- Collapsible control panels for: pin grid, camera intrinsics, ground plane, clipping, walk region, object detection range

---

## ⚡ Asynchronous Threading Model

To prevent slow serial hardware communication from stalling the AI inference pipeline:

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│     Main AI Thread          │      │    Serial Worker Thread      │
│                             │      │       (daemon)               │
│  Frame → Depth → YOLO       │      │                              │
│  → Ground Plane → 16×8 grid │─────►│  Poll queue                 │
│  → Push to queue            │      │  Transmit latest frame       │
│  → Next frame immediately   │      │  If AI busy → re-send last  │
└─────────────────────────────┘      └──────────────────────────────┘
```

The serial thread **never blocks** the AI. If the AI is still processing frame N+1, the STM32 keeps receiving frame N — the pins stay actuated rather than going dead.

---

## 📁 Project Structure

```
tactile-vision-aid/
├── ui/                  # Live simulation dashboard (Gradio / Tkinter)
├── tactile/             # Pin matrix encoder + serial packet builder
├── geometry/            # Ground plane extraction (IRLS) + tactile encoder
├── detection/           # YOLOv8 object detection wrapper
├── depth/               # Depth Anything V2 inference wrapper
├── audio/               # Text-to-speech narration module
├── pipeline.py          # Main orchestrator — wires all modules together
├── types.py             # Shared dataclass / type definitions
└── __init__.py
```

---

## 🔩 Hardware Requirements

| Component | Purpose |
|-----------|---------|
| RGB Camera (USB / Pi Camera) | Scene input |
| STM32 (or any UART MCU) | Drives physical pin actuators |
| 16 × 8 Vibrotactile Pin Array | The tactile display the user touches |
| USB-to-TTL Adapter | PC ↔ STM32 serial link |
| Speaker / Earphone | Audio narration output |
| Battery + Wearable Frame | For portable use |

> ⚠️ **Power note:** Pin actuator arrays draw significant current. Use a dedicated power supply with common ground to the MCU — do not power from the MCU's 3.3V rail.

---

## 🚀 Setup & Installation

### Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics          # YOLOv8
pip install depth-anything-v2    # or clone the official repo
pip install opencv-python pyserial numpy gradio pyttsx3
```

### Download Model Weights

```bash
# Depth Anything V2 (small — fastest)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# YOLOv8n (nano — fastest)
# Downloads automatically on first run via ultralytics
```

### Run the Pipeline

```bash
python pipeline.py
```

The UI opens in your browser. Select your mode, depth model size, and domain, then start the camera.

### Serial Configuration (for physical hardware)

In `pipeline.py` or `tactile/`, set:

```python
SERIAL_PORT = "COM3"        # Windows: COMx | Linux: /dev/ttyUSBx
BAUD_RATE   = 115200
PIN_ROWS    = 8
PIN_COLS    = 16
HEIGHT_LEVELS = 8           # 0 (far) to 7 (close)
```

---

## 📐 The Pin Matrix Encoding

A 16 × 8 grid = **128 pins**. Each pin gets a height value 0–7.

```
Far away          Close
   0 ──────────► 7
   │              │
Low pin        High pin
(barely raised) (fully raised)
```

**What the user feels:**

- A **flat area** (all zeros) = open path, nothing nearby
- A **raised cluster** = obstacle or object in that direction
- **High pins in the centre** = something directly ahead and close
- **Raised edges** = walls or objects to the sides

The **Walk mode BEV** is the primary navigation mode — the user feels obstacles as raised bumps on a flat map of the space ahead, like a tactile bird's-eye view.

---

## 🔮 Future Improvements

- [ ] **Wireless transmission** — replace USB serial with BLE for a fully wearable untethered device
- [ ] **Haptic intensity** — drive actuators at variable vibration frequency, not just height, for richer feedback
- [ ] **GPS + outdoor mapping** — integrate GPS for outdoor navigation waypoints announced via audio
- [ ] **Edge deployment** — run depth + YOLO on a Raspberry Pi 5 or Jetson Nano onboard the wearable
- [ ] **User calibration** — personalised ground plane and step-height calibration per user
- [ ] **Staircase detection** — dedicated IRLS mode to detect and warn about stairs

---

*Built with ❤️ to make the world more accessible.*
