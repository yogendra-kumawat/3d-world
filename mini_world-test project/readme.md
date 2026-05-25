# 💡 Tactile Vision Aid — LED Prototype (Mini Version)

<div align="center">

**Depth → LED Intensity Matrix | Proof-of-Concept for the Physical Pin Display**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![STM32](https://img.shields.io/badge/STM32-03234B?style=for-the-badge&logo=stmicroelectronics&logoColor=white)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-FF6600?style=for-the-badge)]()
[![DepthAnythingV2](https://img.shields.io/badge/Depth_Anything_V2-black?style=for-the-badge)]()

</div>

> **Relationship to the full system:** The full Tactile Vision Aid uses physical vibrating pins whose *height* encodes obstacle proximity. This mini version replaces the physical pins with a **5 × 4 LED matrix** where *brightness* encodes the same height data — making it cheap and fast to prototype and validate the entire depth-to-display pipeline before building the mechanical pin array.

---

## 🧭 Table of Contents

- [The Core Idea](#-the-core-idea)
- [Full Pipeline](#-full-pipeline)
- [Python Side — pro.ipynb](#-python-side--proipynb)
- [Firmware Side — main.c](#-firmware-side--mainc)
- [The 61-Byte Serial Protocol](#-the-61-byte-serial-protocol)
- [LED Matrix Multiplexing](#-led-matrix-multiplexing)
- [Simulation Dashboard](#-simulation-dashboard)
- [Configuration Reference](#-configuration-reference)
- [Setup & Installation](#-setup--installation)

---

## 💡 The Core Idea

In the physical device, **pin height = obstacle closeness**.
In this LED prototype, **LED brightness = obstacle closeness**.

```
Obstacle distance     Pin height (full)    LED brightness (this version)
──────────────────    ─────────────────    ─────────────────────────────
Very close (< 30%)  → Fully raised pin   → Bright LED  🔴 Danger
Mid range  (30–60%) → Half-raised pin    → Medium LED  🟡 Caution
Far away   (> 60%)  → Low / flat pin     → Dim LED     🔵 Clear
```

This makes it possible to verify the entire AI → encoding → serial → hardware pipeline with nothing more than a breadboard and 20 LEDs.

---

## ⚙️ Full Pipeline

```
Video Frame (vid.mp4 or webcam)
        │
        ▼
┌───────────────────────────────────────────┐
│         Depth Anything V2 (vits)          │
│   Single RGB frame → metric depth map    │
│   max_depth = 20 m  |  dataset = vkitti  │
└──────────────────┬────────────────────────┘
                   │ full-res float32 depth
                   ▼
┌───────────────────────────────────────────┐
│              YOLOv8n Detection            │
│   Bounding boxes → boost pins to ≥ 0.8  │
│   (detected objects always prominently   │
│    raised regardless of raw depth)       │
└──────────────────┬────────────────────────┘
                   │ boosted depth
                   ▼
┌───────────────────────────────────────────┐
│         depth_to_pin_heights()            │
│                                           │
│  1. Resize depth to 4 × 5 (INTER_AREA)   │
│  2. Normalise → [0, 1]                    │
│  3. INVERT  (close = high value)          │
│  4. IIR temporal filter (α = 0.4)         │
│     new = 0.4 × current + 0.6 × prev     │
└──────────────────┬────────────────────────┘
                   │ 4×5 float matrix [0,1]
                   ▼
┌───────────────────────────────────────────┐
│         classify_danger_zones()           │
│   > 0.7 → danger (2)                     │
│   > 0.4 → caution (1)                    │
│   else  → clear (0)                      │
└──────────────────┬────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Matplotlib             Serial Packet
 5-panel dashboard      (61 bytes → STM32)
 (simulation)           LED brightness
```

---

## 🐍 Python Side — `pro.ipynb`

### Key Functions

#### `depth_to_pin_heights(depth_map, prev_heights)`

The heart of the system. Converts a full-resolution depth map to a 4 × 5 normalised grid:

```python
grid       = cv2.resize(depth_map, (PIN_COLS=5, PIN_ROWS=4), INTER_AREA)
normalised = (grid - d_min) / (d_max - d_min)   # → [0, 1]
pin_heights = 1.0 - normalised                   # invert: close = 1, far = 0

# Temporal IIR filter (prevents flickering between frames)
pin_heights = 0.4 * pin_heights + 0.6 * prev_heights
```

#### `overlay_yolo_on_pins(pin_heights, boxes, frame_shape)`

Ensures YOLO-detected objects are always prominently raised, even if the raw depth is ambiguous:

```python
# For every detected bounding box:
highlighted[pr1:pr2, pc1:pc2] = np.maximum(
    highlighted[pr1:pr2, pc1:pc2], 0.8)  # clamp up to 0.8 minimum
```

#### `classify_danger_zones(pin_heights)`

Returns a per-cell danger level used for colour coding in the simulator and for future audio alert priority:

| Level | Value | Threshold | Colour |
|-------|-------|-----------|--------|
| Clear | 0 | pin < 0.4 | 🔵 `#4a9eff` |
| Caution | 1 | pin > 0.4 | 🟡 `#ffaa00` |
| Danger | 2 | pin > 0.7 | 🔴 `#ff4444` |

#### `run_tactile_video(video_path, ...)`

Main video loop — processes every `skip`-th frame, renders the 5-panel dashboard, and sends serial packets to the STM32:

```python
run_tactile_video(
    video_path = "vid.mp4",
    model      = model,        # Depth Anything V2
    model_yolo = model_yolo,   # YOLOv8n
    skip       = 1,            # every frame
    max_frames = None,         # full video
)
```

Optionally saves a rendered output video (`tactile_output.mp4`) showing all 5 panels side by side.

---

## ⚙️ Firmware Side — `main.c`

### What It Does

The STM32 receives 61-byte packets, parses a 5 × 4 height matrix from them, and **multiplexes** that matrix onto the LED grid using hardware PWM.

### Initialization

```c
// Four PWM channels → four LED columns (cathode / ground side)
HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);  // Column 1
HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);  // Column 2
HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);  // Column 3
HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);  // Column 4

// Arm interrupt receiver (61 bytes per packet)
HAL_UART_Receive_IT(&huart1, rx_data, 61);
```

### UART Interrupt — Parsing the Matrix

```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (rx_data[0] == 'A') {           // Verify header
        int offset = 1;
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 4; c++) {
                // Convert 3 ASCII digits → integer (e.g. "007" → 7)
                prev_heights[r][c] =
                    (rx_data[offset]   - '0') * 100 +
                    (rx_data[offset+1] - '0') * 10  +
                    (rx_data[offset+2] - '0');
                offset += 3;
            }
        }
    }
    HAL_UART_Receive_IT(&huart1, rx_data, 61);  // re-arm
}
```

### Main Loop — Row-Multiplexed Scanning

The LEDs are driven with a **row-scan / column-PWM** scheme:

```c
// Row pins = PA0–PA4  (provide VCC — only one active at a time)
// Col channels = TIM1 CH1–CH4  (provide PWM ground → controls brightness)

for (int r = 0; r < 5; r++) {

    // Turn all rows off
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0|...|GPIO_PIN_4, GPIO_PIN_RESET);

    // Set PWM for each column based on height
    for (int c = 0; c < 4; c++) {
        int h            = prev_heights[r][c];    // 0–4
        int inverted_h   = 4 - h;                 // invert: h=4 → PWM=0 (bright)
        int pulse        = (inverted_h * 999) / 4;
        __HAL_TIM_SET_COMPARE(&htim1, col_channels[c], pulse);
    }

    // Activate this row
    HAL_GPIO_WritePin(GPIOA, row_pins[r], GPIO_PIN_SET);
    HAL_Delay(2);   // hold for 2 ms → 10 ms full cycle → 100 Hz refresh
}
```

> **Why inverted PWM?** The columns provide the **ground** (cathode) side via PWM. A **lower** compare value = less time at ground = LED is ON more. So `inverted_h = 4 - h` maps height 4 (obstacle close, should be bright) to PWM = 0 (full on).

---

## 📦 The 61-Byte Serial Protocol

```
Byte 0      : 'A'  (header / sync byte)
Bytes 1–60  : 5 rows × 4 cols × 3 ASCII digits = 60 bytes

Example packet:
  A 007 014 000 020 003 011 019 004 ...
  │ └─┘ └─┘ └─┘ └─┘                 20 values total
  │ [0,0][0,1][0,2][0,3]  ← row 0
  header
```

**Encoding in Python:**

```python
# pin_heights are in [0,1] → scale to 0–20 integer range
prev = prev_heights * 20          # 4×5 float → 0–20
h = int(prev[r, c])               # single cell height
h_bytes = str(h).zfill(3).encode()  # "007"
```

**Why 9600 baud?**
At 9600 baud, one 61-byte packet takes ~64 ms → ~15 packets/sec. Sufficient for this LED prototype. The full physical pin system uses 230,400 baud for video framerates.

---

## 📊 Simulation Dashboard

The notebook renders a live **5-panel figure** per frame:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   Panel 1    │   Panel 2    │   Panel 3    │   Panel 4    │
│ Camera+YOLO  │  Depth Map   │ Tactile Grid │ Danger Zone  │
│ bounding     │  (Magma      │  (Top View)  │  Heatmap     │
│ boxes        │  colormap)   │  circles     │  R/Y/G       │
└──────────────┴──────────────┴──────────────┴──────────────┘
┌──────────────────────────────────────────┬─────────────────┐
│           Panel 5 (spans 3 cols)         │                 │
│     3D Pin Array — isometric bar chart   │                 │
│     bar height = pin extension in mm     │                 │
│     bar colour = danger level            │                 │
└──────────────────────────────────────────┴─────────────────┘
```

- **Panel 3 (2D grid):** Circles whose *size* scales with pin height, coloured by danger level
- **Panel 5 (3D bars):** Physical pin height in mm (0–8 mm range), rotatable isometric view
- Saves output as `tactile_output.mp4` with all panels composited

---

## 🔧 Configuration Reference

All tunable parameters live in `TactileConfig`:

```python
class TactileConfig:
    DEVICE_WIDTH_MM   = 80       # Physical device width
    DEVICE_HEIGHT_MM  = 140      # Physical device height

    PIN_COLS          = 5        # Must match STM32 row_pins count
    PIN_ROWS          = 4        # Must match STM32 col_channels count
    PIN_MIN_HEIGHT    = 0.0      # Normalised minimum
    PIN_MAX_HEIGHT    = 8.0      # Physical pin travel in mm
    PIN_DIAMETER_MM   = 2.5      # Pin size for spacing calculations

    DANGER_NEAR       = 0.3      # pin > 0.7 → danger
    DANGER_MID        = 0.6      # pin > 0.4 → caution
    DANGER_FAR        = 1.0      # else      → clear

    TEMPORAL_ALPHA    = 0.4      # IIR blend: 0 = frozen, 1 = instant
```

### Depth Model Options

| Encoder | Speed | Accuracy | Weights file |
|---------|-------|----------|--------------|
| `vits`  | ⚡ Fast | Good | `depth_anything_v2_metric_hypersim_vits.pth` |
| `vitb`  | Medium | Better | `depth_anything_v2_metric_hypersim_vitb.pth` |
| `vitl`  | Slow | Best | `depth_anything_v2_metric_hypersim_vitl.pth` |

| Dataset | Use for |
|---------|---------|
| `hypersim` | Indoor scenes |
| `vkitti` | Outdoor scenes |

---

## ⏱️ Timer Configuration (STM32)

### TIM1 — LED PWM (Columns)

| Parameter | Value | Calculated |
|-----------|-------|------------|
| Clock | 8 MHz (HSI) | — |
| Prescaler | 7 | Effective clock = 1 MHz |
| Period (ARR) | 999 | PWM resolution = 1000 steps |
| PWM Frequency | 1 MHz / 1000 = **1 kHz** | Flicker-free per LED |
| Channels | 1, 2, 3, 4 | One per column |

With a 2 ms row-scan dwell and 5 rows, the full matrix refresh rate is **100 Hz** — imperceptible flicker.

---

## 🚀 Setup & Installation

### Hardware Needed

| Component | Qty | Notes |
|-----------|-----|-------|
| STM32F103 ("Blue Pill") | 1 | |
| LEDs (any colour) | 20 | 5 rows × 4 cols |
| Current-limiting resistors (220Ω) | 20 | One per LED |
| USB-to-TTL adapter | 1 | CH340 or CP2102 |
| Breadboard + jumper wires | — | |

### GPIO Pinout

| STM32 Pin | Direction | Connected To |
|-----------|-----------|--------------|
| PA0–PA4 | Output HIGH (row scan) | LED anodes (via resistor) |
| PA8 (TIM1_CH1) | PWM out (column sink) | LED cathodes — column 1 |
| PA9 (TIM1_CH2) | PWM out (column sink) | LED cathodes — column 2 |
| PA10 (TIM1_CH3) | PWM out (column sink) | LED cathodes — column 3 |
| PA11 (TIM1_CH4) | PWM out (column sink) | LED cathodes — column 4 |
| PC13 | Output | Onboard LED (blinks on each packet) |
| PA9/PA10 (USART1) | UART RX/TX | USB-to-TTL adapter |

### Python Dependencies

```bash
pip install torch torchvision
pip install ultralytics          # YOLOv8
pip install opencv-python pyserial numpy matplotlib
```

### Download Model Weights

```bash
# Depth Anything V2 metric (indoor, small encoder)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/\
depth_anything_v2_metric_hypersim_vits.pth -P checkpoints/
```

### Run

1. Flash `main.c` to the STM32 via STM32CubeIDE
2. Connect USB-to-TTL adapter, note the COM port
3. In `pro.ipynb` set:
   ```python
   PORT      = 'COM6'     # your port
   BAUD_RATE = 9600
   ```
4. Place a video file as `vid.mp4` in the notebook directory
5. Run all cells — the dashboard renders live and the STM32 LEDs mirror the depth

---

## 🔮 Path to the Physical Device

This LED prototype validates the full software pipeline. To upgrade to physical pins:

| LED prototype | Physical pin device |
|---------------|---------------------|
| LED brightness | Pin height (mm) |
| TIM1 PWM → LED cathode | TIM1 PWM → solenoid / servo driver |
| 5 levels (0–4) | 8 discrete height levels |
| 9,600 baud | 230,400 baud |
| No position sync needed | Hall sensor for rotation lock |

---

*Built with ❤️ — a step toward making the world accessible to everyone.*
