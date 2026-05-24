<div align="center">
  <h1>Tactile Vision Aid</h1>
  <p><b>Monocular Depth to Pin-Matrix Display System</b></p>
</div>

<hr>

<h2>🎯 Project Goal</h2>
<p>
  The objective of this project is to provide a software simulation and hardware communication bridge for a wearable tactile vision aid. The system is built to translate a single RGB camera stream into a dynamic 16&times;8 vibrotactile pin matrix byte stream and synthesized speech narration. It is designed to run on commodity hardware and outputs a documented, low-bandwidth byte stream intended for direct consumption by a physical microcontroller (e.g., STM32).
</p>

<h2>🧠 System Logic & Architecture</h2>

<p>The system is divided into a Python vision orchestrator and a C-based Hardware Abstraction Layer (HAL).</p>

<h3>1. Vision Processing Pipeline</h3>
<p>For every incoming video frame, the software executes the following sequence:</p>
<ul>
  <li><b>Depth Estimation:</b> Utilizes Depth Anything V2 to compute a metric depth map of the environment.</li>
  <li><b>Object Detection:</b> Employs YOLOv8 to identify and bound objects within the frame.</li>
  <li><b>Ground-Plane Extraction:</b> Applies iteratively reweighted least squares (IRLS) with strict sanity gates to identify the floor. A radial subtraction formula is then used to mask the ground directly underfoot while preserving the path ahead.</li>
  <li><b>Tactile Encoding:</b> The spatial data is min-pooled into a 16&times;8 matrix with 8 discrete height levels. The encoding switches between a front-facing "Scene Mode" and a YOLO-gated top-down "Walk Mode".</li>
</ul>

<h3>2. Asynchronous Threading Model</h3>
<p>To prevent slow serial hardware communication from bottlenecking the AI inference, the system uses a decoupled multithreaded architecture:</p>
<ul>
  <li><b>Main AI Thread:</b> Processes the video frame, calculates the matrices, places the 16&times;8 grid data into a single-item queue, and immediately advances to the next frame.</li>
  <li><b>Serial Worker Thread:</b> A persistent daemon thread continuously polls the queue. It transmits the latest available frame over PySerial to the MCU. If the AI is still processing the next frame, this thread repeatedly re-sends the last known frame, ensuring continuous hardware actuation without freezing the main software loop.</li>
</ul>

<h3>3. Hardware Abstraction Layer (HAL)</h3>
<p>
  The C firmware on the target microcontroller listens on the UART/USART interface for the tactile byte stream. The Python pipeline packages the 16&times;8 grid data into highly compressed 18-byte chunks (e.g., <code>b"A09909909909911111"</code>). The microcontroller parses these zero-padded height strings and maps them directly to the physical actuator array.
</p>
