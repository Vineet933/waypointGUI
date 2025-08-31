# Auto Bike Control Panel (PyQt5)

**Title:** Capstone Projects Auto-bicycle TVS IQL  
**Subheader:** Auto Bike Control Panel

---

## Features
- Rear & Steering motor control (PWM + direction) via **buttons** or **keyboard**
- 4× ultrasonic distance bars with per-sensor **thresholds** and a global **alert banner**
- USB camera live view with **optional YOLOv8** overlay
- **Edge View** (Canny) below the main camera
- **MiniMap**: circular FOV with Front/Rear/Left/Right quadrants (green/red) + steering angle line
- Single `CONFIG` dict for all pins, limits, thresholds, and camera index
- One-click packaging with PyInstaller

---

## 1. Hardware (BCM numbering)

### Motors
- **Rear Motor:** PWM (GPIO18), DIR (GPIO23)
- **Steering Motor:** PWM (GPIO13), DIR (GPIO19)  
Use an L298N/L293D H-bridge. Power motors separately; share ground with the Pi.

### Ultrasonics
Default pins in `CONFIG`:
```python
"ultrasonic": {
  "front": {"trig": 23, "echo": 24},
  "rear":  {"trig": 17, "echo": 27},
  "left":  {"trig": 5,  "echo": 6},
  "right": {"trig": 22, "echo": 27}
}
```
⚠️ Level-shift **ECHO** down to 3.3 V.

### Camera
USB camera, default index `0`.

### Buzzer
Optional: set `CONFIG["buzzer_pin"]`.

---

## 2. Installation

### Option A: Recommended on Raspberry Pi OS
```bash
sudo apt update
sudo apt install -y python3-pyqt5 python3-opencv python3-pip libatlas-base-dev
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics   # optional, YOLO
python3 -m pip install RPi.GPIO
```

### Option B: Cross-platform
```bash
python3 -m pip install -r requirements.txt
```

---

## 3. Run
```bash
python3 auto_bike_gui.py
```

### Control Modes
- **GUI buttons**: Forward, Reverse, Left, Right, Stop
- **Keyboard**: enable in right panel  
  - ↑ = Forward  
  - ↓ = Reverse  
  - ← = Steer Left  
  - → = Steer Right  
  - H = Increase speed  
  - L = Decrease speed  
  - Space = Stop

---

## 4. YOLO Detection
- Toggle **Enable YOLO** in right panel.
- Set “Run YOLO every N frames” (3–5 recommended on Pi).
- Uses `yolov8n.pt` by default.

---

## 5. MiniMap & Thresholds
- Circle shows quadrants F/L/R/Rear.
- Green if distance ≥ threshold; red if below.
- Thresholds adjustable in UI.
- Steering angle shown as a yellow line.

---

## 6. One-click Packaging
```bash
python3 -m pip install pyinstaller
pyinstaller --onefile --windowed --name AutoBikeGUI auto_bike_gui.py
```

Desktop entry (`~/.local/share/applications/auto_bike_gui.desktop`):
```ini
[Desktop Entry]
Type=Application
Name=Auto Bike Control Panel
Comment=Capstone Projects Auto-bicycle TVS IQL
Exec=/home/pi/dist/AutoBikeGUI
Icon=utilities-system-monitor
Terminal=false
Categories=Utility;Development;
```

---

## 7. Troubleshooting
- **Camera not found**: change `camera_index` or check with `v4l2-ctl --list-devices`
- **Ultrasonic noise**: ensure level shifting, common GND, short wires
- **Motor not moving**: check PSU, driver wiring, EN pins, GUI speed > deadband
- **YOLO too slow**: run every 5 frames or disable
