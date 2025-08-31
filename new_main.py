#!/usr/bin/env python3
"""
Auto Bike Control Panel – PyQt GUI (Raspberry Pi)
Title: "Capstone Projects Auto-bicycle TVS IQL" – "Auto Bike Control Panel"

Modern GUI:
- Rear & Steering motor control with buttons or keyboard (arrows/H/L/Space)
- Ultrasonic bars + per-sensor thresholds + alert banner
- Camera preview + YOLO toggle + Edge view
- MiniMap (circle FOV + steering angle line)
- Camera picker (device dropdown) + Rescan + robust open across backends
- Falls back to 640×480 if requested size not supported; runtime switch cameras
- One CONFIG block for all knobs

NOTE: Runs on PCs without GPIO (safe console prints). On Pi, enable GPIO pins.
"""

import sys
import time
import threading
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

# -------------- CONFIG --------------
CONFIG = {
    "camera_index": 0,
    "video_size": (960, 540),

    # Rear Motor (H-bridge: PWM enable + DIR)
    "rear": {"ena": 18, "dir": 23, "freq": 1000, "invert": False, "deadband": 8, "ramp_pct_per_s": 120, "max_duty": 100},

    # Steering Motor (H-bridge: PWM enable + DIR)
    "steer": {"ena": 13, "dir": 19, "freq": 1000, "invert": False, "deadband": 10, "ramp_pct_per_s": 150, "max_duty": 100},

    # Ultrasonic pins (HC-SR04) – replace with your BCM pins
    "ultrasonic": {
        "front": {"trig": 23, "echo": 24},
        "rear":  {"trig": 17, "echo": 27},
        "left":  {"trig": 5,  "echo": 6},
        "right": {"trig": 22, "echo": 27},
    },

    # Default thresholds (cm)
    "thresholds": {"front": 60, "rear": 60, "left": 40, "right": 40},

    # Optional buzzer pin (set to BCM pin or None)
    "buzzer_pin": None,
}

USE_GPIO = False
try:
    import RPi.GPIO as GPIO  # type: ignore
    USE_GPIO = True
except Exception:
    USE_GPIO = False

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # pip install ultralytics
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

FRAME_W, FRAME_H = CONFIG["video_size"]

# ----------------- Motor class with soft ramp -----------------
class DcMotor:
    def __init__(self, name: str, cfg: Dict):
        self.name = name
        self.cfg = cfg
        self.deadband = int(cfg.get("deadband", 0))
        self.ramp = float(cfg.get("ramp_pct_per_s", 200))
        self.invert = bool(cfg.get("invert", False))
        self._target = 0.0
        self._current = 0.0
        self._stop_evt = threading.Event()
        self._thread = None
        if USE_GPIO:
            GPIO.setmode(GPIO.BCM)
            self.dir_pin = cfg["dir"]
            GPIO.setup(self.dir_pin, GPIO.OUT)
            GPIO.setup(cfg["ena"], GPIO.OUT)
            self.pwm = GPIO.PWM(cfg["ena"], cfg.get("freq", 1000))
            self.pwm.start(0)
        else:
            self.dir_pin = None
            self.pwm = None

    def _apply_dir(self, sign: int):
        if USE_GPIO:
            GPIO.output(self.dir_pin, GPIO.HIGH if sign >= 0 else GPIO.LOW)

    def _loop(self):
        tick = 0.02
        step = self.ramp * tick
        while not self._stop_evt.is_set():
            if abs(self._current - self._target) < 0.5:
                self._current = self._target
            else:
                self._current += step if self._current < self._target else -step
            duty = abs(self._current)
            sign = 1 if self._current >= 0 else -1
            if self.invert:
                sign *= -1
            self._apply_dir(sign)
            if USE_GPIO and self.pwm:
                self.pwm.ChangeDutyCycle(duty)
            else:
                print(f"[{self.name}] duty={duty:.0f}% dir={'F' if sign>=0 else 'R'}")
            time.sleep(tick)

    def _ensure(self):
        if not self._thread or not self._thread.is_alive():
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def set_speed(self, p: int):
        self._target = max(-100, min(100, float(p)))
        self._ensure()

    def forward(self):
        self.set_speed(50)

    def reverse(self):
        self.set_speed(-50)

    def stop(self):
        self._target = 0.0
        self._ensure()

    def cleanup(self):
        try:
            self._stop_evt.set()
            if USE_GPIO and self.pwm:
                self.pwm.stop()
        except Exception:
            pass

class MotorManager:
    def __init__(self):
        self.rear = DcMotor("Rear", CONFIG["rear"])
        self.steer = DcMotor("Steer", CONFIG["steer"])

    def cleanup(self):
        for m in (self.rear, self.steer):
            try:
                m.cleanup()
            except Exception:
                pass
        if USE_GPIO:
            try:
                GPIO.cleanup()
            except Exception:
                pass

# ----------------- Ultrasonic -----------------
class UltrasonicArray(QtCore.QObject):
    distances_changed = QtCore.pyqtSignal(dict)

    def __init__(self, pins: Dict[str, Dict[str, int]], buzzer_pin: Optional[int] = None):
        super().__init__()
        self.pins = pins
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._interval = 0.2
        self._enabled = USE_GPIO
        self._buzzer = buzzer_pin if (USE_GPIO and buzzer_pin is not None) else None
        if USE_GPIO:
            GPIO.setmode(GPIO.BCM)
            for pe in pins.values():
                GPIO.setup(pe["trig"], GPIO.OUT)
                GPIO.setup(pe["echo"], GPIO.IN)
                GPIO.output(pe["trig"], GPIO.LOW)
            if self._buzzer is not None:
                GPIO.setup(self._buzzer, GPIO.OUT)
                GPIO.output(self._buzzer, GPIO.LOW)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _measure(self, pe: Dict[str, int]) -> float:
        try:
            trig, echo = pe["trig"], pe["echo"]
            GPIO.output(trig, True); time.sleep(0.00001); GPIO.output(trig, False)
            start = time.time(); timeout = start + 0.02
            while GPIO.input(echo) == 0 and time.time() < timeout:
                start = time.time()
            stop_t = time.time(); timeout2 = time.time() + 0.03
            while GPIO.input(echo) == 1 and time.time() < timeout2:
                stop_t = time.time()
            dist = ((stop_t - start) * 34300) / 2.0
            return dist if 2 <= dist <= 500 else float('nan')
        except Exception:
            return float('nan')

    def _fake(self, name: str) -> float:
        t = time.time()
        base = {"front": 110, "rear": 130, "left": 90, "right": 90}.get(name, 100)
        return base + 25*np.sin(t)

    def _loop(self):
        while not self._stop.is_set():
            vals = {}
            for name, pe in self.pins.items():
                vals[name] = self._measure(pe) if self._enabled else self._fake(name)
            self.distances_changed.emit(vals)
            time.sleep(self._interval)

# ----------------- Video + YOLO with robust camera open -----------------
class VideoWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    edge_ready = QtCore.pyqtSignal(np.ndarray)
    started_cam = QtCore.pyqtSignal(bool)
    status_msg = QtCore.pyqtSignal(str)

    def __init__(self, preferred_index: int, size=(FRAME_W, FRAME_H)):
        super().__init__()
        self.size = size
        self.index_candidates = [preferred_index, 0, 1, 2, 3]
        self.backend_candidates = [
            cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0,
            cv2.CAP_ANY,
            cv2.CAP_GSTREAMER if hasattr(cv2, "CAP_GSTREAMER") else 0,
        ]
        self._stop = threading.Event()
        self._thread = None
        self._cap = None
        self._do_yolo = False
        self._every_n = 3
        self._model = None
        self._frame_count = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def reinit_with_index(self, index: int):
        self.index_candidates = [index] + [i for i in [0, 1, 2, 3] if i != index]

    def configure_yolo(self, enable: bool, every_n: int = 3):
        self._do_yolo = enable and YOLO_AVAILABLE
        self._every_n = max(1, int(every_n))
        if self._do_yolo and self._model is None and YOLO_AVAILABLE:
            try:
                self._model = YOLO("yolov8n.pt")
            except Exception as e:
                self.status_msg.emit(f"[YOLO] load failed: {e}")
                self._do_yolo = False

    def _try_open(self) -> bool:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

        w, h = self.size
        for idx in self.index_candidates:
            for backend in self.backend_candidates:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception as e:
                    self.status_msg.emit(f"Open error idx={idx} backend={backend}: {e}")
                    continue
                if not cap or not cap.isOpened():
                    continue
                # request size, then verify; fall back to 640x480
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, _ = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ok, _ = cap.read()
                if ok:
                    self._cap = cap
                    self.status_msg.emit(f"Camera OK on /dev/video{idx} (backend {backend})")
                    return True
                cap.release()
        self.status_msg.emit("No camera found (tried indexes 0..3 & common backends)")
        return False

    def _loop(self):
        ok = self._try_open()
        self.started_cam.emit(ok)
        if not ok:
            return
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self.status_msg.emit("Frame read failed — retrying open…")
                if not self._try_open():
                    break
                else:
                    continue
            self._frame_count += 1
            # Edge view (Canny)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 1.1)
            edges = cv2.Canny(blur, 80, 160)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            # YOLO overlay
            if self._do_yolo and self._model is not None and (self._frame_count % self._every_n == 0):
                try:
                    res = self._model(frame, verbose=False)[0]
                    for b in res.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        conf = float(b.conf[0]); cls = int(b.cls[0])
                        name = res.names.get(cls, str(cls))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    self.status_msg.emit(f"[YOLO] inference error: {e}")
                    self._do_yolo = False
            self.frame_ready.emit(frame)
            self.edge_ready.emit(edges_rgb)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

# ----------------- UI helpers -----------------
class Toggle(QtWidgets.QCheckBox):
    def __init__(self, label: str = ""):
        super().__init__(label)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QCheckBox { color:#e6f7ff; }
            QCheckBox::indicator { width: 50px; height: 26px; }
            QCheckBox::indicator:unchecked { border-radius:13px; background:#777; }
            QCheckBox::indicator:checked   { border-radius:13px; background:#00e676; }
            QCheckBox::indicator:unchecked:pressed { background:#666; }
            QCheckBox::indicator:checked:pressed   { background:#00c853; }
            """
        )

def colored_bar(value: float, threshold: float, max_cm: int = 400) -> Tuple[int, str]:
    v = 0 if np.isnan(value) else int(max(0, min(max_cm, value)))
    if np.isnan(value):
        style = ("QProgressBar { text-align:center; color:#0b0f19; background:#eceff1; border-radius:8px; }"
                 "QProgressBar::chunk { background:#b0bec5; border-radius:8px; }")
        return v, style
    ok = (v >= int(threshold))
    color = "#00e676" if ok else "#ff5252"
    style = ("QProgressBar { text-align:center; color:#0b0f19; background:#fafafa; border-radius:8px; }"
             f"QProgressBar::chunk {{ background:{color}; border-radius:8px; }}")
    return v, style

class CameraWidget(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(FRAME_W, FRAME_H)
        self.setScaledContents(True)
        self.setStyleSheet("background:#0b0f19; border-radius:12px;")

    def update_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))

class MiniMapWidget(QtWidgets.QWidget):
    """Circular FOV + steering angle indicator."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 260)
        self.distances = {"front": np.nan, "rear": np.nan, "left": np.nan, "right": np.nan}
        self.thresholds = {"front": 60, "rear": 60, "left": 40, "right": 40}
        self.steer_angle_deg = 0.0
        self.setToolTip("Top=Front, Bottom=Rear. Yellow line = steering angle.")

    def set_state(self, distances: Dict[str, float], thresholds: Dict[str, float], steer_angle_deg: float):
        self.distances.update(distances)
        self.thresholds.update(thresholds)
        self.steer_angle_deg = max(-60.0, min(60.0, float(steer_angle_deg)))
        self.update()

    def _ok(self, key: str) -> bool:
        d = self.distances.get(key, np.nan)
        t = self.thresholds.get(key, 0)
        return (not np.isnan(d)) and d >= t

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)
        p.fillRect(rect, QtGui.QColor("#0b0f19"))
        d = min(rect.width(), rect.height())
        cx = rect.left() + rect.width()//2
        cy = rect.top() + rect.height()//2
        r = d//2 - 6
        circle_rect = QtCore.QRect(cx - r, cy - r, 2*r, 2*r)
        pen = QtGui.QPen(QtGui.QColor("#455a64"), 3)
        p.setPen(pen)
        p.drawEllipse(circle_rect)
        # quadrants
        colors = {True: QtGui.QColor("#00e676"), False: QtGui.QColor("#ff5252")}
        p.setPen(QtCore.Qt.NoPen)
        for key, start_deg in [("front", 45*16), ("right", -45*16), ("rear", 225*16), ("left", 135*16)]:
            p.setBrush(colors[self._ok(key)])
            p.drawPie(circle_rect, start_deg, 90*16)
        # labels
        p.setPen(QtGui.QPen(QtGui.QColor("#e6f7ff")))
        p.setFont(QtGui.QFont("", 9, QtGui.QFont.Bold))
        p.drawText(cx-15, circle_rect.top()+14, "F")
        p.drawText(cx-15, circle_rect.bottom()-2, "R")
        p.drawText(circle_rect.left()+4, cy+4, "L")
        p.drawText(circle_rect.right()-16, cy+4, "R")
        # steering line
        p.setPen(QtGui.QPen(QtGui.QColor("#ffd54f"), 4))
        ang = np.deg2rad(-self.steer_angle_deg)
        x2 = cx + int(r * 0.95 * np.sin(ang))
        y2 = cy - int(r * 0.95 * np.cos(ang))
        p.drawLine(cx, cy, x2, y2)
        p.end()

# ----------------- Main Window -----------------
class ControlPanel(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capstone Projects Auto-bicycle TVS IQL – Auto Bike Control Panel")
        self.setMinimumSize(1480, 980)
        self.setStyleSheet(self._root_styles())


        # build layouts (logic same as before)
        c = QtWidgets.QWidget(); self.setCentralWidget(c)
        main = QtWidgets.QVBoxLayout(c)


        # Futuristic centered glowing titles
        title = QtWidgets.QLabel("Capstone Projects Auto-bicycle TVS IQL")
        title.setObjectName("title")
        title.setAlignment(QtCore.Qt.AlignCenter)
        subtitle = QtWidgets.QLabel("Auto Bike Control Panel")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        main.addWidget(title)
        main.addWidget(subtitle)
        # state
        self.thresholds = CONFIG["thresholds"].copy()
        self.motors = MotorManager()
        self._distances: Dict[str, float] = {k: float('nan') for k in ["front", "rear", "left", "right"]}
        self._steer_angle = 0.0

        # workers
        self.ultra = UltrasonicArray(CONFIG["ultrasonic"], CONFIG.get("buzzer_pin"))
        self.ultra.distances_changed.connect(self.on_distances)
        self.ultra.start()

        self.video = VideoWorker(CONFIG["camera_index"])
        self.video.frame_ready.connect(self.on_frame)
        self.video.edge_ready.connect(self.on_edge)
        self.video.started_cam.connect(self.on_cam_started)
        # status messages live in label
        # (label is created in _build_camera_side, so we connect after creating UI)

        # central layout
        c = QtWidgets.QWidget(); self.setCentralWidget(c)
        main = QtWidgets.QVBoxLayout(c); main.setContentsMargins(12, 12, 12, 8); main.setSpacing(10)

        title = QtWidgets.QLabel("Capstone Projects Auto-bicycle TVS IQL"); title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Auto Bike Control Panel"); subtitle.setObjectName("subtitle")
        main.addWidget(title); main.addWidget(subtitle)

        row = QtWidgets.QHBoxLayout(); row.setSpacing(12); main.addLayout(row)
        row.addWidget(self._build_motor_group(), 2)
        row.addWidget(self._build_sensor_group(), 3)

        camrow = QtWidgets.QHBoxLayout(); camrow.setSpacing(12); main.addLayout(camrow)
        cam_stack = QtWidgets.QVBoxLayout(); cam_stack.setSpacing(8)
        self.camera = CameraWidget()
        self.camera_edges = CameraWidget()
        cam_stack.addWidget(self.camera)
        cam_stack.addWidget(self.camera_edges)
        camrow.addLayout(cam_stack, 3)
        rightcol = QtWidgets.QVBoxLayout(); rightcol.setSpacing(8)
        rightcol.addWidget(self._build_camera_side())
        self.minimap = MiniMapWidget(); rightcol.addWidget(self.minimap)
        camrow.addLayout(rightcol, 1)

        footer = QtWidgets.QHBoxLayout(); footer.setContentsMargins(0, 8, 0, 0)
        footer.addWidget(QtWidgets.QLabel("Team:"))
        self.team_edit = QtWidgets.QLineEdit(); self.team_edit.setPlaceholderText("Add team names…")
        footer.addWidget(self.team_edit)
        main.addLayout(footer)

        # after UI is ready, connect video status label and start
        self.video.status_msg.connect(lambda s: self.lbl_cam.setText(f"Camera: {s}"))
        self.video.start()

        # UI refresh
        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.timeout.connect(self._refresh_sensor_bars)
        self._ui_timer.start(120)

    # -------------- Groups --------------
    def _build_motor_group(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Motors")
        grid = QtWidgets.QGridLayout(gb); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(8)

        # Rear controls
        grid.addWidget(self._section_label("Rear Motor"), 0, 0, 1, 4)
        self.rear_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.rear_speed.setRange(-100, 100); self.rear_speed.setValue(0)
        grid.addWidget(QtWidgets.QLabel("Speed"), 1, 0); grid.addWidget(self.rear_speed, 1, 1, 1, 3)
        pad = QtWidgets.QHBoxLayout()
        self.btn_rear_rev = QtWidgets.QToolButton(); self.btn_rear_rev.setText("◀ Reverse")
        self.btn_rear_fwd = QtWidgets.QToolButton(); self.btn_rear_fwd.setText("Forward ▶")
        self.btn_rear_stop = QtWidgets.QToolButton(); self.btn_rear_stop.setText("⏹ Stop")
        for b in (self.btn_rear_rev, self.btn_rear_fwd, self.btn_rear_stop): b.setProperty("kind", "pad")
        pad.addWidget(self.btn_rear_rev); pad.addWidget(self.btn_rear_fwd); pad.addWidget(self.btn_rear_stop)
        grid.addLayout(pad, 2, 0, 1, 4)

        # Steering
        grid.addWidget(self._section_label("Steering Motor"), 3, 0, 1, 4)
        self.steer_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.steer_speed.setRange(0, 100); self.steer_speed.setValue(0)
        grid.addWidget(QtWidgets.QLabel("Strength"), 4, 0); grid.addWidget(self.steer_speed, 4, 1, 1, 3)
        spad = QtWidgets.QHBoxLayout()
        self.btn_left = QtWidgets.QToolButton(); self.btn_left.setText("⬅ Left")
        self.btn_right = QtWidgets.QToolButton(); self.btn_right.setText("Right ➡")
        self.btn_center = QtWidgets.QToolButton(); self.btn_center.setText("⏹ Center")
        for b in (self.btn_left, self.btn_right, self.btn_center): b.setProperty("kind", "pad")
        spad.addWidget(self.btn_left); spad.addWidget(self.btn_right); spad.addWidget(self.btn_center)
        grid.addLayout(spad, 5, 0, 1, 4)

        # wiring
        self.rear_speed.valueChanged.connect(lambda v: self.motors.rear.set_speed(v))
        self.steer_speed.valueChanged.connect(lambda v: self.motors.steer.set_speed(v))
        self.btn_rear_fwd.clicked.connect(lambda: self.motors.rear.set_speed(max(40, abs(self.rear_speed.value()))))
        self.btn_rear_rev.clicked.connect(lambda: self.motors.rear.set_speed(-max(40, abs(self.rear_speed.value()))))
        self.btn_rear_stop.clicked.connect(self.motors.rear.stop)
        self.btn_left.clicked.connect(lambda: self._nudge_steer(-max(35, self.steer_speed.value())))
        self.btn_right.clicked.connect(lambda: self._nudge_steer(max(35, self.steer_speed.value())))
        self.btn_center.clicked.connect(self._center_steer)
        return gb

    def _build_sensor_group(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Obstacle Sensors")
        grid = QtWidgets.QGridLayout(gb); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)

        self.alert = QtWidgets.QLabel("All Clear"); self.alert.setObjectName("alert")
        grid.addWidget(self.alert, 0, 0, 1, 4)

        self._bars: Dict[str, QtWidgets.QProgressBar] = {}
        self._spins: Dict[str, QtWidgets.QSpinBox] = {}
        row = 1
        for key, label in zip(["front", "rear", "left", "right"], ["Front", "Rear", "Left", "Right"]):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            bar = QtWidgets.QProgressBar(); bar.setRange(0, 400); bar.setFormat("%v cm"); bar.setFixedWidth(280)
            self._bars[key] = bar; grid.addWidget(bar, row, 1)
            spin = QtWidgets.QSpinBox(); spin.setRange(5, 400); spin.setValue(self.thresholds[key])
            spin.valueChanged.connect(lambda v, n=key: self._set_threshold(n, v))
            self._spins[key] = spin
            grid.addWidget(QtWidgets.QLabel("Threshold"), row, 2); grid.addWidget(spin, row, 3)
            row += 1
        return gb

    def _build_camera_side(self) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox("Camera  Detection  Control Mode")
        form = QtWidgets.QFormLayout(gb)

        # Camera picker
        self.combo_cam = QtWidgets.QComboBox(); self.combo_cam.addItems([f"/dev/video{i}" for i in range(0, 4)])
        self.btn_rescan = QtWidgets.QPushButton("Rescan")
        form.addRow("Device", self.combo_cam)
        form.addRow(self.btn_rescan)

        # YOLO controls
        self.toggle_yolo = Toggle("Enable YOLO"); self.toggle_yolo.setChecked(False)
        self.spin_every = QtWidgets.QSpinBox(); self.spin_every.setRange(1, 10); self.spin_every.setValue(3)
        form.addRow(self.toggle_yolo)
        form.addRow("Run YOLO every N frames", self.spin_every)

        # Status label
        self.lbl_cam = QtWidgets.QLabel("Camera: …")
        form.addRow(self.lbl_cam)

        # Control mode radios
        self.radio_gui = QtWidgets.QRadioButton("Use GUI buttons")
        self.radio_kbd = QtWidgets.QRadioButton("Use keyboard (arrows/H/L)")
        self.radio_gui.setChecked(True)
        form.addRow(self.radio_gui)
        form.addRow(self.radio_kbd)

        # Wire-up
        self.toggle_yolo.stateChanged.connect(self._apply_yolo)
        self.spin_every.valueChanged.connect(self._apply_yolo)
        self.radio_gui.toggled.connect(self._apply_control_mode)
        self.radio_kbd.toggled.connect(self._apply_control_mode)
        self.combo_cam.currentIndexChanged.connect(self._change_camera)
        self.btn_rescan.clicked.connect(self._rescan_cameras)
        return gb

    # -------------- Events & helpers --------------
    def _set_threshold(self, name: str, v: int):
        self.thresholds[name] = int(v)

    def _nudge_steer(self, val: int):
        # update angle estimate and motor
        self._steer_angle = max(-45.0, min(45.0, self._steer_angle + (2.5 if val > 0 else -2.5)))
        self.motors.steer.set_speed(val)
        self.minimap.set_state(self._distances, self.thresholds, self._steer_angle)

    def _center_steer(self):
        self._steer_angle = 0.0
        self.motors.steer.stop()
        self.minimap.set_state(self._distances, self.thresholds, self._steer_angle)

    def _rescan_cameras(self):
        found = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0)
            ok = cap.isOpened()
            try: cap.release()
            except Exception: pass
            if ok:
                found.append(i)
        self.combo_cam.clear()
        if not found:
            self.combo_cam.addItem("No cameras")
            self.lbl_cam.setText("Camera: Not Found")
        else:
            for i in found:
                self.combo_cam.addItem(f"/dev/video{i}", i)
            self.lbl_cam.setText(f"Camera: Found {len(found)} device(s)")

    def _change_camera(self, idx: int):
        data = self.combo_cam.currentData()
        if data is None:
            return
        self.video.reinit_with_index(int(data))
        self.lbl_cam.setText(f"Camera: switching to /dev/video{int(data)}…")

    def on_cam_started(self, ok: bool):
        self.lbl_cam.setText("Camera: OK" if ok else "Camera: Not Found")

    def on_frame(self, frame: np.ndarray):
        self.camera.update_frame(frame)

    def on_edge(self, edge_rgb: np.ndarray):
        h, w, _ = edge_rgb.shape
        qimg = QtGui.QImage(edge_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        self.camera_edges.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def on_distances(self, vals: Dict[str, float]):
        self._distances.update(vals)
        self.minimap.set_state(self._distances, self.thresholds, self._steer_angle)

    def _refresh_sensor_bars(self):
        any_bad = False
        for k in ["front", "rear", "left", "right"]:
            dist = self._distances.get(k, float('nan'))
            thr = self.thresholds.get(k, 50)
            v, style = colored_bar(dist, thr)
            bar = self._bars[k]
            bar.setValue(v); bar.setStyleSheet(style)
            if not np.isnan(dist) and dist < thr:
                any_bad = True
        self.alert.setText("⚠ Obstacle within threshold" if any_bad else "All Clear")
        self.alert.setProperty("danger", any_bad)
        self.alert.style().unpolish(self.alert); self.alert.style().polish(self.alert)
        self.minimap.set_state(self._distances, self.thresholds, self._steer_angle)

    def _apply_yolo(self):
        enable = self.toggle_yolo.isChecked(); every = self.spin_every.value()
        self.video.configure_yolo(enable, every)
        if enable and not YOLO_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "YOLO not available", "Install 'ultralytics' to enable detection.")

    def _apply_control_mode(self):
        using_kbd = self.radio_kbd.isChecked()
        # enable/disable button pads for clarity
        for btn in [self.btn_rear_fwd, self.btn_rear_rev, self.btn_rear_stop, self.btn_left, self.btn_right, self.btn_center]:
            btn.setEnabled(not using_kbd)
        # focus for key events
        if using_kbd:
            self.setFocus()
        else:
            self.clearFocus()

    # Keyboard control
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if not self.radio_kbd.isChecked():
            return super().keyPressEvent(event)
        key = event.key()
        step = 5
        if key == QtCore.Qt.Key_Up:
            val = max(40, abs(self.rear_speed.value()))
            self.rear_speed.setValue(val); self.motors.rear.set_speed(val)
        elif key == QtCore.Qt.Key_Down:
            val = -max(40, abs(self.rear_speed.value()))
            self.rear_speed.setValue(val); self.motors.rear.set_speed(val)
        elif key == QtCore.Qt.Key_Left:
            self._nudge_steer(-max(35, self.steer_speed.value()))
        elif key == QtCore.Qt.Key_Right:
            self._nudge_steer(max(35, self.steer_speed.value()))
        elif key == QtCore.Qt.Key_Space:
            self.motors.rear.stop(); self._center_steer()
        elif key == QtCore.Qt.Key_H:
            self.rear_speed.setValue(min(100, self.rear_speed.value() + step))
        elif key == QtCore.Qt.Key_L:
            self.rear_speed.setValue(max(-100, self.rear_speed.value() - step))
        else:
            return super().keyPressEvent(event)

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text); lbl.setObjectName("section"); return lbl

    def _root_styles(self) -> str:
        return (
            """
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f2027, stop:0.5 #203a43, stop:1 #2c5364); }


            QLabel#title {
            font-size:34px; font-weight:900; color:#00e5ff;
            text-align:center;
            text-shadow: 0px 0px 12px #00e5ff;
            }
            QLabel#subtitle {
            font-size:22px; font-weight:600; color:#80deea;
            text-align:center;
            text-shadow: 0px 0px 8px #80deea;
            margin-bottom:12px;
            }


            QLabel#section { font-size:16px; font-weight:700; color:#ffd54f; }


            QGroupBox {
            border:2px solid #3949ab;
            border-radius:16px;
            margin-top:20px;
            background:rgba(255,255,255,0.03);
            color:#e6f7ff;
            box-shadow: 0px 0px 20px rgba(0,229,255,0.4);
            }
            QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding:0 10px;
            color:#00e5ff;
            font-weight:bold;
            }


            QToolButton[kind="pad"] {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #00e5ff, stop:1 #2979ff);
            color:white; border:none; border-radius:14px; padding:12px 16px; font-weight:700;
            box-shadow: 0px 0px 15px #00e5ff;
            }
            QToolButton[kind="pad"]:hover { background:#1565c0; }
            QToolButton[kind="pad"]:pressed { background:#0d47a1; }


            QSlider::groove:horizontal {
            height:10px;
            background:#263238;
            border-radius:5px;
            box-shadow: inset 0px 0px 5px #000;
            }
            QSlider::handle:horizontal {
            background:#00e676;
            width:22px; margin:-6px 0;
            border-radius:11px;
            box-shadow:0px 0px 8px #00e676;
            }


            QSpinBox, QLineEdit {
            background:#102027; color:#e6f7ff;
            border:1px solid #37474f; border-radius:8px; padding:6px;
            }


            QProgressBar { height:26px; border-radius:8px; background:#263238; }
            QProgressBar::chunk { border-radius:8px; }


            QLabel#alert { font-size:16px; font-weight:800; padding:10px; border-radius:12px; background:#1b5e20; color:#e8f5e9; }
            QLabel#alert[danger="true"] { background:#b71c1c; color:#ffebee; }
            """
        )

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self.ultra.stop(); self.video.stop(); self.motors.cleanup()
        finally:
            super().closeEvent(e)

# ----------------- Entry -----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Auto Bike Control Panel")
    w = ControlPanel(); w.show()
    sys.exit(app.exec_())
