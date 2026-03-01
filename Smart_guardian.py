"""
Smart Guardian System v2 — Improved Accident & Fight Detection
Fixes:
  - Fast-moving vehicle detection (lower confidence + optical flow velocity)
  - Accident detection via sudden deceleration / trajectory overlap
  - Relaxed stationary threshold for persons post-collision
  - Motion blur pre-processing for clearer detections
  - Better fight clustering
"""

import cv2
import math
import time
import os
import sys
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import io
from IPython.display import display
from IPython.display import Image as IPImage
from PIL import Image as PILImage
from google.colab import files

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  —  tune these to your video
# ══════════════════════════════════════════════════════════════

VIDEO_PATH           = "accident.mp4"   # overridden below from upload

# Detection
CONF_PERSON          = 0.35   # lower = catch more persons (incl. fallen)
CONF_VEHICLE         = 0.25   # LOW on purpose — catch fast/blurry vehicles
CONF_VEHICLE_STILL   = 0.40   # stricter when vehicle is slow (reduces false positives)
IOU_NMS              = 0.50   # non-max suppression overlap threshold

# Accident triggers
VELOCITY_SPIKE_PX    = 55     # vehicle centroid moved >55px in one frame → "fast"
SUDDEN_STOP_FRAMES   = 6      # was fast for N frames then speed drops >70% → "sudden stop"
OVERLAP_IOU_THRESH   = 0.08   # bounding box IoU vehicle↔person to flag proximity collision
PERSON_DOWN_RATIO    = 1.10   # person box width/height > this → likely lying down
STATIONARY_FRAMES    = 30     # frames person barely moves after a fast vehicle nearby
STATIONARY_DIST_PX   = 40     # px — relaxed (victim may twitch)
HISTORY_LEN          = 20     # frames of velocity history per tracked object

# Fight triggers
FIGHT_PERSON_COUNT   = 3
FIGHT_PROXIMITY_PX   = 130

# Alert
ALERT_COOLDOWN_SEC   = 7.0
BANNER_FRAMES        = 90     # how long banner stays on screen

# Display
DISPLAY_EVERY_N      = 2      # skip frames for Colab preview speed

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
PERSON_CLASS    = "person"


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def box_iou(b1, b2):
    """Intersection-over-Union of two xyxy boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter)

def aspect_ratio(box):
    """width / height"""
    x1,y1,x2,y2 = box
    h = max(1, y2-y1)
    return (x2-x1) / h


# ══════════════════════════════════════════════════════════════
#  PRE-PROCESSING  — reduce blur before inference
# ══════════════════════════════════════════════════════════════

def preprocess(frame):
    """
    Sharpen the frame to help detect motion-blurred vehicles.
    Uses an unsharp mask approach.
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    sharp   = cv2.addWeighted(frame, 1.6, blurred, -0.6, 0)
    return sharp


# ══════════════════════════════════════════════════════════════
#  OPTICAL FLOW — per-region motion magnitude
# ══════════════════════════════════════════════════════════════

class FlowAnalyzer:
    """
    Computes dense optical flow between consecutive frames
    and returns the mean flow magnitude inside a bounding box.
    """
    def __init__(self):
        self.prev_gray = None

    def update(self, gray):
        flow = None
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=13,
                iterations=3, poly_n=5, poly_sigma=1.1,
                flags=0
            )
        self.prev_gray = gray
        return flow

    @staticmethod
    def region_magnitude(flow, box, frame_shape):
        if flow is None:
            return 0.0
        h, w = frame_shape[:2]
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = flow[y1:y2, x1:x2]
        mag = np.sqrt(region[...,0]**2 + region[...,1]**2)
        return float(np.mean(mag))


# ══════════════════════════════════════════════════════════════
#  OBJECT TRACKER  (centroid + velocity history)
# ══════════════════════════════════════════════════════════════

class ObjectTracker:
    """
    Tracks any set of bounding boxes across frames.
    Keeps a short velocity history per tracked id.
    """
    def __init__(self, max_distance=160, history=HISTORY_LEN):
        self.next_id    = 0
        self.centroids  = {}          # id → (cx,cy)
        self.boxes      = {}          # id → xyxy
        self.vel_hist   = defaultdict(lambda: deque(maxlen=history))
        self.still_cnt  = defaultdict(int)

    def update(self, new_boxes):
        if not new_boxes:
            self.centroids.clear()
            self.boxes.clear()
            return {}

        new_cents = [box_center(b) for b in new_boxes]

        if not self.centroids:
            for i, (c, b) in enumerate(zip(new_cents, new_boxes)):
                self._register(c, b)
            return dict(self.centroids)

        matched, used = {}, set()
        for pid, oc in list(self.centroids.items()):
            best_d, best_j = float("inf"), -1
            for j, nc in enumerate(new_cents):
                if j in used: continue
                d = euclidean(oc, nc)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d < 160:
                matched[pid] = (new_cents[best_j], new_boxes[best_j])
                used.add(best_j)

        for pid, (nc, nb) in matched.items():
            speed = euclidean(self.centroids[pid], nc)
            self.vel_hist[pid].append(speed)
            if speed < STATIONARY_DIST_PX:
                self.still_cnt[pid] += 1
            else:
                self.still_cnt[pid] = 0
            self.centroids[pid] = nc
            self.boxes[pid]     = nb

        for pid in set(self.centroids) - set(matched):
            del self.centroids[pid]
            del self.boxes[pid]
            self.vel_hist.pop(pid, None)
            self.still_cnt.pop(pid, None)

        for j, (nc, nb) in enumerate(zip(new_cents, new_boxes)):
            if j not in used:
                self._register(nc, nb)

        return dict(self.centroids)

    def _register(self, c, b):
        self.centroids[self.next_id] = c
        self.boxes[self.next_id]     = b
        self.next_id += 1

    def avg_speed(self, pid):
        h = self.vel_hist.get(pid)
        return float(np.mean(h)) if h else 0.0

    def recent_max_speed(self, pid, n=SUDDEN_STOP_FRAMES):
        h = self.vel_hist.get(pid)
        if not h or len(h) < 2: return 0.0
        recent = list(h)[-n:]
        return max(recent)

    def speed_drop(self, pid):
        """True if vehicle was fast recently then suddenly slowed."""
        h = self.vel_hist.get(pid)
        if not h or len(h) < SUDDEN_STOP_FRAMES + 2: return False
        arr    = list(h)
        recent = arr[-SUDDEN_STOP_FRAMES:]
        before = arr[-SUDDEN_STOP_FRAMES*2:-SUDDEN_STOP_FRAMES]
        if not before: return False
        avg_before = max(before)
        avg_recent = np.mean(recent)
        return (avg_before > VELOCITY_SPIKE_PX and
                avg_recent  < avg_before * 0.30)   # dropped >70%

    def is_stationary(self, pid):
        return self.still_cnt.get(pid, 0) >= STATIONARY_FRAMES

    def is_lying_down(self, pid):
        b = self.boxes.get(pid)
        if b is None: return False
        return aspect_ratio(b) > PERSON_DOWN_RATIO


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def draw_box(frame, box, label, color, thickness=2):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(frame, (x1, max(y1-th-8,0)), (x1+tw+4, max(y1,th+8)), color, -1)
    cv2.putText(frame, label, (x1+2, max(y1-4, th+4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)

def draw_banner(frame, text, color=(0,0,200)):
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = max(1.0, w/700)
    thick = max(2, int(scale*2))
    ov    = frame.copy()
    cv2.rectangle(ov, (0, h//2-65), (w, h//2+65), (10,10,10), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    (tw,th),_ = cv2.getTextSize(text, font, scale, thick)
    tx = (w-tw)//2; ty = h//2+th//2
    cv2.putText(frame, text, (tx+3,ty+3), font, scale, (0,0,0), thick+2)
    cv2.putText(frame, text, (tx,  ty),   font, scale, color,   thick)

def draw_velocity_arrow(frame, center, vel_history, color):
    """Draw a small motion arrow on tracked objects."""
    if len(vel_history) < 2: return
    speed = vel_history[-1]
    if speed < 5: return
    cx, cy = int(center[0]), int(center[1])
    cv2.arrowedLine(frame, (cx,cy), (cx, cy-min(int(speed),40)),
                    color, 2, tipLength=0.4)


# ══════════════════════════════════════════════════════════════
#  COLAB DISPLAY
# ══════════════════════════════════════════════════════════════

disp_handle = display(display_id=True)

def show_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PILImage.fromarray(rgb)
    img.thumbnail((960, 540))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    disp_handle.update(IPImage(data=buf.getvalue()))


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════

def run(video_path):
    print("[INFO] Loading YOLOv8s model …  (using 's' for better accuracy)")
    # yolov8s > yolov8n for small/blurry objects; still real-time on GPU
    model = YOLO("yolov8s.pt")
    names = model.names

    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {W}×{H} @ {fps:.1f}fps | {total} frames\n")

    out_path = "output_guardian_v2.mp4"
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W, H))

    person_tracker  = ObjectTracker()
    vehicle_tracker = ObjectTracker()
    flow_analyzer   = FlowAnalyzer()

    last_accident   = 0.0
    last_fight      = 0.0
    acc_banner      = 0
    fight_banner    = 0
    event_log       = []
    frame_idx       = 0

    while True:
        ret, raw = cap.read()
        if not ret: break
        frame_idx += 1
        now   = time.time()

        # ── Pre-process for blur ─────────────────────────────────────────
        frame_sharp = preprocess(raw)
        gray        = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        flow        = flow_analyzer.update(gray)

        # ── Dual-pass inference ──────────────────────────────────────────
        # Pass 1: standard conf for persons & slow vehicles
        r1 = model(frame_sharp, conf=CONF_VEHICLE_STILL,
                   iou=IOU_NMS, verbose=False)[0]

        # Pass 2: very low conf on SHARPENED frame to catch fast vehicles
        r2 = model(frame_sharp, conf=CONF_VEHICLE,
                   iou=IOU_NMS, verbose=False)[0]

        person_boxes  = []
        vehicle_boxes = []
        seen_vboxes   = []   # dedup

        def collect(results, low_conf_pass=False):
            for box in results.boxes:
                cname = names[int(box.cls[0])]
                conf  = float(box.conf[0])
                xyxy  = box.xyxy[0].tolist()

                if cname == PERSON_CLASS and conf >= CONF_PERSON:
                    person_boxes.append(xyxy)

                elif cname in VEHICLE_CLASSES:
                    # On low-conf pass only keep if optical flow says it's fast
                    if low_conf_pass:
                        flow_mag = FlowAnalyzer.region_magnitude(flow, xyxy, raw.shape)
                        if flow_mag < 4.0:   # not really moving → skip
                            continue
                    # Dedup (IoU with already-collected vehicles)
                    is_dup = any(box_iou(xyxy, vb) > 0.40 for vb in seen_vboxes)
                    if not is_dup:
                        vehicle_boxes.append(xyxy)
                        seen_vboxes.append(xyxy)

        collect(r1, low_conf_pass=False)
        collect(r2, low_conf_pass=True)   # adds fast/blurry vehicles missed above

        # ── Update trackers ──────────────────────────────────────────────
        person_tracker.update(person_boxes)
        vehicle_tracker.update(vehicle_boxes)

        # ── ACCIDENT DETECTION ───────────────────────────────────────────
        #
        # Multi-signal approach — any of these raise a flag:
        #   A) Vehicle near person (IoU overlap) + person stationary/down
        #   B) Vehicle had sudden deceleration (crash stop)
        #   C) Person is lying down near a vehicle
        #
        accident_flag   = False
        accident_reason = ""

        for vbox in vehicle_boxes:
            v_flow  = FlowAnalyzer.region_magnitude(flow, vbox, raw.shape)
            v_speed = max(v_flow, 0)

            for pbox in person_boxes:
                overlap = box_iou(vbox, pbox)
                prox    = euclidean(box_center(vbox), box_center(pbox))

                # Signal A: direct overlap or very close
                if overlap > OVERLAP_IOU_THRESH or prox < 80:
                    accident_flag   = True
                    accident_reason = "vehicle-person overlap"
                    break

                # Signal C: lying-down person within 200px of vehicle
                ar = aspect_ratio(pbox)
                if ar > PERSON_DOWN_RATIO and prox < 200:
                    accident_flag   = True
                    accident_reason = f"person lying down near vehicle (AR={ar:.2f})"
                    break

            if accident_flag: break

        # Signal B: sudden stop of any tracked vehicle
        for pid in list(vehicle_tracker.vel_hist.keys()):
            if vehicle_tracker.speed_drop(pid):
                accident_flag   = True
                accident_reason = f"vehicle sudden deceleration (id={pid})"
                break

        # Also require: at least one person detected in frame
        if accident_flag and not person_boxes:
            accident_flag = False

        if accident_flag and (now - last_accident) > ALERT_COOLDOWN_SEC:
            last_accident = now
            acc_banner    = BANNER_FRAMES
            msg = f"[Frame {frame_idx}] 🚨 ACCIDENT DETECTED ({accident_reason}) → Calling Ambulance..."
            print(msg)
            event_log.append(msg)

        # ── FIGHT DETECTION ──────────────────────────────────────────────
        if len(person_boxes) >= FIGHT_PERSON_COUNT:
            cents = [box_center(b) for b in person_boxes]
            close = sum(
                1 for i, c in enumerate(cents)
                if any(euclidean(c, cents[j]) < FIGHT_PROXIMITY_PX
                       for j in range(len(cents)) if j != i)
            )
            if close >= FIGHT_PERSON_COUNT and (now - last_fight) > ALERT_COOLDOWN_SEC:
                last_fight   = now
                fight_banner = BANNER_FRAMES
                msg = f"[Frame {frame_idx}] 🚔 FIGHT DETECTED ({close} persons close) → Calling Police..."
                print(msg)
                event_log.append(msg)

        # ── DRAW BOXES ───────────────────────────────────────────────────
        for pid, vbox in vehicle_tracker.boxes.items():
            spd   = vehicle_tracker.avg_speed(pid)
            is_fast = vehicle_tracker.recent_max_speed(pid) > VELOCITY_SPIKE_PX
            label = f"Vehicle  v={spd:.0f}px" + (" ⚡" if is_fast else "")
            color = (0, 60, 255) if is_fast else (255, 140, 0)
            draw_box(raw, vbox, label, color, thickness=3 if is_fast else 2)
            draw_velocity_arrow(raw, vehicle_tracker.centroids[pid],
                                list(vehicle_tracker.vel_hist[pid]), color)

        for pid, pbox in person_tracker.boxes.items():
            is_down = person_tracker.is_lying_down(pid)
            is_still= person_tracker.is_stationary(pid)
            status  = " 🔴DOWN" if is_down else (" STILL" if is_still else "")
            color   = (0, 0, 255) if (is_down or is_still) else (0, 230, 80)
            draw_box(raw, pbox, f"Person{status}", color)

        # ── BANNERS ──────────────────────────────────────────────────────
        if acc_banner > 0:
            draw_banner(raw, "  ACCIDENT DETECTED  ", (0, 0, 220))
            acc_banner -= 1
        if fight_banner > 0:
            draw_banner(raw, "  FIGHT DETECTED  ", (30, 30, 220))
            fight_banner -= 1

        # ── HUD ──────────────────────────────────────────────────────────
        hud = (f"Frame {frame_idx}/{total}  "
               f"Persons:{len(person_boxes)}  "
               f"Vehicles:{len(vehicle_boxes)}  "
               f"| SmartGuardian v2")
        cv2.putText(raw, hud, (10,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220,220,220), 2)

        writer.write(raw)

        if frame_idx % DISPLAY_EVERY_N == 0:
            show_frame(raw)

    cap.release()
    writer.release()
    print(f"\n✅ Done — {frame_idx} frames processed.")
    print(f"📹 Output: {out_path}")
    return out_path, event_log


# ══════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    vpath = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    run(vpath)