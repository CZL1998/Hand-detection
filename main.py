"""
Real-time letter recognition via air drawing.

Setup (one-time):
    pip install -r requirements.txt
    python train_model.py

Controls:
    ☝  Index finger only  →  draw letter in air
    ✊  Fist               →  clear recognised text
    Q                     →  quit
"""
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import sys
import time

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

MODEL_PATH  = "letter_model.keras"
PAUSE_SECS      = 1.5    # seconds of stillness to trigger recognition
STROKE_GAP_SECS = 0.5    # max gap between strokes to keep the same letter alive
MOVE_THRESH     = 0.012  # normalised movement below this = "still"
MIN_POINTS  = 20     # minimum trajectory points required to attempt recognition
TRAIL_THICK = 12     # drawing stroke width (pixels on full-res canvas)
TRAIL_COLOR = (255, 255, 255)
CONF_THRESH = 0.45   # minimum CNN confidence to accept a letter
MAX_LETTERS = 24     # maximum letters shown in result bar

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Model ────────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] {MODEL_PATH} not found.")
        print("Please run:  python train_model.py")
        sys.exit(1)
    print(f"Loading model from {MODEL_PATH} ...")
    m = tf.keras.models.load_model(MODEL_PATH)
    print("Model ready.\n")
    return m


# ── Gesture helpers ───────────────────────────────────────────────────────

def only_index_up(lm):
    """True when only the index finger is extended (drawing mode)."""
    return (
        lm[8].y  < lm[6].y and   # index tip above PIP
        lm[12].y > lm[10].y and  # middle curled
        lm[16].y > lm[14].y and  # ring curled
        lm[20].y > lm[18].y      # pinky curled
    )


def is_fist(lm):
    """True for a fully closed fist (clear gesture)."""
    return (
        lm[8].y  > lm[6].y and
        lm[12].y > lm[10].y and
        lm[16].y > lm[14].y and
        lm[20].y > lm[18].y
    )


# ── Recognition ──────────────────────────────────────────────────────────

def _letterbox(img, size=28):
    """Resize a grayscale image to size×size preserving aspect ratio."""
    h, w = img.shape
    scale = size / max(h, w, 1)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def recognise(draw_canvas, trajectory, model):
    """Crop the drawn region from canvas and run CNN inference."""
    if len(trajectory) < MIN_POINTS:
        return None, 0.0

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    pad = 20
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(draw_canvas.shape[1], max(xs) + pad)
    y2 = min(draw_canvas.shape[0], max(ys) + pad)

    roi = draw_canvas[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return None, 0.0

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img28 = _letterbox(gray, 28)

    # Slight dilation to better match EMNIST stroke width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img28  = cv2.dilate(img28, kernel)

    inp  = img28.astype(np.float32) / 255.0
    inp  = inp.reshape(1, 28, 28, 1)
    pred = model.predict(inp, verbose=0)[0]
    idx  = int(np.argmax(pred))
    conf = float(pred[idx])
    return chr(ord("A") + idx), conf


# ── Drawing helpers ───────────────────────────────────────────────────────

def put(frame, text, x, y, scale=0.75, color=(200, 200, 200), thick=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_result_bar(frame, letters, h, w):
    """Semi-transparent bottom bar with recognised letters."""
    bar_h   = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    put(frame, "Recognised:", 12, h - bar_h + 22,
        scale=0.6, color=(120, 120, 120), thick=1)
    display = letters if letters else "—"
    cv2.putText(frame, display, (12, h - 18),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 2, cv2.LINE_AA)


def reset_drawing(canvas, trajectory):
    canvas[:] = 0
    trajectory.clear()


# ── Main loop ─────────────────────────────────────────────────────────────

def run():
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame0 = cap.read()
    h, w = frame0.shape[:2] if ok else (720, 1280)

    canvas     = np.zeros((h, w, 3), dtype=np.uint8)
    trajectory = []
    letters    = ""

    last_norm      = None   # normalised (x, y) of fingertip last frame
    still_since    = None   # timestamp when movement dropped below threshold
    last_draw_time = None   # timestamp when drawing mode was last active

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res      = hands.process(rgb)
            rgb.flags.writeable = True

            tip_px  = None
            drawing = False

            if res.multi_hand_landmarks:
                hand_lm = res.multi_hand_landmarks[0]
                lm      = hand_lm.landmark

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Always compute index tip position for cursor display
                tip_px = (int(lm[8].x * w), int(lm[8].y * h))

                if only_index_up(lm):
                    drawing        = True
                    last_draw_time = time.time()
                    tip_norm       = (lm[8].x, lm[8].y)

                    # Draw stroke on canvas — but don't connect across a gap
                    if last_norm is not None:
                        prev_px = (int(last_norm[0] * w), int(last_norm[1] * h))
                        cv2.line(canvas, prev_px, tip_px, TRAIL_COLOR, TRAIL_THICK)
                    cv2.circle(canvas, tip_px, TRAIL_THICK // 2, TRAIL_COLOR, -1)
                    trajectory.append(tip_px)

                    # Movement detection
                    if last_norm is not None:
                        dx = tip_norm[0] - last_norm[0]
                        dy = tip_norm[1] - last_norm[1]
                        moved = (dx * dx + dy * dy) ** 0.5
                        if moved < MOVE_THRESH:
                            if still_since is None:
                                still_since = time.time()
                        else:
                            still_since = None

                    last_norm = tip_norm

                    # Trigger recognition after pause
                    if still_since and time.time() - still_since >= PAUSE_SECS:
                        letter, conf = recognise(canvas, trajectory, model)
                        if letter and conf >= CONF_THRESH:
                            letters = (letters + letter)[-MAX_LETTERS:]
                            print(f"  → {letter}  ({conf:.0%})")
                        reset_drawing(canvas, trajectory)
                        last_draw_time = None
                        last_norm      = None
                        still_since    = None

                elif is_fist(lm):
                    # Fist always clears, even within gap
                    letters = ""
                    reset_drawing(canvas, trajectory)
                    last_draw_time = None
                    last_norm      = None
                    still_since    = None

                else:
                    # Pen lifted — check stroke gap window
                    in_gap = (last_draw_time is not None and
                              time.time() - last_draw_time <= STROKE_GAP_SECS)
                    if in_gap:
                        # User is repositioning between strokes — keep canvas alive
                        last_norm   = None   # don't connect to old position on resume
                        still_since = None   # pause timer paused while pen is up
                    else:
                        # Gap exceeded: commit whatever is drawn
                        if len(trajectory) >= MIN_POINTS:
                            letter, conf = recognise(canvas, trajectory, model)
                            if letter and conf >= CONF_THRESH:
                                letters = (letters + letter)[-MAX_LETTERS:]
                        reset_drawing(canvas, trajectory)
                        last_draw_time = None
                        last_norm      = None
                        still_since    = None

            else:
                # No hand detected
                in_gap = (last_draw_time is not None and
                          time.time() - last_draw_time <= STROKE_GAP_SECS)
                if not in_gap:
                    reset_drawing(canvas, trajectory)
                    last_draw_time = None
                last_norm   = None
                still_since = None
                tip_px      = None

            # ── Composite drawing canvas onto frame ──────────────────────
            drawn_mask = canvas.sum(axis=2) > 0
            blended    = cv2.addWeighted(frame, 0.15, canvas, 0.85, 0)
            frame[drawn_mask] = blended[drawn_mask]

            # ── Cursor dot + pause progress ring ─────────────────────────
            if tip_px:
                cv2.circle(frame, tip_px, 12, (0, 180, 255), 2, cv2.LINE_AA)
                if still_since:
                    prog = min((time.time() - still_since) / PAUSE_SECS, 1.0)
                    cv2.ellipse(frame, tip_px, (18, 18),
                                -90, 0, int(360 * prog),
                                (0, 255, 100), 3, cv2.LINE_AA)

            # ── Status hint ───────────────────────────────────────────────
            pen_up_in_gap = (not drawing and last_draw_time is not None and
                             time.time() - last_draw_time <= STROKE_GAP_SECS)
            if drawing:
                hint     = "Drawing... (pause 1.5 s to confirm)"
                hint_col = (80, 255, 80)
            elif pen_up_in_gap:
                hint     = "Pen up — extend index to continue stroke"
                hint_col = (0, 200, 255)
            elif res.multi_hand_landmarks:
                hint     = "Extend index finger to draw"
                hint_col = (200, 200, 200)
            else:
                hint     = "No hand detected"
                hint_col = (100, 100, 100)

            put(frame, hint, 10, 32, color=hint_col)
            put(frame, "Fist = Clear  |  Q = Quit",
                10, 58, scale=0.60, color=(130, 130, 130), thick=1)

            draw_result_bar(frame, letters, h, w)

            cv2.imshow("Hand Detection - Air Writing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
