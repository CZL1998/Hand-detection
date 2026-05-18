"""
Collect training samples for the fist-family ASL letters: E, M, N, S, T.

Usage:
    python collect_fist_data.py

Controls:
  SPACE      → capture one sample for the current letter
  BACKSPACE  → undo the last capture
  N          → skip to next letter (even if quota not met)
  A          → toggle auto-capture (captures after 1 s of hand stillness)
  Q / ESC    → save collected data and quit

Output:
  fist_data.npz   — feature matrix X (n_samples × 63) and label array y
"""
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
import numpy as np
import cv2
import mediapipe as mp

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LETTERS        = ["E", "M", "N", "S", "T"]
SAMPLES_PER    = 80          # target samples per letter
SAVE_PATH      = "fist_data.npz"
AUTO_HOLD_SECS = 1.0         # seconds of stillness before auto-capture fires
MOVE_THRESH    = 0.015       # normalised movement below this = "hand still"

WIN = "Fist Data Collection"


# ── Feature extraction (must match train_fist_classifier.py) ─────────────────

def _extract(hand_lm):
    """Return a 63-d normalised landmark feature vector."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                   dtype=np.float32)
    pts -= pts[0]                                         # centre at wrist
    pts /= (float(np.linalg.norm(pts[9, :2])) + 1e-6)    # scale by wrist→MCP9
    return pts.flatten()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _bar(img, x, y, w, h, ratio, col_on, col_off=(35, 35, 35)):
    cv2.rectangle(img, (x, y), (x + w,          y + h), col_off, -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0, ratio)), y + h), col_on, -1)
    cv2.rectangle(img, (x, y), (x + w,          y + h), (70, 70, 70), 1)


def _put(img, text, x, y, scale=0.55, color=(180, 180, 180), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    ok, f0 = cap.read()
    H, W   = f0.shape[:2] if ok else (720, 1280)
    PW     = W // 2          # each panel is half the frame width

    X_all : list = []        # feature vectors
    y_all : list = []        # integer labels (index into LETTERS)

    cur_idx     = 0          # index of letter currently being collected
    auto_mode   = False
    last_pos    = None       # (x, y) of middle MCP last frame
    still_since = None       # time.time() when hand became still
    flash_until = 0.0        # timestamp until which the green flash is shown

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while cur_idx < len(LETTERS):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now   = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            hand_lm    = None
            stable_pct = 0.0

            if res.multi_hand_landmarks:
                hand_lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                # Track stability via middle MCP (landmark 9)
                ref  = hand_lm.landmark[9]
                pos  = (ref.x, ref.y)
                if last_pos is not None:
                    dist = ((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)**0.5
                    still_since = (still_since or now) if dist < MOVE_THRESH else None
                last_pos = pos
                if still_since:
                    stable_pct = min((now - still_since) / AUTO_HOLD_SECS, 1.0)
            else:
                last_pos    = None
                still_since = None

            # ── Capture logic ─────────────────────────────────────────────
            letter_count = sum(1 for yy in y_all if yy == cur_idx)

            if (auto_mode and stable_pct >= 1.0
                    and hand_lm is not None
                    and letter_count < SAMPLES_PER):
                X_all.append(_extract(hand_lm))
                y_all.append(cur_idx)
                flash_until = now + 0.25
                still_since = None     # require new stillness before next auto-fire
                letter_count += 1

            # Auto-advance to next letter when quota is full
            if letter_count >= SAMPLES_PER:
                cur_idx    += 1
                still_since = None
                continue

            # ── Camera panel ──────────────────────────────────────────────
            cam = frame[:, W // 2 - PW // 2 : W // 2 + PW // 2].copy()

            # Stability arc around middle MCP
            if hand_lm and still_since and stable_pct > 0:
                lm9     = hand_lm.landmark[9]
                cx, cy  = int(lm9.x * PW), int(lm9.y * H)
                cv2.ellipse(cam, (cx, cy), (32, 32), -90,
                            0, int(360 * stable_pct),
                            (0, 220, 220), 3, cv2.LINE_AA)

            # Green flash on capture
            if now < flash_until:
                overlay = np.full_like(cam, (0, 210, 90))
                cv2.addWeighted(cam, 0.5, overlay, 0.5, 0, cam)

            _put(cam, "Camera", 10, 24, scale=0.65, color=(100, 100, 100))

            # ── Info panel ────────────────────────────────────────────────
            info = np.full((H, PW, 3), 14, np.uint8)

            # Letter checklist (left column)
            for i, lt in enumerate(LETTERS):
                cnt  = sum(1 for yy in y_all if yy == i)
                done = cnt >= SAMPLES_PER
                curr = (i == cur_idx)
                col  = ((0, 200, 100) if done else
                        (0, 220, 220) if curr else
                        (55, 55, 55))
                tick = "v " if done else "  "
                _put(info, f"{tick}{lt}  {cnt:>3}/{SAMPLES_PER}",
                     24, 48 + i * 38,
                     scale=0.75, color=col, thick=2 if curr else 1)

            # Giant target letter (centre)
            ltr = LETTERS[cur_idx]
            cv2.putText(info, ltr,
                        (PW // 2 - 68, H // 2 + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 7.0,
                        (0, 220, 220), 8, cv2.LINE_AA)

            # Progress bar for current letter
            lc = sum(1 for yy in y_all if yy == cur_idx)
            by = H // 2 + 130
            _bar(info, 24, by, PW - 48, 22, lc / SAMPLES_PER,
                 col_on=(0, 170, 90))
            _put(info, f"{lc} / {SAMPLES_PER}", 24, by - 7,
                 color=(130, 130, 130))

            # Auto-capture toggle indicator
            auto_col = (0, 190, 200) if auto_mode else (70, 70, 70)
            _put(info, f"Auto-capture: {'ON ' if auto_mode else 'OFF'}  [A to toggle]",
                 24, H - 80, color=auto_col)

            _put(info, "Tip: vary hand angle a little between captures for robustness",
                 24, H - 52, color=(60, 60, 60))
            _put(info,
                 "SPACE=capture   BKSP=undo   N=next   A=auto   Q=save+quit",
                 24, H - 18, color=(55, 55, 55))

            # ── Combine panels and display ────────────────────────────────
            cv2.imshow(WIN, np.hstack([cam, info]))

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):          # Q or ESC → save and quit
                break
            elif key == ord("n"):              # skip to next letter
                cur_idx    += 1
                still_since = None
            elif key == ord("a"):              # toggle auto-capture
                auto_mode   = not auto_mode
                still_since = None
            elif key == 8:                     # BACKSPACE → undo last sample
                if y_all and y_all[-1] == cur_idx:
                    X_all.pop()
                    y_all.pop()
            elif key == ord(" ") and hand_lm is not None:   # SPACE → capture
                lc = sum(1 for yy in y_all if yy == cur_idx)
                if lc < SAMPLES_PER:
                    X_all.append(_extract(hand_lm))
                    y_all.append(cur_idx)
                    flash_until = now + 0.25

    cap.release()
    cv2.destroyAllWindows()

    # ── Save ──────────────────────────────────────────────────────────────────
    if X_all:
        np.savez(SAVE_PATH,
                 X      = np.array(X_all, dtype=np.float32),
                 y      = np.array(y_all, dtype=np.int32),
                 labels = np.array(LETTERS))
        print(f"\nSaved {len(X_all)} samples → '{SAVE_PATH}'")
        for i, lt in enumerate(LETTERS):
            print(f"  {lt}: {sum(1 for yy in y_all if yy == i)}")
    else:
        print("No data collected — nothing saved.")


if __name__ == "__main__":
    run()
