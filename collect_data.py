"""
Collect training samples for all 24 static ASL letters (A–Y, excluding J and Z).

Usage:
    python collect_data.py          # resumes from existing asl_data.npz
    python collect_data.py --fresh  # ignores existing data and starts over

Controls:
  SPACE      → capture one sample for the current letter
  BACKSPACE  → undo last capture for the current letter
  N          → next letter
  P          → previous letter
  A          → toggle auto-capture (fires after 1 s of hand stillness)
  Q / ESC    → save and quit

Output:
  asl_data.npz  — feature matrix X (n_samples × 63) and label array y
                  Automatically merged with existing data on resume.
"""
import os, sys, warnings, logging
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

# All static ASL letters (J and Z require motion — excluded)
LETTERS        = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # 24 letters
SAMPLES_PER    = 60          # target per letter  (60 × 24 = 1 440 total)
SAVE_PATH      = "asl_data.npz"
AUTO_HOLD_SECS = 1.0
MOVE_THRESH    = 0.015

WIN = "ASL Data Collection"


# ── Feature extraction (identical to train_classifier.py) ────────────────────

def _extract(hand_lm):
    """63-d normalised landmark feature vector."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                   dtype=np.float32)
    pts -= pts[0]                                         # centre at wrist
    pts /= (float(np.linalg.norm(pts[9, :2])) + 1e-6)    # scale by wrist→MCP9
    return pts.flatten()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _bar(img, x, y, w, h, ratio, col_on, col_off=(35, 35, 35)):
    cv2.rectangle(img, (x, y), (x + w, y + h), col_off, -1)
    if ratio > 0:
        cv2.rectangle(img, (x, y),
                      (x + int(w * min(ratio, 1.0)), y + h), col_on, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)


def _put(img, text, x, y, scale=0.55, color=(180, 180, 180), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(fresh: bool = False):
    # ── Resume existing data ──────────────────────────────────────────────────
    X_all: list = []
    y_all: list = []

    if not fresh and os.path.exists(SAVE_PATH):
        data  = np.load(SAVE_PATH, allow_pickle=True)
        X_all = list(data["X"])
        y_all = list(data["y"].astype(int))
        total = len(X_all)
        print(f"Resumed from '{SAVE_PATH}': {total} existing samples.")
    else:
        print("Starting fresh collection.")

    # Start at the first letter that still needs samples
    def _count(idx):
        return sum(1 for yy in y_all if yy == idx)

    cur_idx = next((i for i in range(len(LETTERS)) if _count(i) < SAMPLES_PER), 0)

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    ok, f0 = cap.read()
    H, W   = f0.shape[:2] if ok else (720, 1280)
    PW     = W // 2

    auto_mode   = False
    last_pos    = None
    still_since = None
    flash_until = 0.0

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
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
                ref = hand_lm.landmark[9]
                pos = (ref.x, ref.y)
                if last_pos is not None:
                    dist = ((pos[0]-last_pos[0])**2 + (pos[1]-last_pos[1])**2)**0.5
                    still_since = (still_since or now) if dist < MOVE_THRESH else None
                last_pos = pos
                if still_since:
                    stable_pct = min((now - still_since) / AUTO_HOLD_SECS, 1.0)
            else:
                last_pos    = None
                still_since = None

            letter_count = _count(cur_idx)

            # ── Auto-capture ──────────────────────────────────────────────
            if (auto_mode and stable_pct >= 1.0
                    and hand_lm is not None
                    and letter_count < SAMPLES_PER):
                X_all.append(_extract(hand_lm))
                y_all.append(cur_idx)
                flash_until  = now + 0.25
                still_since  = None
                letter_count += 1

            # ── Camera panel ──────────────────────────────────────────────
            cam = frame[:, W//2 - PW//2 : W//2 + PW//2].copy()

            if hand_lm and still_since and stable_pct > 0:
                lm9    = hand_lm.landmark[9]
                cx, cy = int(lm9.x * PW), int(lm9.y * H)
                cv2.ellipse(cam, (cx, cy), (32, 32), -90,
                            0, int(360 * stable_pct),
                            (0, 220, 220), 3, cv2.LINE_AA)

            if now < flash_until:
                ov = np.full_like(cam, (0, 210, 90))
                cv2.addWeighted(cam, 0.5, ov, 0.5, 0, cam)

            _put(cam, "Camera", 10, 24, scale=0.65, color=(100, 100, 100))

            # ── Info panel ────────────────────────────────────────────────
            info = np.full((H, PW, 3), 14, np.uint8)

            # ── Letter progress grid (6 columns × 4 rows) ─────────────────
            cols, rows = 6, 4
            gx0, gy0  = 16, 16
            cell_w    = (PW - gx0 * 2) // cols
            cell_h    = 52

            for i, lt in enumerate(LETTERS):
                row, col = divmod(i, cols)
                cx_  = gx0 + col * cell_w
                cy_  = gy0 + row * cell_h
                cnt  = _count(i)
                done = cnt >= SAMPLES_PER
                curr = (i == cur_idx)

                # Cell background
                bg_col = (25, 50, 25) if done else (40, 40, 80) if curr else (20, 20, 20)
                cv2.rectangle(info, (cx_, cy_), (cx_ + cell_w - 4, cy_ + cell_h - 4),
                              bg_col, -1)
                if curr:
                    cv2.rectangle(info, (cx_, cy_),
                                  (cx_ + cell_w - 4, cy_ + cell_h - 4),
                                  (0, 180, 220), 2)

                # Letter label
                lt_col = (0, 220, 100) if done else (0, 220, 220) if curr else (100, 100, 100)
                cv2.putText(info, lt, (cx_ + 6, cy_ + 28),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, lt_col, 2, cv2.LINE_AA)

                # Mini count
                _put(info, str(cnt), cx_ + 6, cy_ + cell_h - 8,
                     scale=0.38, color=(80, 80, 80))

                # Mini bar
                bw = cell_w - 12
                _bar(info, cx_ + 6, cy_ + cell_h - 6, bw, 3,
                     cnt / SAMPLES_PER,
                     col_on=(0, 160, 80) if done else (0, 100, 180))

            # ── Current letter detail ──────────────────────────────────────
            detail_y = gy0 + rows * cell_h + 12

            ltr = LETTERS[cur_idx]
            cv2.putText(info, ltr,
                        (PW // 2 - 48, detail_y + 90),
                        cv2.FONT_HERSHEY_DUPLEX, 5.5,
                        (0, 220, 220), 7, cv2.LINE_AA)

            lc = _count(cur_idx)
            _bar(info, 20, detail_y + 108, PW - 40, 18,
                 lc / SAMPLES_PER, col_on=(0, 170, 90))
            _put(info, f"{lc} / {SAMPLES_PER}", 20, detail_y + 103,
                 color=(120, 120, 120))

            # ── Overall progress ───────────────────────────────────────────
            total_done = len(y_all)
            total_need = SAMPLES_PER * len(LETTERS)
            _bar(info, 20, H - 95, PW - 40, 8,
                 total_done / total_need, col_on=(0, 130, 80), col_off=(25, 25, 25))
            _put(info, f"Total: {total_done} / {total_need}",
                 20, H - 100, color=(90, 90, 90))

            # ── Controls & tips ────────────────────────────────────────────
            ac = "ON " if auto_mode else "OFF"
            _put(info, f"Auto: {ac} [A]   Tip: vary angle between captures",
                 20, H - 68, color=(0, 160, 180) if auto_mode else (60, 60, 60))
            _put(info, "SPACE=capture  BKSP=undo  N=next  P=prev  A=auto  Q=save+quit",
                 20, H - 18, color=(50, 50, 50))

            # ── Combine & show ────────────────────────────────────────────
            cv2.imshow(WIN, np.hstack([cam, info]))

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                cur_idx     = (cur_idx + 1) % len(LETTERS)
                still_since = None
            elif key == ord("p"):
                cur_idx     = (cur_idx - 1) % len(LETTERS)
                still_since = None
            elif key == ord("a"):
                auto_mode   = not auto_mode
                still_since = None
            elif key == 8:                          # BACKSPACE — undo
                indices = [i for i, yy in enumerate(y_all) if yy == cur_idx]
                if indices:
                    idx = indices[-1]
                    X_all.pop(idx)
                    y_all.pop(idx)
            elif key == ord(" ") and hand_lm is not None:
                if _count(cur_idx) < SAMPLES_PER:
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
            print(f"  {lt}: {_count(i)}")
    else:
        print("No data collected — nothing saved.")


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    run(fresh=fresh)
