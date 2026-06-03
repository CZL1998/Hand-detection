"""
ASL finger-spelling classifier (static letters only).
J and Z require motion tracking and return '?'.

Coordinate system (MediaPipe normalised):
  x: 0 = left, 1 = right
  y: 0 = top,  1 = bottom  ← smaller y means higher in the image
  z: depth (not used here)

Fist-family letters (E, M, N, S, T) are handled by a small SVM trained with
collect_fist_data.py + train_fist_classifier.py.  While no model file exists
the code falls back to geometric heuristics.
"""
import math
import os
import numpy as np

# ML models — loaded once at import time.
# asl_classifier.pkl  (full 24-letter model) takes priority over
# fist_classifier.pkl (fist-family only).  Geometric heuristics are the fallback.
_ASL_MODEL  = None   # full 24-letter model (train_classifier.py)
_FIST_MODEL = None   # fist-family model    (train_fist_classifier.py)
try:
    import joblib
    _DIR = os.path.dirname(__file__)
    _asl_pkl  = os.path.join(_DIR, "asl_classifier.pkl")
    _fist_pkl = os.path.join(_DIR, "fist_classifier.pkl")
    if os.path.exists(_asl_pkl):
        _ASL_MODEL  = joblib.load(_asl_pkl)
    elif os.path.exists(_fist_pkl):
        _FIST_MODEL = joblib.load(_fist_pkl)
except Exception:
    pass


def _d(a, b):
    """2-D Euclidean distance between two landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def _fist_features(lm):
    """63-d feature vector: landmarks centred at wrist, scaled by wrist→MCP9."""
    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    pts -= pts[0]                                       # origin = wrist
    pts /= (float(np.linalg.norm(pts[9, :2])) + 1e-6)  # normalise scale
    return pts.flatten()


def classify_asl(hand_landmarks, handedness_label):
    """
    Classify a static ASL letter from MediaPipe hand landmarks.

    Args:
        hand_landmarks: mediapipe NormalizedLandmarkList (21 points)
        handedness_label: "Left" or "Right" (as reported by MediaPipe)

    Returns:
        Single uppercase letter A-Z, or '?' for uncertain / dynamic gestures.
    """
    lm = hand_landmarks.landmark

    # ── Full 24-letter ML model (highest priority) ────────────────────
    if _ASL_MODEL is not None:
        feats  = _fist_features(lm)
        scaled = _ASL_MODEL["scaler"].transform(feats.reshape(1, -1))
        idx    = int(_ASL_MODEL["clf"].predict(scaled)[0])
        return _ASL_MODEL["labels"][idx]

    right = (handedness_label == "Right")

    # ── Reference length: wrist → middle-finger MCP ─────────────────
    ref = max(_d(lm[0], lm[9]), 1e-6)

    # ── Finger extension: tip.y < PIP.y  (tip higher than middle joint) ──
    idx_up = lm[8].y  < lm[6].y    # index
    mid_up = lm[12].y < lm[10].y   # middle
    rng_up = lm[16].y < lm[14].y   # ring
    pnk_up = lm[20].y < lm[18].y   # pinky
    n = idx_up + mid_up + rng_up + pnk_up

    # ── Thumb extended sideways (handedness-aware) ────────────────────
    # Right hand: tip to the LEFT of IP joint = extended
    thumb_out  = (lm[4].x < lm[3].x) if right else (lm[4].x > lm[3].x)
    thumb_high = lm[4].y < lm[5].y   # thumb tip above index MCP

    # ── Normalised tip-to-tip distances ──────────────────────────────
    ti = _d(lm[4], lm[8])  / ref   # thumb → index tip
    tm = _d(lm[4], lm[12]) / ref   # thumb → middle tip
    tr = _d(lm[4], lm[16]) / ref   # thumb → ring tip
    im = _d(lm[8], lm[12]) / ref   # index → middle tip

    # Horizontal spread between index and middle tips
    im_h = abs(lm[8].x - lm[12].x) / ref

    # ── Finger reach: tip-to-own-MCP distance ────────────────────────
    # Large = extended, small = curled into palm
    ir = _d(lm[8],  lm[5])  / ref  # index
    mr = _d(lm[12], lm[9])  / ref  # middle
    rr = _d(lm[16], lm[13]) / ref  # ring
    pr = _d(lm[20], lm[17]) / ref  # pinky

    # ── Horizontal reach of index / middle (for G, H) ────────────────
    idx_horiz = abs(lm[8].x  - lm[5].x) / ref
    mid_horiz = abs(lm[12].x - lm[9].x) / ref

    # ═════════════════════════════════════════════════════════════════
    # Classification tree
    # ═════════════════════════════════════════════════════════════════

    # ── B: all 4 fingers up, thumb folded ───────────────────────────
    if n == 4:
        return "B"

    # ── 3 fingers up ────────────────────────────────────────────────
    if n == 3:
        # F: index curled to touch thumb; middle+ring+pinky up
        if mid_up and rng_up and pnk_up and not idx_up:
            return "F"
        # W: index+middle+ring up
        return "W"

    # ── 2 fingers up ────────────────────────────────────────────────
    if n == 2:
        if idx_up and mid_up:
            # R: index wraps over middle — tips x-order reverses relative to MCPs
            mcp_idx_left = lm[5].x < lm[9].x   # index MCP left of middle MCP?
            tip_idx_left = lm[8].x < lm[12].x   # index TIP left of middle TIP?
            is_crossed   = (mcp_idx_left != tip_idx_left)
            if is_crossed:
                return "R"
            # K: thumb elevated AND positioned between the two finger tips
            # catches both lateral (thumb_out) and forward thumb orientations
            lo_x, hi_x         = min(lm[8].x, lm[12].x), max(lm[8].x, lm[12].x)
            thumb_between_pair  = lo_x < lm[4].x < hi_x
            if (thumb_out or thumb_between_pair) and thumb_high:
                return "K"
            return "V" if im_h > 0.20 else "U"
        return "U"                               # other 2-up combinations → closest

    # ── 1 finger up ─────────────────────────────────────────────────
    if n == 1:
        if idx_up:
            return "L" if thumb_out else "D"
        if pnk_up:
            return "Y" if thumb_out else "I"
        return "?"

    # ── 0 fingers up ────────────────────────────────────────────────

    # G / H: index (and middle) pointing sideways
    if idx_horiz > 0.45:
        return "H" if mid_horiz > 0.35 else "G"

    # X: index tip hooked back (tip below its DIP joint)
    if lm[8].y > lm[7].y and ir > 0.22:
        return "X"

    # O: all fingertips converge near thumb tip
    if ti < 0.40 and tm < 0.50 and tr < 0.60:
        return "O"

    # C: open curve
    # Wider ti range tolerates angle variation; ir/mr lower-bound keeps E separated
    # (E has deeply-curled fingers with ir/mr < 0.45, C stays above 0.40)
    if 0.30 < ti < 0.88 and ir > 0.40 and mr > 0.40 and not thumb_out:
        return "C"

    # Q: thumb + index both point downward
    if thumb_out and lm[4].y > lm[0].y:
        return "Q"

    # ── Fist family: A, E, M, N, S, T ───────────────────────────────

    # A: thumb beside fist (extended sideways)
    if thumb_out:
        return "A"

    # ML classifier for E / M / N / S / T (trained via collect_fist_data.py)
    if _FIST_MODEL is not None:
        feats  = _fist_features(lm)
        scaled = _FIST_MODEL["scaler"].transform(feats.reshape(1, -1))
        idx    = int(_FIST_MODEL["clf"].predict(scaled)[0])
        return _FIST_MODEL["labels"][idx]

    # ── Geometric fallbacks (active until model is trained) ──────────

    # E: all four fingertips deeply curled toward palm
    if ir < 0.45 and mr < 0.45 and rr < 0.45 and pr < 0.45:
        return "E"

    # T: thumb pokes up between index and middle from the palm side.
    # Require a margin so S (thumb near MCP area) doesn't mis-fire here.
    lx, rx = sorted([lm[5].x, lm[9].x])
    span   = rx - lx
    if (lx + span * 0.1 < lm[4].x < rx - span * 0.1
            and lm[4].y > lm[5].y + (lm[0].y - lm[5].y) * 0.15):
        return "T"

    # S vs M/N — the critical distinction is DEPTH ORDER:
    #   S   : thumb wraps to the FRONT of the fist  (closer to camera → smaller z)
    #   M/N : fingers drape OVER the thumb           (finger tips closer → smaller z)
    #
    # MediaPipe z: origin = wrist; more negative = closer to camera.
    # We require a margin (0.02) so that z-noise does not flip the decision.
    avg_tip_z = (lm[8].z + lm[12].z + lm[16].z) / 3
    if lm[4].z < avg_tip_z - 0.02:   # thumb clearly in front → S
        return "S"

    # Thumb is behind the fingers: M (3 draped) or N (2 draped).
    # tr = thumb-to-ring distance; small when ring finger also covers thumb.
    dx_middle = abs(lm[4].x - lm[9].x) / ref
    if tr < 0.45:          # ring tip near thumb → M
        return "M"
    if dx_middle < 0.32:   # thumb aligned with middle MCP → N
        return "N"

    return "S"             # fallback (z margin was inconclusive)


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic gesture detector — J and Z
# ══════════════════════════════════════════════════════════════════════════════

class JZDetector:
    """
    Stateful per-hand detector for dynamic ASL letters J and Z.

    Both letters start from a static prerequisite pose that is held briefly,
    then the fingertip traces a characteristic path:
      J : pinky-up (I pose) → downward stroke with a hook at the bottom
      Z : index-up (D pose) → right → diagonal-down-left → right  (Z shape)

    Usage:
        det = JZDetector()                              # one per tracked hand
        letter = det.update(static_letter, lm)         # call every frame
        # returns 'J', 'Z', or None
    """
    _STILL   = 5    # consecutive prerequisite-pose frames to arm tracking
    _TIMEOUT = 50   # max tracking frames before reset  (≈ 1.7 s at 30 fps)
    _MIN_SPAN = 0.08  # minimum x+y trajectory span to count as intentional

    def __init__(self):
        self._prereq   = None   # 'J' or 'Z'
        self._still    = 0
        self._tracking = False
        self._pts: list = []    # (x, y) fingertip positions during tracking

    def update(self, static_letter: str, lm) -> "str | None":
        """Feed one frame; returns detected letter or None."""
        if static_letter == 'I':
            prereq, tip = 'J', (lm[20].x, lm[20].y)   # pinky tip
        elif static_letter == 'D':
            prereq, tip = 'Z', (lm[8].x,  lm[8].y)    # index tip
        else:
            return self._flush()

        if self._prereq != prereq:
            self._reset(prereq)

        if not self._tracking:
            self._still += 1
            if self._still >= self._STILL:
                self._tracking = True
            return None

        self._pts.append(tip)
        if len(self._pts) > self._TIMEOUT:
            self._reset()
            return None

        return self._classify()

    # ── private ────────────────────────────────────────────────────

    def _flush(self):
        result = self._classify() if self._tracking else None
        self._reset()
        return result

    def _reset(self, prereq=None):
        self._prereq   = prereq
        self._still    = 0
        self._tracking = False
        self._pts      = []

    def _classify(self):
        pts = self._pts
        if len(pts) < 8:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_x + span_y < self._MIN_SPAN:
            return None
        letter = _jz_j(xs, ys, span_x, span_y) if self._prereq == 'J' \
            else _jz_z(xs, ys, span_x, span_y)
        if letter:
            self._reset()
        return letter


def _jz_j(xs, ys, span_x, span_y):
    """J: downward stroke + lateral hook at the bottom."""
    net_dy = ys[-1] - ys[0]      # positive = moved down (screen y increases)
    if net_dy > 0.08 and span_y > span_x and span_x > 0.04:
        return 'J'
    return None


def _jz_z(xs, ys, span_x, span_y):
    """Z: index tip traces right → down-left → right  (2 x-direction reversals)."""
    if span_x < 0.10 or span_y < 0.05:
        return None
    # Hysteresis: must move 12 % of total span before a direction change counts
    step      = span_x * 0.12
    reversals = 0
    anchor    = xs[0]
    last_dir  = 0
    for x in xs[1:]:
        diff = x - anchor
        if abs(diff) > step:
            curr = 1 if diff > 0 else -1
            if last_dir != 0 and curr != last_dir:
                reversals += 1
            last_dir = curr
            anchor   = x
    return 'Z' if reversals >= 2 else None
