"""
ASL finger-spelling classifier (static letters only).
J and Z require motion tracking and return '?'.

Coordinate system (MediaPipe normalised):
  x: 0 = left, 1 = right
  y: 0 = top,  1 = bottom  ← smaller y means higher in the image
  z: depth (not used here)
"""
import math


def _d(a, b):
    """2-D Euclidean distance between two landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


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
            if im < 0.22:                   # tips nearly touching → R (crossed)
                return "R"
            if thumb_out and thumb_high:    # thumb between two fingers → K
                return "K"
            return "V" if im_h > 0.20 else "U"
        return "U"                          # other 2-up combinations → closest

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

    # C: open curve — tips moderately far from thumb, thumb not extended
    if 0.35 < ti < 0.80 and not thumb_out:
        return "C"

    # Q: thumb + index both point downward
    if thumb_out and lm[4].y > lm[0].y:
        return "Q"

    # ── Fist family: A, E, M, N, S, T ───────────────────────────────

    # A: thumb beside fist (extended sideways)
    if thumb_out:
        return "A"

    # E: all four fingertips deeply curled toward palm
    if ir < 0.45 and mr < 0.45 and rr < 0.45 and pr < 0.45:
        return "E"

    # T: thumb tip horizontally between index MCP and middle MCP
    lx, rx = sorted([lm[5].x, lm[9].x])
    if lx < lm[4].x < rx and lm[4].y > lm[5].y:
        return "T"

    # M / N: fingers draped over thumb; distinguish by which MCPs are nearby
    dx_ring   = abs(lm[4].x - lm[13].x) / ref
    dx_middle = abs(lm[4].x - lm[9].x)  / ref
    if dx_ring < 0.28:
        return "M"
    if dx_middle < 0.28:
        return "N"

    # S: thumb across curled fist (default fist)
    return "S"
