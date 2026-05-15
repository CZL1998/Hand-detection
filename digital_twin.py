"""
Hand Digital Twin — real-time 3D hand with skin rendering.

Left panel : camera feed with MediaPipe landmark overlay.
Right panel: shaded skin-coloured 3D hand, mouse-draggable.

Controls:
  Drag mouse on right panel  →  rotate 3D view (yaw / pitch)
  Q                          →  quit
"""
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # suppress TF C++ info/warning logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN notice
os.environ.setdefault("GLOG_minloglevel", "2")  # suppress MediaPipe C++ INFO logs
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import math
import cv2
import numpy as np
import mediapipe as mp

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WIN   = "Hand Digital Twin"
FOCAL = 2.5   # perspective focal length

# ── Skin shading parameters ───────────────────────────────────────────────
SKIN_BASE = np.array([120, 175, 210], np.float32)   # BGR: warm beige skin
AMBIENT   = 0.30
DIFFUSE   = 0.70

# Light direction (world space, pointing FROM surface TO light source)
_L = np.array([0.35, 0.80, -0.45], np.float32)
_L /= np.linalg.norm(_L)

# ── Joint radii (normalised 3-D units, hand spans roughly [-1, 1]) ───────
JOINT_R = [
    0.072,   #  0  wrist
    0.040,   #  1  thumb CMC
    0.037,   #  2  thumb MCP
    0.033,   #  3  thumb IP
    0.029,   #  4  thumb TIP
    0.044,   #  5  index MCP
    0.036,   #  6  index PIP
    0.031,   #  7  index DIP
    0.027,   #  8  index TIP
    0.042,   #  9  middle MCP
    0.036,   # 10  middle PIP
    0.032,   # 11  middle DIP
    0.028,   # 12  middle TIP
    0.040,   # 13  ring MCP
    0.034,   # 14  ring PIP
    0.029,   # 15  ring DIP
    0.025,   # 16  ring TIP
    0.037,   # 17  pinky MCP
    0.030,   # 18  pinky PIP
    0.026,   # 19  pinky DIP
    0.022,   # 20  pinky TIP
]

# Finger bones only (no palm cross-bars)
BONES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index
    (0, 9),  (9, 10), (10,11), (11,12),   # middle
    (0,13), (13,14), (14,15), (15,16),    # ring
    (0,17), (17,18), (18,19), (19,20),    # pinky
]

# Palm: fan of triangles from wrist
PALM_TRIS = [(0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17)]

# Adjacent MCP pairs for finger webbing
WEB_PAIRS = [(5, 9), (9, 13), (13, 17)]


# ── 3-D math ──────────────────────────────────────────────────────────────

def _rot(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rx = np.array([[1, 0, 0],   [0, cp, -sp], [0, sp, cp]], np.float32)
    return Rx @ Ry


def _to_3d(hand_lm, aspect: float = 1.0):
    """Centred, normalised (21,3) array + wrist screen position in [0,1].

    aspect = image_width / image_height
    MediaPipe normalises x by image width and y by image height, so the same
    physical distance is smaller in x for wide cameras. Multiplying x by the
    aspect ratio restores isotropic physical proportions.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], np.float32)
    wrist_screen = (float(pts[0, 0]), float(pts[0, 1]))
    pts -= pts[0]
    pts[:, 0] *= aspect                    # correct x-axis compression
    pts[:, 1] *= -1                        # flip y: screen-down → 3D-up
    pts /= (np.max(np.abs(pts)) + 1e-6)
    return pts, wrist_screen


def _project(pts: np.ndarray, pw: int, ph: int, wrist_screen: tuple):
    """Perspective-project (N,3) → list of (px, py); wrist at its true position."""
    scale = ph * 0.72
    cx    = int(wrist_screen[0] * pw)
    cy    = int(wrist_screen[1] * ph)
    out   = []
    for x, y, z in pts:
        d = FOCAL / (FOCAL + z + 1e-6)
        out.append((int(cx + x * d * scale),
                    int(cy - y * d * scale)))
    return out


# ── Skin rendering helpers ────────────────────────────────────────────────

def _proj_radius(r3d: float, z: float, ph: int) -> int:
    """Convert a 3-D radius to a screen-space pixel radius."""
    d = FOCAL / (FOCAL + float(z) + 1e-6)
    return max(2, int(r3d * d * ph * 0.72))


def _skin_shade(a3d: np.ndarray, b3d: np.ndarray) -> float:
    """Lambertian shading for the front-facing surface of a bone capsule."""
    bone   = b3d - a3d
    bone_n = bone / (np.linalg.norm(bone) + 1e-6)
    view   = np.array([0., 0., 1.], np.float32)
    side   = np.cross(bone_n, view)
    slen   = np.linalg.norm(side)
    if slen < 1e-4:
        side = np.array([1., 0., 0.], np.float32)
    else:
        side /= slen
    front = np.cross(side, bone_n)
    if front[2] > 0:
        front = -front
    front /= (np.linalg.norm(front) + 1e-6)
    diffuse = max(0.0, float(np.dot(front, _L)))
    return AMBIENT + DIFFUSE * diffuse


def _fill_capsule(img: np.ndarray,
                  pA: np.ndarray, pB: np.ndarray,
                  rA: int, rB: int,
                  color: tuple) -> None:
    """Fill a tapered capsule (trapezoid + hemispherical end caps) in BGR."""
    d  = pB - pA
    ln = float(np.linalg.norm(d))
    if ln < 0.5:
        cv2.circle(img, (int(pA[0]), int(pA[1])), max(1, rA), color, -1)
        return
    n = np.array([-d[1], d[0]]) / ln

    quad = np.round(np.array([
        pA + rA * n,
        pB + rB * n,
        pB - rB * n,
        pA - rA * n,
    ])).astype(np.int32)

    cv2.fillPoly(img, [quad], color)
    cv2.circle(img, (int(round(pA[0])), int(round(pA[1]))), max(1, rA), color, -1)
    cv2.circle(img, (int(round(pB[0])), int(round(pB[1]))), max(1, rB), color, -1)


def _render_skin(pts3d: np.ndarray, proj: list, ph: int, pw: int) -> np.ndarray:
    """
    Render a shaded, flesh-coloured hand onto a black (ph × pw) canvas.

    Pipeline:
      1. Palm fill  — triangle fan + finger webbing quads
      2. 3-layer bone capsules (shadow / mid / highlight) for roundness
      3. Surface smoothing — light blur within skin region
    """
    img = np.zeros((ph, pw, 3), np.uint8)

    def _color(base: np.ndarray, shade: float) -> tuple:
        return tuple(int(v) for v in np.clip(base * shade, 0, 255))

    light_2d  = np.array([_L[0], -_L[1]], np.float32)
    light_2d /= np.linalg.norm(light_2d) + 1e-6

    palm_ids   = [0, 1, 5, 9, 13, 17]
    avg_y_palm = float(np.mean([pts3d[i][1] for i in palm_ids]))
    shade_palm = float(np.clip(AMBIENT + DIFFUSE * (avg_y_palm * 0.5 + 0.6), 0.25, 1.0))
    palm_color = _color(SKIN_BASE, shade_palm)

    # ── 1a. Palm triangle fan ─────────────────────────────────────────────
    for a, b, c in PALM_TRIS:
        tri = np.array([proj[a], proj[b], proj[c]], np.int32)
        cv2.fillPoly(img, [tri], palm_color)

    # ── 1b. Finger webbing between adjacent MCP pairs ─────────────────────
    for ma, mb in WEB_PAIRS:
        pA = np.array(proj[ma],     np.float32)
        pB = np.array(proj[mb],     np.float32)
        nA = np.array(proj[ma + 1], np.float32)
        nB = np.array(proj[mb + 1], np.float32)
        wA = (pA + 0.30 * (nA - pA)).astype(np.int32)
        wB = (pB + 0.30 * (nB - pB)).astype(np.int32)
        quad = np.array([proj[ma], wA, wB, proj[mb]], np.int32)
        cv2.fillPoly(img, [quad], palm_color)

    # ── 2. 3-layer capsules: far → near ───────────────────────────────────
    for i, j in sorted(BONES, key=lambda b: -(pts3d[b[0]][2] + pts3d[b[1]][2])):
        shade = float(np.clip(_skin_shade(pts3d[i], pts3d[j]), 0.25, 1.0))
        pA    = np.array(proj[i], np.float32)
        pB    = np.array(proj[j], np.float32)
        rA    = _proj_radius(JOINT_R[i], pts3d[i][2], ph)
        rB    = _proj_radius(JOINT_R[j], pts3d[j][2], ph)

        seg   = pB - pA
        perp  = np.array([-seg[1], seg[0]], np.float32)
        perp /= np.linalg.norm(perp) + 1e-6
        lp    = float(np.dot(light_2d, perp))
        off   = perp * lp * rA * 0.35

        # Shadow layer — full width, darkened
        _fill_capsule(img, pA, pB, rA, rB, _color(SKIN_BASE, shade * 0.66))
        # Mid layer — 70% width, shifted toward light
        _fill_capsule(img,
                      pA + off * 0.4, pB + off * 0.4,
                      max(1, int(rA * 0.70)), max(1, int(rB * 0.70)),
                      _color(SKIN_BASE, shade))
        # Highlight layer — 35% width, full shift
        _fill_capsule(img,
                      pA + off, pB + off,
                      max(1, int(rA * 0.35)), max(1, int(rB * 0.35)),
                      _color(SKIN_BASE, min(1.0, shade * 1.28)))

    # ── 3. Smooth skin surface ────────────────────────────────────────────
    skin_mask = img.sum(axis=2) > 0
    blurred   = cv2.GaussianBlur(img, (7, 7), 0)
    img[skin_mask] = blurred[skin_mask]

    return img


# ── Background ────────────────────────────────────────────────────────────

def _make_bg(h: int, w: int) -> np.ndarray:
    bg = np.full((h, w, 3), 8, np.uint8)
    for x in range(0, w, 50):
        cv2.line(bg, (x, 0), (x, h), (22, 22, 22), 1)
    for y in range(0, h, 50):
        cv2.line(bg, (0, y), (w, y), (22, 22, 22), 1)
    return bg


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头。")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    ok, f0 = cap.read()
    H, W   = f0.shape[:2] if ok else (720, 1280)
    PW     = W // 2
    bg     = _make_bg(H, PW)

    view = {"yaw": 0.30, "pitch": -0.20, "down": False, "lx": 0, "ly": 0}

    def on_mouse(evt, x, y, flags, _):
        rx = x - PW
        if evt == cv2.EVENT_LBUTTONDOWN and rx >= 0:
            view.update(down=True, lx=rx, ly=y)
        elif evt == cv2.EVENT_LBUTTONUP:
            view["down"] = False
        elif evt == cv2.EVENT_MOUSEMOVE and view["down"]:
            view["yaw"]   += (rx - view["lx"]) * 0.008
            view["pitch"]  = max(-1.4, min(1.4,
                              view["pitch"] + (y - view["ly"]) * 0.008))
            view["lx"], view["ly"] = rx, y

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]
            PW    = W // 2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            # ── Left panel: camera with landmarks ────────────────────────
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            cx        = W // 2
            cam_panel = frame[:, cx - PW // 2: cx + PW // 2].copy()
            cv2.putText(cam_panel, "Camera", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

            # ── Right panel: skin mesh ────────────────────────────────────
            twin = bg.copy()
            R    = _rot(view["yaw"], view["pitch"])

            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    pts3d, wrist_screen = _to_3d(hl, W / H)
                    rotated = (R @ pts3d.T).T
                    proj    = _project(rotated, PW, H, wrist_screen)
                    skin    = _render_skin(rotated, proj, H, PW)
                    twin    = cv2.add(twin, skin)

                n = len(res.multi_hand_landmarks)
                cv2.putText(twin, f"Hands detected: {n}", (10, H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)

            cv2.putText(twin, "3D Twin  (skin)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(twin, "Drag to rotate  |  Q = Quit", (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

            # ── Combine ───────────────────────────────────────────────────
            out = np.hstack([cam_panel, twin])
            cv2.line(out, (PW, 0), (PW, H), (55, 55, 55), 2)

            cv2.imshow(WIN, out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
