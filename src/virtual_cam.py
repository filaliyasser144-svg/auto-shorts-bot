#!/usr/bin/env python3
"""
virtual_cam.py - Virtual Camera Operator

Turns a horizontal clip into a 9:16 clip whose "camera" follows the active
speaker like a real operator would:
  * detects every face + mouth movement, matches it with speech in the audio
  * picks the active speaker with hysteresis (no flickering between people)
  * starts the move slightly BEFORE the new speaker talks (editor-style)
  * smooth follow + dead-zone (small head moves are ignored)
  * fast re-frame for nearby speakers, hard cut for far-apart speakers
  * two-shot when both people fit inside the vertical crop
  * safety clamp: the speaker's face can never leave the frame

Usage:
    python virtual_cam.py in.mp4 out.mp4 --start 125 --end 175
    python virtual_cam.py in.mp4 out.mp4 --debug     # also writes out.mp4.debug.mp4
"""
import argparse
import math
import os
import subprocess
import tempfile
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

OUT_W, OUT_H = 1080, 1920

# ---- tunable settings (use --debug to see their effect) ---------------------
ANALYZE_FPS = 10      # face analysis rate (frames per second)
MIN_SCORE = 0.025     # mouth movement needed to count as "talking"
SWITCH_HOLD = 0.6     # seconds a new speaker must lead before we switch
MIN_SHOT = 1.6        # minimum seconds between two switches
LEAD = 0.25           # camera starts moving this many seconds before first word
DEADZONE = 0.05       # fraction of crop width where tiny motion is ignored
SAFE_ZONE = 0.35      # speaker's face stays within +-35% of crop width from center
MIN_FACE = 0.035      # ignore faces narrower than this fraction of frame width


# ---- helpers ----------------------------------------------------------------
def cut_clip(src, start, end, dst):
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if end:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-c:a", "aac", dst]
    subprocess.run(cmd, check=True)


def crop_size(W, H, zoom):
    base_h = min(H, W * 16 / 9)
    ch = base_h / zoom
    return ch * 9 / 16, ch


def smooth_damp(cur, tgt, vel, smooth_time, dt, max_speed):
    omega = 2.0 / smooth_time
    x = omega * dt
    e = 1.0 / (1 + x + 0.48 * x * x + 0.235 * x ** 3)
    change = cur - tgt
    max_change = max_speed * smooth_time
    change = max(-max_change, min(max_change, change))
    t2 = cur - change
    temp = (vel + omega * change) * dt
    vel = (vel - omega * temp) * e
    out = t2 + (change + temp) * e
    if (tgt - cur > 0) == (out > tgt):
        out, vel = tgt, 0.0
    return out, vel


# ---- 1. audio: is someone speaking? ----------------------------------------
def speech_activity(clip, n, dt):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-vn", "-ac", "1",
         "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.float32)
    win = int(16000 * dt)
    if a.size < win * 3:
        return np.ones(n, bool)
    m = a.size // win
    rms = np.sqrt((a[:m * win].reshape(m, win) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    floor, peak = np.percentile(db, 5), np.percentile(db, 95)
    if peak - floor < 12:                      # no real silence in this clip: trust the mouths only
        return np.ones(n, bool)
    sp = db > peak - 25                        # speech is rarely >25 dB below its own peak
    sp = np.convolve(sp.astype(float), np.ones(3), "same") >= 1  # bridge tiny gaps
    out = np.zeros(n, bool)
    k = min(n, sp.size)
    out[:k] = sp[:k]
    return out


# ---- 2. faces + mouth movement ----------------------------------------------
MODEL_URL = os.getenv(
    "FACE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task")
MODEL_PATH = os.getenv("FACE_MODEL_PATH", os.path.join("models", "face_landmarker.task"))


class MediapipeDetector:
    """Face + lip landmarks with MediaPipe Tasks (FaceLandmarker).
    Called as detector(bgr_frame, t_ms, W, H) -> list of dicts (cx, cy, fw, fh, mar)."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
            print("downloading face model ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_faces=4,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.lm = mp_vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, bgr, t_ms, W, H):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(img, int(t_ms))
        dets = []
        for L in res.face_landmarks:
            xs = np.array([p.x for p in L]) * W
            ys = np.array([p.y for p in L]) * H
            mar = (abs(L[13].y - L[14].y) * H) / (abs(L[61].x - L[291].x) * W + 1e-6)
            dets.append(dict(cx=(xs.min() + xs.max()) / 2, cy=(ys.min() + ys.max()) / 2,
                             fw=xs.max() - xs.min(), fh=ys.max() - ys.min(), mar=mar))
        return dets


def analyze(clip, detector=None):
    detector = detector or MediapipeDetector()
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    if W == 0 or H == 0:
        raise RuntimeError(f"cannot read video: {clip}")
    stride = max(1, round(fps / ANALYZE_FPS))
    dt = stride / fps
    scale = min(1.0, 640 / W)

    tracks, next_id, samples, idx = {}, 0, [], 0
    while True:
        if idx % stride:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, None, fx=scale, fy=scale) if scale < 1 else fr
        t = idx / fps
        dets = [d for d in detector(small, t * 1000, W, H) if d["fw"] >= MIN_FACE * W]

        free = set(tracks)
        faces = []
        for d in sorted(dets, key=lambda d: -d["fw"]):
            best, bd = None, 0.2 * W
            for tid in free:
                dist = math.hypot(d["cx"] - tracks[tid]["cx"], d["cy"] - tracks[tid]["cy"])
                if dist < bd:
                    best, bd = tid, dist
            if best is None:
                best = next_id
                next_id += 1
                tracks[best] = {"hist": deque(maxlen=7)}
            else:
                free.discard(best)
            tr = tracks[best]
            tr.update(cx=d["cx"], cy=d["cy"], seen=t)
            tr["hist"].append(d["mar"])
            score = float(np.std(tr["hist"])) if len(tr["hist"]) >= 4 else 0.0
            faces.append(dict(id=best, cx=d["cx"], cy=d["cy"],
                              fw=d["fw"], fh=d["fh"], score=score))
        for tid in list(tracks):
            if t - tracks[tid]["seen"] > 2.0:
                del tracks[tid]
        samples.append(dict(t=t, faces=faces))
        idx += 1
    cap.release()
    return samples, fps, W, H, dt, stride


# ---- 3. who is the active speaker? ------------------------------------------
def decide_speakers(samples, speech, dt):
    n = len(samples)
    spk = [None] * n
    lead = int(round(LEAD / dt))
    hold, shot = SWITCH_HOLD / dt, MIN_SHOT / dt
    cur, last_i = None, -10 ** 9
    cand, cand_i, cand_seen = None, 0, 0

    for i, s in enumerate(samples):
        faces = {f["id"]: f for f in s["faces"]}
        best = None
        if speech[i]:
            sc = [f for f in s["faces"] if f["score"] >= MIN_SCORE]
            if sc:
                best = max(sc, key=lambda f: f["score"])["id"]

        if cur is None:
            if best is not None:
                cur, last_i = best, i
                for j in range(i + 1):          # fill the start with the first speaker
                    if spk[j] is None:
                        spk[j] = cur
        elif cur not in faces:                  # current speaker left the frame
            pick = best
            if pick is None and faces:
                pick = max(faces.values(), key=lambda f: f["fw"])["id"]
            if pick is not None:
                cur, last_i, cand = pick, i, None
        elif best is not None and best != cur:
            if cand != best or i - cand_seen > 0.5 / dt:
                cand, cand_i = best, i
            cand_seen = i
            if i - cand_i >= hold and i - last_i >= shot:
                for j in range(max(cand_i - lead, last_i), i + 1):
                    spk[j] = best               # move starts BEFORE the first word
                cur, last_i, cand = best, i, None
        elif best == cur:
            cand = None
        if spk[i] is None:
            spk[i] = cur
    return spk


# ---- 4. where should the camera look? ---------------------------------------
def build_targets(samples, spk, W, H, cw, ch):
    times, tx, ty, last = [], [], [], (W / 2, H / 2)
    for s, sp in zip(samples, spk):
        faces = s["faces"]
        if faces:
            main = next((f for f in faces if f["id"] == sp), None) \
                or max(faces, key=lambda f: f["fw"])
            x, y = main["cx"], main["cy"]
            if len(faces) >= 2:                 # two-shot if everyone fits
                l = min(f["cx"] - f["fw"] for f in faces)
                r = max(f["cx"] + f["fw"] for f in faces)
                if r - l <= cw * 0.92:
                    x = (l + r) / 2
            last = (x, y)
        x, y = last
        times.append(s["t"])
        tx.append(x)
        ty.append(y + 0.10 * ch if ch < H - 1 else H / 2)   # headroom
    return np.array(times), np.array(tx), np.array(ty)


# ---- 5. the "camera operator" motion ----------------------------------------
def simulate(n, fps, times, tx, ty, W, H, cw, ch):
    ft = np.arange(n) / fps
    X, Y = np.interp(ft, times, tx), np.interp(ft, times, ty)
    dt = 1 / fps
    cx, cy, vx, vy = X[0], Y[0], 0.0, 0.0
    cams = []
    for i in range(n):
        err = X[i] - cx
        if abs(err) > 1.1 * cw:                 # far speaker -> hard cut
            cx, vx = X[i], 0.0
        else:
            aim = cx if abs(err) <= DEADZONE * cw else X[i] - math.copysign(DEADZONE * cw, err)
            st = 0.20 if abs(err) > 0.3 * cw else 0.45   # fast re-frame, slow drift
            cx, vx = smooth_damp(cx, aim, vx, st, dt, cw * 4)
        cx = min(max(cx, X[i] - SAFE_ZONE * cw), X[i] + SAFE_ZONE * cw)   # never lose them
        cy, vy = smooth_damp(cy, Y[i], vy, 0.6, dt, ch * 2)
        cx = min(max(cx, cw / 2), W - cw / 2)
        cy = min(max(cy, ch / 2), H - ch / 2)
        cams.append((cx, cy))
    return cams


# ---- 6. render ----------------------------------------------------------------
def render(clip, out, cams, cw, ch, fps):
    cap = cv2.VideoCapture(clip)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", str(fps), "-i", "-", "-i", clip,
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out], stdin=subprocess.PIPE)
    s, i = OUT_H / ch, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cx, cy = cams[min(i, len(cams) - 1)]
        x0, y0 = cx - cw / 2, cy - ch / 2
        M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])   # sub-pixel = no jitter
        ff.stdin.write(cv2.warpAffine(fr, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC).tobytes())
        i += 1
    ff.stdin.close()
    ff.wait()
    cap.release()


def debug_video(clip, samples, spk, cams, cw, ch, fps, stride, path):
    """Source video with face boxes (green = active speaker) and the crop window."""
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    k = 960 / W
    size = (960, int(H * k))
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        j = min(i // stride, len(samples) - 1)
        for f in samples[j]["faces"]:
            col = (0, 255, 0) if f["id"] == spk[j] else (0, 165, 255)
            x1, y1 = int(f["cx"] - f["fw"] / 2), int(f["cy"] - f["fh"] / 2)
            x2, y2 = int(f["cx"] + f["fw"] / 2), int(f["cy"] + f["fh"] / 2)
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 3)
            cv2.putText(fr, f'#{f["id"]} {f["score"]:.3f}', (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cx, cy = cams[min(i, len(cams) - 1)]
        cv2.rectangle(fr, (int(cx - cw / 2), int(cy - ch / 2)),
                      (int(cx + cw / 2), int(cy + ch / 2)), (255, 255, 255), 4)
        wr.write(cv2.resize(fr, size))
        i += 1
    wr.release()
    cap.release()


def reframe(src, dst, start=None, end=None, zoom=1.0, debug=False, detector=None):
    """Full pipeline: cut -> analyze faces/speech -> virtual camera -> 9:16 render."""
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        cut_clip(src, start, end, clip)

        samples, fps, W, H, dt, stride = analyze(clip, detector)
        if not samples:
            raise RuntimeError("no frames read")
        cw, ch = crop_size(W, H, zoom)
        speech = speech_activity(clip, len(samples), dt)
        spk = decide_speakers(samples, speech, dt)
        times, tx, ty = build_targets(samples, spk, W, H, cw, ch)

        cap = cv2.VideoCapture(clip)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        cams = simulate(n, fps, times, tx, ty, W, H, cw, ch)

        render(clip, dst, cams, cw, ch, fps)
        if debug:
            debug_video(clip, samples, spk, cams, cw, ch, fps, stride, dst + ".debug.mp4")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--zoom", type=float, default=1.0, help="1.0 = widest crop; 1.2 = tighter")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    reframe(a.src, a.dst, a.start, a.end, a.zoom, a.debug)
    print("done:", a.dst)


if __name__ == "__main__":
    main()
