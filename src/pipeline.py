import glob
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import traceback

import cv2
import numpy as np
import requests
from faster_whisper import WhisperModel

W, H = 1080, 1920
WORK, OUT = pathlib.Path("work"), pathlib.Path("out")

URL = os.environ["VIDEO_URL"]
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or ""
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
WHISPER = os.environ.get("WHISPER_MODEL") or "small"
LANG = os.environ.get("LANG_CODE") or None
N_CLIPS = int(os.environ.get("NUM_CLIPS") or "5")
MIN_D, MAX_D = 25, 60
GAP = 0.45  # أي صمت أطول من هذا (بالثواني) يُحذف
API = f"https://api.telegram.org/bot{TOKEN}"


def run_cmd(cmd):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def say(text):
    try:
        requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": text}, timeout=30)
    except Exception as e:  # noqa
        print("telegram error:", e)


def send_video(path, caption):
    if path.stat().st_size > 49 * 1024 * 1024:
        say(f"⚠️ {path.name} أكبر من 50MB. حمّله من Artifacts في GitHub.")
        return
    with open(path, "rb") as f:
        r = requests.post(
            f"{API}/sendVideo",
            data={"chat_id": CHAT, "caption": caption[:1000], "supports_streaming": "true",
                  "width": W, "height": H},
            files={"video": f},
            timeout=600,
        )
    if not r.ok:
        print("send error:", r.text)


def download():
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    cmd = ["yt-dlp", "--no-playlist", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "--merge-output-format", "mp4", "-o", WORK / "src.%(ext)s"]
    if pathlib.Path("cookies.txt").exists():
        cmd += ["--cookies", "cookies.txt"]
    run_cmd(cmd + [URL])
    return WORK / "src.mp4"


def transcribe(src):
    audio = WORK / "audio.wav"
    run_cmd(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", audio])
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(audio), word_timestamps=True, vad_filter=True,
                               beam_size=1, language=LANG)
    words, sentences = [], []
    for s in segs:
        sentences.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in (s.words or []):
            if w.word.strip():
                words.append({"w": w.word.strip(), "s": w.start, "e": max(w.end, w.start + 0.05)})
    return words, sentences


def snap(clip, sentences):
    st = min(sentences, key=lambda s: abs(s["start"] - clip["start"]))["start"]
    en = min(sentences, key=lambda s: abs(s["end"] - clip["end"]))["end"]
    clip["start"], clip["end"] = st, en
    return clip


def non_overlapping(clips, n):
    chosen = []
    for c in sorted(clips, key=lambda c: -float(c.get("score", 0))):
        d = c["end"] - c["start"]
        if d < 15 or d > MAX_D + 15:
            continue
        if any(c["start"] < o["end"] and c["end"] > o["start"] for o in chosen):
            continue
        chosen.append(c)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda c: c["start"])


def gemini_pick(sentences):
    lines = "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in sentences)
    prompt = (
        "You are a viral short-form video editor. Below is a timestamped transcript "
        "(seconds). Pick the " + str(N_CLIPS * 2) + " best self-contained moments for "
        "TikTok/Reels/Shorts. Rules: each " + str(MIN_D) + "-" + str(MAX_D) + " seconds; "
        "start where a thought begins and end where it completes; strong hook in the first "
        "sentence; no overlaps. Return ONLY a JSON array: "
        '[{"start":float,"end":float,"hook":"punchy on-screen hook, max 8 words, '
        'in the same language as the transcript","title":"short caption title","score":0-100}]'
        "\n\nTRANSCRIPT:\n" + lines
    )
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}},
        timeout=300,
    )
    r.raise_for_status()
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(txt)
    return [snap({"start": float(c["start"]), "end": float(c["end"]),
                  "hook": c.get("hook", ""), "title": c.get("title", ""),
                  "score": c.get("score", 50)}, sentences) for c in data]


def fallback_pick(sentences):
    cands = []
    for i, s in enumerate(sentences):
        start, end, j = s["start"], s["end"], i
        while j < len(sentences) and sentences[j]["end"] - start <= 45:
            end = sentences[j]["end"]
            j += 1
        if end - start >= MIN_D:
            nwords = sum(len(x["text"].split()) for x in sentences[i:j])
            cands.append({"start": start, "end": end, "hook": "", "title": "",
                          "score": nwords / (end - start)})
    return cands


def pick_clips(sentences):
    cands = []
    if GEMINI_KEY:
        try:
            cands = gemini_pick(sentences)
        except Exception as e:  # noqa
            print("Gemini failed, using fallback:", e)
    if not cands:
        cands = fallback_pick(sentences)
    return non_overlapping(cands, N_CLIPS)


def face_center(src, start, dur):
    cap = cv2.VideoCapture(str(src))
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    xs, ys = [], []
    for t in np.arange(start, start + dur, 1.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        faces = det.detectMultiScale(g, 1.2, 5, minSize=(w // 20, w // 20))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            xs.append((x + fw / 2) / w)
            ys.append((y + fh / 2) / h)
    cap.release()
    if xs:
        return float(np.median(xs)), float(np.median(ys))
    return 0.5, 0.4


def ts(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def esc(s):
    return s.replace("{", "(").replace("}", ")").replace("\n", " ")


def chunks(ws):
    out, cur = [], []
    for w in ws:
        cur.append(w)
        if len(cur) >= 3 or re.search(r"[.!?؟،,:;]$", w["w"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


ASS_HEAD = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,Noto Sans Arabic,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,3,2,60,60,520,1
Style: Hook,Noto Sans Arabic,68,&H00FFFFFF,&H00FFFFFF,&HC8000000,&H00000000,-1,0,0,0,100,100,0,0,3,22,0,8,70,70,230,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def build_ass(segs, hook, total, path):
    ws, off = [], 0.0
    for sg in segs:
        for w in sg["words"]:
            ws.append({"w": w["w"], "nt": off + (w["s"] - sg["s"]), "ne": off + (w["e"] - sg["s"])})
        off += sg["e"] - sg["s"]

    lines = [ASS_HEAD]
    for ch in chunks(ws):
        for i, w in enumerate(ch):
            end = ch[i + 1]["nt"] if i + 1 < len(ch) else max(w["ne"], w["nt"] + 0.2)
            parts = []
            for k, x in enumerate(ch):
                if k == i:
                    parts.append("{\\c&H00FFFF&\\fscx115\\fscy115}" + esc(x["w"]) +
                                 "{\\c&HFFFFFF&\\fscx100\\fscy100}")
                else:
                    parts.append(esc(x["w"]))
            lines.append(f"Dialogue: 0,{ts(w['nt'])},{ts(end)},Cap,,0,0,0,,{' '.join(parts)}")

    if hook:
        lines.append(f"Dialogue: 1,{ts(0.15)},{ts(min(3.8, total))},Hook,,0,0,0,,"
                     f"{{\\fad(200,300)}}{esc(hook)}")

    ms = int(total * 1000)
    lines.append(f"Dialogue: 5,{ts(0)},{ts(total)},Cap,,0,0,0,,"
                 f"{{\\an7\\pos(0,1890)\\fscx0\\t(0,{ms},\\fscx100)\\1c&H00D7FF&\\bord0\\shad0\\p1}}"
                 f"m 0 0 l {W} 0 {W} 26 0 26{{\\p0}}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_segments(cw, start):
    segs = []
    for w in cw:
        if segs and w["s"] - segs[-1]["last_e"] <= GAP:
            segs[-1]["words"].append(w)
            segs[-1]["last_e"] = w["e"]
        else:
            segs.append({"words": [w], "last_e": w["e"]})
    prev_end = start
    for sg in segs:
        sg["s"] = max(prev_end, sg["words"][0]["s"] - 0.10)
        sg["e"] = sg["last_e"] + 0.15
        prev_end = sg["e"]
    return segs


def render(idx, clip, src, words, sw, sh):
    start, end = clip["start"], clip["end"]
    cw_words = [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.2]
    if not cw_words:
        return None
    segs = build_segments(cw_words, start)
    total = sum(sg["e"] - sg["s"] for sg in segs)

    fx, fy = face_center(src, start, end - start)
    if sw / sh >= 9 / 16:
        base_h, base_w = sh, sh * 9 / 16
    else:
        base_w, base_h = sw, sw * 16 / 9

    ass = WORK / f"c{idx}.ass"
    build_ass(segs, clip.get("hook", ""), total, ass)

    parts = []
    for i, sg in enumerate(segs):
        f = 1.0 if i % 2 == 0 else 0.82  # تكبير (punch-in) في كل قطعة ثانية
        ch_, cw_ = int(base_h * f) // 2 * 2, int(base_w * f) // 2 * 2
        x = int(min(max(fx * sw - cw_ / 2, 0), sw - cw_)) // 2 * 2
        y = int(min(max(fy * sh - 0.38 * ch_, 0), sh - ch_)) // 2 * 2
        a, b = sg["s"] - start, sg["e"] - start
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,"
                     f"crop={cw_}:{ch_}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1,fps=30[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,aresample=48000[a{i}]")
    n = len(segs)
    parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    parts.append(f"[vc]ass=filename={ass}[vo]")

    music = glob.glob("assets/music/*.mp3")
    if music:
        parts.append("[ac]loudnorm=I=-16:TP=-1.5:LRA=11[an]")
        parts.append("[1:a]volume=0.10,aresample=48000[m]")
        parts.append("[an][m]amix=inputs=2:duration=longest:dropout_transition=0,volume=1.8[ao]")
    else:
        parts.append("[ac]loudnorm=I=-16:TP=-1.5:LRA=11[ao]")

    script = WORK / f"c{idx}.txt"
    script.write_text(";\n".join(parts), encoding="utf-8")

    out = OUT / f"short_{idx}.mp4"
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start + 1:.3f}", "-i", WORK / "src.mp4"]
    if music:
        cmd += ["-stream_loop", "-1", "-i", random.choice(music)]
    cmd += ["-filter_complex_script", script, "-map", "[vo]", "-map", "[ao]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.2f}", "-movflags", "+faststart", out]
    run_cmd(cmd)
    return out


def main():
    try:
        say("⬇️ جاري تحميل الفيديو...")
        src = download()
        cap = cv2.VideoCapture(str(src))
        sw, sh_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        say("🎙️ تفريغ الصوت (قد يستغرق وقتاً)...")
        words, sentences = transcribe(src)
        if not sentences:
            say("❌ لم أجد كلاماً في الفيديو.")
            sys.exit(1)

        say("🧠 اختيار أقوى المقاطع...")
        clips = pick_clips(sentences)
        if not clips:
            say("❌ لم أستطع اختيار مقاطع مناسبة.")
            sys.exit(1)

        for i, clip in enumerate(clips, 1):
            say(f"🎬 مونتاج المقطع {i}/{len(clips)}...")
            out = render(i, clip, src, words, sw, sh_)
            if out:
                send_video(out, clip.get("title") or clip.get("hook") or f"Short {i}")
        say("✅ انتهى! هذه شورتاتك.")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        say("❌ حدث خطأ أثناء المعالجة. راجع سجل Actions في GitHub.")
        sys.exit(1)


if __name__ == "__main__":
    main()
