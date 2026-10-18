import glob
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
import wave

import cv2
import numpy as np
import requests

W, H = 1080, 1920
SR = 48000
WORK, OUT = pathlib.Path("work"), pathlib.Path("out")

URL = os.environ["VIDEO_URL"]
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or ""
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or ""
GROQ_KEY = os.environ.get("GROQ_API_KEY") or ""
GROQ_MODEL = os.environ.get("GROQ_WHISPER_MODEL") or "whisper-large-v3"
WHISPER = os.environ.get("WHISPER_MODEL") or "medium"
LANG = os.environ.get("LANG_CODE") or None
N_CLIPS = int(os.environ.get("NUM_CLIPS") or "5")
HOOK_MODE = (os.environ.get("HOOK_MODE") or "auto").lower()   # auto | always | never
MUSIC_MODE = (os.environ.get("MUSIC") or "on").lower()        # on | off
MUSIC_LEVEL = float(os.environ.get("MUSIC_LEVEL") or "0.12")
MIN_D, MAX_D = 25, 60
GAP = 0.45  # أي صمت أطول من هذا (بالثواني) يُحذف
API = f"https://api.telegram.org/bot{TOKEN}"


# ---------------------------------------------------------------- helpers
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


# ---------------------------------------------------------------- download
def download():
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    base = ["yt-dlp", "--no-playlist", "--merge-output-format", "mp4",
            "-o", str(WORK / "src.%(ext)s")]
    if pathlib.Path("cookies.txt").exists():
        base += ["--cookies", "cookies.txt"]
    attempts = [
        ["-f", "bv*[height<=1080]+ba/b[height<=1080]/b"],
        ["-f", "b", "--extractor-args", "youtube:player_client=tv,web_safari"],
        ["-f", "b", "--extractor-args", "youtube:player_client=android_vr"],
        ["-f", "b", "--extractor-args", "youtube:player_client=mweb"],
    ]
    last = "unknown error"
    for extra in attempts:
        cmd = base + extra + [URL]
        print("$", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        files = [f for f in WORK.glob("src.*") if f.suffix not in (".part", ".ytdl")]
        if r.returncode == 0 and files:
            f = files[0]
            if f.suffix != ".mp4":
                run_cmd(["ffmpeg", "-y", "-i", f, "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", "20", "-c:a", "aac", WORK / "src.mp4"])
            return WORK / "src.mp4"
        errs = [l for l in r.stderr.splitlines() if l.startswith("ERROR")]
        last = errs[-1] if errs else last
    raise RuntimeError(last)


# ---------------------------------------------------------------- transcription
def transcribe_groq(src):
    """Whisper large-v3 على سيرفرات Groq (مجاني، دقيق جداً للهجات)."""
    pattern = WORK / "chunk_%03d.mp3"
    for old in WORK.glob("chunk_*.mp3"):
        old.unlink()
    run_cmd(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "32k", "-f", "segment",
             "-segment_time", "900", "-reset_timestamps", "1", pattern])
    chunks_ = sorted(WORK.glob("chunk_*.mp3"))
    words, sentences, offset = [], [], 0.0
    for ch in chunks_:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(ch)],
            capture_output=True, text=True, check=True).stdout.strip())
        data = {"model": GROQ_MODEL, "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"], "temperature": "0"}
        if LANG:
            data["language"] = LANG
        j = None
        for attempt in range(6):
            with open(ch, "rb") as f:
                r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                                  headers={"Authorization": f"Bearer {GROQ_KEY}"},
                                  data=data, files={"file": (ch.name, f, "audio/mpeg")}, timeout=600)
            if r.status_code == 429:
                wait = min(float(r.headers.get("retry-after", "20")), 120)
                print("Groq rate limit, waiting", wait)
                time.sleep(wait + 1)
                continue
            r.raise_for_status()
            j = r.json()
            break
        if j is None:
            raise RuntimeError("Groq rate limit")
        for s in j.get("segments", []):
            sentences.append({"start": s["start"] + offset, "end": s["end"] + offset,
                              "text": s["text"].strip()})
        for w in j.get("words", []):
            if w["word"].strip():
                words.append({"w": w["word"].strip(), "s": w["start"] + offset,
                              "e": max(w["end"], w["start"] + 0.05) + offset})
        offset += dur
    if not words:
        raise RuntimeError("Groq returned no words")
    return words, sentences


def transcribe_local(src):
    from faster_whisper import WhisperModel
    audio = WORK / "audio.wav"
    run_cmd(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", audio])
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(audio), word_timestamps=True, vad_filter=True,
                               beam_size=5, language=LANG, condition_on_previous_text=False)
    words, sentences = [], []
    for s in segs:
        sentences.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in (s.words or []):
            if w.word.strip():
                words.append({"w": w.word.strip(), "s": w.start, "e": max(w.end, w.start + 0.05)})
    return words, sentences


def transcribe(src):
    if GROQ_KEY:
        try:
            print("Transcribing with Groq", GROQ_MODEL)
            return transcribe_groq(src)
        except Exception as e:  # noqa
            print("Groq failed, using local whisper:", e)
            say("⚠️ Groq لم يعمل، سأستخدم Whisper المحلي (أبطأ وأقل دقة).")
    print("Transcribing locally with", WHISPER)
    return transcribe_local(src)


# ---------------------------------------------------------------- clip selection
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


def gemini_models():
    if GEMINI_MODEL:
        return [GEMINI_MODEL]
    names = []
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"key": GEMINI_KEY, "pageSize": 200}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:  # noqa
        print("list models failed:", e)
    bad = ("lite", "image", "tts", "live", "audio", "embed", "vision", "robotics",
           "computer", "exp", "customtools", "thinking")
    flash = [n for n in names if "flash" in n and not any(x in n for x in bad)]
    stable = sorted([n for n in flash if "preview" not in n], reverse=True)
    prev = sorted([n for n in flash if "preview" in n], reverse=True)
    return (stable + prev)[:4] or ["gemini-2.5-flash"]


PROMPT = """You are an expert short-form video editor (TikTok / Reels / YouTube Shorts) working with a creator whose videos are in Arabic or English.
Below is a timestamped transcript (seconds). Pick the {n} best self-contained moments.

Rules:
- Each clip is {min_d}-{max_d} seconds. No overlaps.
- The clip must START on its strongest, most intriguing sentence (not mid-thought, no filler like "so/ok/well"), and END where the thought completes.
- content_type: "educational" (teaching, advice, explanation, knowledge), "important" (serious message, news, warning, deep life lesson), or "entertainment" (funny moments, streams, banter, stories).
- hook: ONLY for educational or important clips. A short on-screen line (max 9 words) stating the point/benefit of the clip so the viewer immediately knows why to keep watching. Same language as the transcript. For entertainment use "".
- teaser_start / teaser_end: OPTIONAL strongest standalone sentence inside the clip (3-9 seconds, at least 8 seconds after the clip start) that will be played FIRST as a cold open. Use null for both if the clip has no such line, or if it would spoil a joke.
- mood: calm, energetic, dramatic, or funny (for light background music).
- title: short caption title in the same language as the transcript. score: 0-100 viral potential.

Return ONLY a JSON array:
[{{"start":float,"end":float,"content_type":"...","hook":"...","teaser_start":float|null,"teaser_end":float|null,"mood":"...","title":"...","score":0}}]

TRANSCRIPT:
{lines}"""


def gemini_pick(sentences):
    lines = "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in sentences)
    prompt = PROMPT.format(n=N_CLIPS * 2, min_d=MIN_D, max_d=MAX_D, lines=lines)
    last = None
    for model in gemini_models():
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": GEMINI_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"responseMimeType": "application/json",
                                           "temperature": 0.4}},
                timeout=300,
            )
            r.raise_for_status()
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(txt)
            print("Gemini model used:", model)
            out = []
            for c in data:
                clip = snap({"start": float(c["start"]), "end": float(c["end"]),
                             "content_type": str(c.get("content_type") or "").lower(),
                             "hook": c.get("hook") or "", "title": c.get("title") or "",
                             "mood": str(c.get("mood") or "calm").lower(),
                             "score": c.get("score", 50)}, sentences)
                try:
                    ts_, te_ = c.get("teaser_start"), c.get("teaser_end")
                    if ts_ is not None and te_ is not None:
                        ts_, te_ = float(ts_), float(te_)
                        if clip["start"] + 6 <= ts_ and te_ <= clip["end"] and 2.5 <= te_ - ts_ <= 10:
                            clip["teaser"] = (ts_, te_)
                except (TypeError, ValueError):
                    pass
                out.append(clip)
            return out
        except Exception as e:  # noqa
            print("Gemini model", model, "failed:", e)
            last = e
    raise last or RuntimeError("no Gemini model available")


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
                          "content_type": "", "mood": "calm", "score": nwords / (end - start)})
    return cands


def pick_clips(sentences):
    cands = []
    if GEMINI_KEY:
        try:
            cands = gemini_pick(sentences)
        except Exception as e:  # noqa
            print("Gemini failed, using fallback:", e)
            say("⚠️ Gemini لم يعمل، سأختار المقاطع بطريقة تلقائية بسيطة.")
    if not cands:
        cands = fallback_pick(sentences)
    return non_overlapping(cands, N_CLIPS)


def face_center(src, start, dur):
    try:
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
    except Exception as e:  # noqa
        print("face detection unavailable, using center crop:", e)
    return 0.5, 0.4


# ---------------------------------------------------------------- captions + hook
def ts(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def esc(s):
    return s.replace("{", "(").replace("}", ")").replace("\n", " ")


def wrap_hook(text, maxc=24, maxlines=3):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > maxc:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\\N".join(esc(l) for l in lines[:maxlines])


def chunks(ws, maxc=26):
    out, cur, chars = [], [], 0
    for w in ws:
        if cur and chars + len(w["w"]) > maxc:
            out.append(cur)
            cur, chars = [], 0
        cur.append(w)
        chars += len(w["w"]) + 1
        if len(cur) >= 3 or re.search(r"[.!?؟،,:;]$", w["w"]):
            out.append(cur)
            cur, chars = [], 0
    if cur:
        out.append(cur)
    return out


def is_latin(words):
    txt = "".join(w["w"] for w in words)
    ar = sum(1 for c in txt if "\u0600" <= c <= "\u06ff")
    la = sum(1 for c in txt if c.isascii() and c.isalpha())
    return la > ar


def ass_head(latin):
    font = "Noto Sans" if latin else "Noto Sans Arabic"
    cap_size = 88 if latin else 98
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,{font},{cap_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,9,3,2,50,50,540,1
Style: Hook,{font},78,&H00FFFFFF,&H00FFFFFF,&H28101010,&H00000000,-1,0,0,0,100,100,0,0,3,26,0,8,60,60,300,1
Style: Bar,{font},40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def build_ass(segs, hook, total, path):
    ws, off = [], 0.0
    for sg in segs:
        for w in sg["words"]:
            ws.append({"w": w["w"], "nt": off + (w["s"] - sg["s"]), "ne": off + (w["e"] - sg["s"])})
        off += sg["e"] - sg["s"]

    latin = is_latin(ws)
    if latin:
        for x in ws:
            x["w"] = x["w"].upper()
    lines = [ass_head(latin)]
    for ch in chunks(ws, 16 if latin else 26):
        for i, w in enumerate(ch):
            end = ch[i + 1]["nt"] if i + 1 < len(ch) else max(w["ne"], w["nt"] + 0.2)
            parts = []
            for k, x in enumerate(ch):
                if k == i:
                    parts.append("{\\c&H00FFFF&\\fscx112\\fscy112}" + esc(x["w"]) +
                                 "{\\c&HFFFFFF&\\fscx100\\fscy100}")
                else:
                    parts.append(esc(x["w"]))
            lines.append(f"Dialogue: 0,{ts(w['nt'])},{ts(end)},Cap,,0,0,0,,{' '.join(parts)}")

    if hook:
        lines.append(f"Dialogue: 2,{ts(0.1)},{ts(min(4.2, total))},Hook,,0,0,0,,"
                     f"{{\\fad(120,280)\\fscx78\\fscy78\\t(0,180,\\fscx100\\fscy100)}}{wrap_hook(hook)}")

    ms = int(total * 1000)
    lines.append(f"Dialogue: 5,{ts(0)},{ts(total)},Bar,,0,0,0,,"
                 f"{{\\an7\\pos(0,1890)\\fscx0\\t(0,{ms},\\fscx100)\\1c&H00D7FF&\\bord0\\shad0\\p1}}"
                 f"m 0 0 l {W} 0 {W} 26 0 26{{\\p0}}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- sound design (synthesized, free)
def _note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


MOODS = {
    "calm": dict(chords=[[57, 60, 64], [53, 57, 60], [55, 60, 64], [55, 59, 62]],
                 chord_s=4.0, lp=1500, pluck=False, kick=False, bpm=60),
    "energetic": dict(chords=[[57, 60, 64], [53, 57, 60], [55, 60, 64], [55, 59, 62]],
                      chord_s=2.0, lp=2600, pluck=True, kick=True, bpm=110),
    "dramatic": dict(chords=[[50, 53, 57], [46, 50, 53], [53, 57, 60], [48, 52, 55]],
                     chord_s=4.5, lp=1200, pluck=False, kick=False, bpm=60),
    "funny": dict(chords=[[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]],
                  chord_s=2.0, lp=3500, pluck=True, kick=False, bpm=100),
}


def synth_music(dur, mood):
    p = MOODS.get(mood, MOODS["calm"])
    n = int(dur * SR)
    out = np.zeros(n)
    pos, k = 0.0, 0
    while pos < dur:
        chord = p["chords"][k % len(p["chords"])]
        seg = p["chord_s"]
        i0, i1 = int(pos * SR), min(n, int((pos + seg + 0.8) * SR))
        if i1 <= i0:
            break
        tt = np.arange(i0, i1) / SR
        local = tt - pos
        env = np.minimum(1.0, local / 0.5) * np.clip((seg + 0.8 - local) / 0.8, 0, 1)
        tone = np.zeros(len(tt))
        for m in chord:
            f = _note(m)
            for det in (-0.003, 0.0, 0.003):
                for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
                    tone += amp * np.sin(2 * np.pi * f * (1 + det) * h * tt)
        tone += 1.5 * np.sin(2 * np.pi * _note(chord[0] - 12) * tt)
        out[i0:i1] += tone * env * 0.05
        if p["pluck"]:
            step = 60.0 / p["bpm"] / 2
            t0 = pos
            idx = 0
            while t0 < pos + seg and t0 < dur:
                a = int(t0 * SR)
                b = min(n, a + int(0.4 * SR))
                lt = np.arange(b - a) / SR
                f = _note(chord[idx % 3] + 12)
                out[a:b] += 0.9 * np.sin(2 * np.pi * f * lt) * np.exp(-lt * 9) * 0.05
                t0 += step
                idx += 1
        pos += seg
        k += 1
    if p["kick"]:
        beat = 60.0 / p["bpm"]
        t0 = 0.0
        while t0 < dur:
            a = int(t0 * SR)
            b = min(n, a + int(0.25 * SR))
            lt = np.arange(b - a) / SR
            freq = 45 + 90 * np.exp(-lt * 30)
            out[a:b] += 1.6 * np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-lt * 14) * 0.08
            t0 += beat
    # low-pass in the frequency domain, then a soft reverb
    spec = np.fft.rfft(out)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= 1.0 / (1.0 + (freqs / p["lp"]) ** 4)
    out = np.fft.irfft(spec, n)
    rev = out.copy()
    for d, g in ((0.083, 0.35), (0.137, 0.25), (0.211, 0.18)):
        s = int(d * SR)
        rev[s:] += out[:-s] * g
    out = rev
    fade = int(0.8 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    peak = np.max(np.abs(out)) or 1.0
    out = out / peak
    left = out
    right = np.roll(out, int(0.012 * SR)) * 0.97
    return np.stack([left, right], axis=1)


def synth_whoosh():
    n = int(0.55 * SR)
    noise = np.random.default_rng(7).standard_normal(n)
    y = np.zeros(n)
    prev = 0.0
    for i in range(n):
        fc = 300 * (20 ** (i / n))
        a = 1 - math.exp(-2 * math.pi * fc / SR)
        prev += a * (noise[i] - prev)
        y[i] = prev
    x = np.linspace(0, 1, n)
    y *= np.sin(np.pi * x ** 0.8) ** 2
    y /= np.max(np.abs(y)) or 1.0
    return np.stack([y, y], axis=1)


def synth_pop():
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 700 * t) + 0.4 * np.sin(2 * np.pi * 1400 * t)) * np.exp(-t * 26)
    y /= np.max(np.abs(y)) or 1.0
    return np.stack([y, y], axis=1)


def read_wav(path):
    with wave.open(str(path)) as w:
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    data = data.reshape(-1, ch)
    if ch == 1:
        data = np.repeat(data, 2, axis=1)
    return data


def write_wav(path, data):
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def mix_audio(voice_path, out_path, mood, sfx_events):
    """voice + موسيقى هادئة تنخفض تلقائياً عند الكلام + مؤثرات انتقال خفيفة."""
    voice = read_wav(voice_path)
    n = len(voice)
    mix = voice.copy()

    if MUSIC_MODE != "off":
        music = synth_music(n / SR + 0.5, mood)[:n]
        # غلاف صوت المتحدث بمعدل 100 إطار/ثانية
        mono = np.abs(voice).mean(axis=1)
        hop = SR // 100
        frames = len(mono) // hop
        env = np.array([mono[i * hop:(i + 1) * hop].mean() for i in range(frames)])
        ref = np.percentile(env, 95) or 1.0
        env = np.clip(env / ref, 0, 1)
        sm, cur = np.zeros_like(env), 0.0
        for i, v in enumerate(env):          # attack سريع، release بطيء
            cur += (v - cur) * (0.5 if v > cur else 0.06)
            sm[i] = cur
        duck = np.interp(np.arange(n) / SR, np.arange(frames) / 100.0, sm)
        gain = MUSIC_LEVEL * (1.0 - 0.72 * duck)
        mix += music * gain[:, None]

    sounds = {"whoosh": synth_whoosh() * 0.30, "pop": synth_pop() * 0.28}
    for t, kind in sfx_events:
        s = sounds[kind]
        a = int(t * SR)
        b = min(n, a + len(s))
        if a < n and b > a:
            mix[a:b] += s[:b - a]

    peak = np.max(np.abs(mix))
    if peak > 0.97:
        mix *= 0.97 / peak
    write_wav(out_path, mix)


# ---------------------------------------------------------------- render
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


def show_hook(clip):
    hook = (clip.get("hook") or "").strip()
    ctype = clip.get("content_type", "")
    if HOOK_MODE == "never":
        return ""
    if HOOK_MODE == "always":
        return hook or (clip.get("title") or "").strip()
    return hook if ctype in ("educational", "important") else ""


def render(idx, clip, src, words, sw, sh):
    start, end = clip["start"], clip["end"]
    cw_words = [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.2]
    if not cw_words:
        return None
    segs_main = build_segments(cw_words, start)

    segs_teaser = []
    if clip.get("teaser"):
        t0, t1 = clip["teaser"]
        tw = [w for w in words if w["s"] >= t0 - 0.05 and w["e"] <= t1 + 0.2]
        if tw:
            segs_teaser = build_segments(tw, t0)
    segs = segs_teaser + segs_main
    total = sum(sg["e"] - sg["s"] for sg in segs)
    teaser_len = sum(sg["e"] - sg["s"] for sg in segs_teaser)

    fx, fy = face_center(src, start, end - start)
    if sw / sh >= 9 / 16:
        base_h, base_w = sh, sh * 9 / 16
    else:
        base_w, base_h = sw, sw * 16 / 9

    hook = show_hook(clip)
    ass = WORK / f"c{idx}.ass"
    build_ass(segs, hook, total, ass)

    parts, starts, off = [], [], 0.0
    for i, sg in enumerate(segs):
        f = 1.0 if i % 2 == 0 else 0.82  # تكبير (punch-in) في كل قطعة ثانية
        ch_, cw_ = int(base_h * f) // 2 * 2, int(base_w * f) // 2 * 2
        x = int(min(max(fx * sw - cw_ / 2, 0), sw - cw_)) // 2 * 2
        y = int(min(max(fy * sh - 0.38 * ch_, 0), sh - ch_)) // 2 * 2
        a, b = sg["s"] - start, sg["e"] - start
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,"
                     f"crop={cw_}:{ch_}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1,fps=30[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,aresample={SR}[a{i}]")
        starts.append(off)
        off += sg["e"] - sg["s"]
    n = len(segs)
    parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    parts.append(f"[vc]ass=filename={ass}[vo]")
    parts.append("[ac]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[an]")

    script = WORK / f"c{idx}.txt"
    script.write_text(";\n".join(parts), encoding="utf-8")

    vid = WORK / f"v{idx}.mp4"
    voice = WORK / f"voice{idx}.wav"
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start + 1:.3f}", "-i", WORK / "src.mp4",
           "-filter_complex_script", script,
           "-map", "[vo]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-t", f"{total:.2f}", vid,
           "-map", "[an]", "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "2", "-t", f"{total:.2f}", voice]
    run_cmd(cmd)

    # مؤثرات: واحد عند الانتقال من اللقطة القوية، وواحد للهوك، وهمسات خفيفة عند بعض القطعات
    sfx = []
    if hook:
        sfx.append((0.1, "pop"))
    if segs_teaser:
        sfx.append((max(teaser_len - 0.15, 0), "whoosh"))
    last = -9.0
    for i in range(len(segs_teaser) + 1, n, 2):
        if starts[i] - last >= 3.0 and len(sfx) < 9:
            sfx.append((max(starts[i] - 0.12, 0), "whoosh"))
            last = starts[i]

    mixed = WORK / f"mix{idx}.wav"
    mix_audio(voice, mixed, clip.get("mood", "calm"), sfx)

    out = OUT / f"short_{idx}.mp4"
    run_cmd(["ffmpeg", "-y", "-i", vid, "-i", mixed, "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
             "-movflags", "+faststart", out])
    return out


# ---------------------------------------------------------------- main
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
        say("❌ حدث خطأ: " + str(sys.exc_info()[1])[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()
