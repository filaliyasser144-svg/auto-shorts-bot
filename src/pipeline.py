import base64
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import requests

W, H = 1080, 1920
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
HOOK_MODE = (os.environ.get("HOOK_MODE") or "always").lower()  # always | auto | never
TEXT_FIX = (os.environ.get("TEXT_FIX") or "on").lower()        # on | off
MIN_D, MAX_D = 25, 60
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


# ---------------------------------------------------------------- transcription (first pass)
def transcribe_groq(src):
    """Whisper large-v3 على سيرفرات Groq (مجاني)."""
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


# ---------------------------------------------------------------- Gemini
_NAMES = []


def gemini_models(pro=False):
    if GEMINI_MODEL:
        return [GEMINI_MODEL]
    if not _NAMES:
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"key": GEMINI_KEY, "pageSize": 200}, timeout=30)
            r.raise_for_status()
            _NAMES.extend(m["name"].split("/")[-1] for m in r.json().get("models", [])
                          if "generateContent" in m.get("supportedGenerationMethods", []))
        except Exception as e:  # noqa
            print("list models failed:", e)
    bad = ("lite", "image", "tts", "live", "audio", "embed", "vision", "robotics",
           "computer", "exp", "customtools", "thinking")
    ok = [n for n in _NAMES if not any(x in n for x in bad)]

    def order(xs):
        return (sorted([n for n in xs if "preview" not in n], reverse=True) +
                sorted([n for n in xs if "preview" in n], reverse=True))

    flash = order([n for n in ok if "flash" in n])[:3]
    pros = order([n for n in ok if "-pro" in n])[:2] if pro else []
    return pros + flash or ["gemini-2.5-flash"]


def gemini_generate(parts, temperature=0.4, pro=False):
    last = None
    models = gemini_models(pro)
    for mi, model in enumerate(models):
        for attempt in range(2):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": GEMINI_KEY},
                    json={"contents": [{"parts": parts}],
                          "generationConfig": {"responseMimeType": "application/json",
                                               "temperature": temperature}},
                    timeout=300,
                )
                if r.status_code == 429 and attempt == 0 and mi == len(models) - 1:
                    time.sleep(25)
                    continue
                r.raise_for_status()
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(txt)
                print("Gemini model used:", model)
                return data
            except Exception as e:  # noqa
                print("Gemini model", model, "failed:", e)
                last = e
                break
    raise last or RuntimeError("no Gemini model available")


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


PROMPT = """You are an expert short-form video editor (TikTok / Reels / YouTube Shorts) working with a creator whose videos are in Arabic or English.
Below is a timestamped transcript (seconds). It is automatic and may contain mistakes. Pick the {n} best self-contained moments.

Rules:
- Each clip is {min_d}-{max_d} seconds. No overlaps.
- The clip must START on its strongest, most intriguing sentence (not mid-thought, no filler like "so/ok/well"), and END where the thought completes.
- content_type: "educational" (teaching, advice, explanation, knowledge), "important" (serious message, news, warning, deep life lesson), or "entertainment" (funny moments, streams, banter, stories).
- hook: ONLY for educational or important clips. A bold on-screen title of 3-7 words in the same language as the transcript that tells the viewer the clip's point or promise and makes them want to keep watching (a strong question, a surprising claim, or a clear benefit). Never misleading. For entertainment use "".
- teaser_start / teaser_end: OPTIONAL strongest standalone sentence inside the clip (3-9 seconds, at least 8 seconds after the clip start) that will be played FIRST as a cold open. Use null for both if the clip has no such line, or if it would spoil a joke.
- title: short caption title in the same language as the transcript. score: 0-100 viral potential.

Return ONLY a JSON array:
[{{"start":float,"end":float,"content_type":"...","hook":"...","teaser_start":float|null,"teaser_end":float|null,"title":"...","score":0}}]

TRANSCRIPT:
{lines}"""


def gemini_pick(sentences):
    lines = "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in sentences)
    prompt = PROMPT.format(n=N_CLIPS * 2, min_d=MIN_D, max_d=MAX_D, lines=lines)
    data = gemini_generate([{"text": prompt}])
    out = []
    for c in data:
        clip = snap({"start": float(c["start"]), "end": float(c["end"]),
                     "content_type": str(c.get("content_type") or "").lower(),
                     "hook": c.get("hook") or "", "title": c.get("title") or "",
                     "score": c.get("score", 50)}, sentences)
        out.append(clip)
    return out


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
                          "content_type": "", "score": nwords / (end - start)})
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


# ---------------------------------------------------------------- text correction (listen to the audio)
def norm_tok(t):
    t = re.sub(r"[\u064b-\u0652\u0640]", "", t)
    t = t.translate(str.maketrans("أإآٱ", "اااا")).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w]", "", t.lower())


def place(wsub, toks, t0, t1, start_strict=True):
    """يوزّع كلمات النص الصحيح على زمن الكلام الفعلي (مع الحفاظ على فترات الصمت)."""
    if not toks:
        return []
    if wsub:
        runs = []
        for w in wsub:
            if runs and w["s"] - runs[-1][1] <= 0.3:
                runs[-1][1] = max(runs[-1][1], w["e"])
            else:
                runs.append([w["s"], w["e"]])
    else:
        runs = [[t0, max(t1, t0 + 0.12 * len(toks))]]
    total = sum(e - s for s, e in runs) or 0.1

    def at(x, is_start):
        for s, e in runs:
            if x < e - s or (not is_start and x <= e - s):
                return s + x
            x -= e - s
        return runs[-1][1]

    wts = [len(t) + 1 for t in toks]
    sum_w, c, out = sum(wts), 0, []
    for t, wt in zip(toks, wts):
        s_ = at(c / sum_w * total, True)
        e_ = at((c + wt) / sum_w * total, False)
        c += wt
        out.append({"w": t, "s": s_, "e": max(e_, s_ + 0.05)})
    return out


def align_words(ws, new_tokens):
    a = [norm_tok(w["w"]) for w in ws]
    b = [norm_tok(t) for t in new_tokens]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    anchors = [(i, j, n) for i, j, n in sm.get_matching_blocks()
               if n > 0 and (n >= 2 or len(b[j]) >= 4)]
    anchors.append((len(ws), len(new_tokens), 0))
    res, pa, pb = [], 0, 0
    prev_end = ws[0]["s"]
    for i, j, n in anchors:
        next_start = ws[i]["s"] if i < len(ws) else ws[-1]["e"]
        res += place(ws[pa:i], new_tokens[pb:j], prev_end, next_start)
        for k in range(n):
            res.append({"w": new_tokens[j + k], "s": ws[i + k]["s"], "e": ws[i + k]["e"]})
        if n:
            prev_end = ws[i + n - 1]["e"]
        elif res:
            prev_end = res[-1]["e"]
        pa, pb = i + n, j + n
    last_e = 0.0
    for w in res:  # ترتيب زمني سليم بدون تداخل
        w["s"] = max(w["s"], last_e)
        w["e"] = max(w["e"], w["s"] + 0.05)
        last_e = w["e"]
    return res


FIX_PROMPT = """Listen carefully to this audio and write EXACTLY what is spoken, word for word.
The speakers use Arabic (any dialect, especially Algerian Darija and other Maghrebi dialects), English, and often mix in French or English words.
Rules:
- Arabic speech: write it in Arabic script exactly as pronounced, keeping the speakers' own dialect words. Algerian Darija stays Darija: do NOT convert it to Modern Standard Arabic, do NOT translate it, do NOT "correct" the grammar. No diacritics.
- French or English words said inside the speech: write them in Latin letters, correctly spelled.
- English speech: normal spelling.
- Two or more people may speak. Write everything every speaker says, in order. Do not label speakers.
- Do not summarize, do not skip, do not repeat, do not add words.
- The draft below is an automatic transcript with MANY mistakes. Use it only as a hint for names and terms; trust the audio.
Return ONLY JSON: {{"text": "the exact words spoken"}}

DRAFT:
{draft}"""


def fix_text(idx, clip, words, src):
    """يصحح نص المقطع بالاستماع للصوت مباشرة، ثم يحافظ على التوقيت الأصلي."""
    if not GEMINI_KEY or TEXT_FIX == "off":
        return words
    a = max(clip["start"] - 0.3, 0.0)
    b = clip["end"] + 0.5
    ws = [w for w in words if w["s"] >= a and w["e"] <= b]
    if len(ws) < 5:
        return words
    try:
        audio = WORK / f"fix{idx}.mp3"
        run_cmd(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src,
                 "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", audio])
        data = base64.b64encode(audio.read_bytes()).decode()
        prompt = FIX_PROMPT.format(draft=" ".join(w["w"] for w in ws))
        res = gemini_generate([{"text": prompt},
                               {"inline_data": {"mime_type": "audio/mp3", "data": data}}],
                              temperature=0, pro=True)
        text = res.get("text", "") if isinstance(res, dict) else ""
        toks = text.split()
        if not (0.5 * len(ws) <= len(toks) <= 1.8 * len(ws)):
            print("text fix rejected: word count", len(toks), "vs", len(ws))
            return words
        fixed = align_words(ws, toks)
        rest = [w for w in words if not (w["s"] >= a and w["e"] <= b)]
        return sorted(rest + fixed, key=lambda w: w["s"])
    except Exception as e:  # noqa
        print("text fix failed, keeping original text:", e)
        return words


# ---------------------------------------------------------------- faces + camera shots + speaker following
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DET_W = 800        # عرض الصورة المستخدمة للكشف (أسرع وكافٍ)


class FaceFinder:
    """يكشف الوجوه: YuNet (دقيق مع الزوايا والوجوه الصغيرة)، وإلا Haar."""

    def __init__(self):
        self.yunet = None
        self.haar = []
        print("cv2", getattr(cv2, "__version__", "?"), "| CascadeClassifier:", hasattr(cv2, "CascadeClassifier"),
              "| FaceDetectorYN:", hasattr(cv2, "FaceDetectorYN"), flush=True)
        try:
            model = WORK / "yunet.onnx"
            if not model.exists() or model.stat().st_size < 100_000:
                r = requests.get(YUNET_URL, timeout=60)
                r.raise_for_status()
                model.write_bytes(r.content)
            self.yunet = cv2.FaceDetectorYN.create(str(model), "", (DET_W, DET_W), 0.6, 0.3, 5000)
            print("face detector: YuNet", flush=True)
        except Exception as e:  # noqa
            print("YuNet unavailable:", e, flush=True)
        if self.yunet is None:
            try:
                for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_profileface.xml"):
                    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                    if not c.empty():
                        self.haar.append((name, c))
                print("face detector: Haar x", len(self.haar), flush=True)
            except Exception as e:  # noqa
                print("Haar unavailable:", e, flush=True)

    @property
    def ok(self):
        return self.yunet is not None or bool(self.haar)

    def detect(self, img, thr=0.6):
        """وجوه الصورة: [(cx, cy, w, h, mouth_x, mouth_y)] بإحداثيات الصورة."""
        h, w = img.shape[:2]
        out = []
        try:
            if self.yunet is not None:
                self.yunet.setInputSize((w, h))
                self.yunet.setScoreThreshold(thr)
                _, faces = self.yunet.detect(img)
                for f in (faces if faces is not None else []):
                    x, y, fw, fh = [float(v) for v in f[:4]]
                    mx, my = (float(f[10]) + float(f[12])) / 2, (float(f[11]) + float(f[13])) / 2
                    out.append((x + fw / 2, y + fh / 2, fw, fh, mx, my))
            elif self.haar:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                m = (max(w // 30, 24),) * 2
                raw = []
                for name, c in self.haar:
                    raw += [tuple(b) for b in c.detectMultiScale(g, 1.1, 5, minSize=m)]
                    if "profile" in name:
                        raw += [(w - x - fw, y, fw, fh) for x, y, fw, fh in
                                c.detectMultiScale(cv2.flip(g, 1), 1.1, 5, minSize=m)]
                for x, y, fw, fh in sorted(raw, key=lambda b: -b[2]):
                    cx, cy = x + fw / 2, y + fh / 2
                    if all(abs(cx - o[0]) > 0.5 * max(fw, o[2]) or abs(cy - o[1]) > 0.5 * max(fh, o[3])
                           for o in out):
                        out.append((cx, cy, float(fw), float(fh), cx, y + 0.78 * fh))
        except Exception as e:  # noqa
            print("face detect error:", e, flush=True)
        return out


def merge_slots(det_lists):
    """يحوّل وجوه عدة إطارات إلى 'خانات' ثابتة، خانة لكل شخص."""
    slots = []
    for faces in det_lists:
        for f in faces:
            for sl in slots:
                if (abs(f[0] - sl["v"][0]) < 0.6 * max(f[2], sl["v"][2]) and
                        abs(f[1] - sl["v"][1]) < 0.6 * max(f[3], sl["v"][3])):
                    sl["v"] = [(a * sl["n"] + b) / (sl["n"] + 1) for a, b in zip(sl["v"], f)]
                    sl["n"] += 1
                    break
            else:
                slots.append({"v": list(f), "n": 1})
    if not slots:
        return []
    need = 2 if len(det_lists) >= 3 else 1
    keep = [sl for sl in slots if sl["n"] >= need] or [max(slots, key=lambda sl: sl["n"])]
    big = max(sl["v"][2] for sl in keep)
    keep = [sl for sl in keep if sl["v"][2] >= 0.35 * big]      # تجاهل الوجوه الصغيرة جداً في الخلفية
    keep.sort(key=lambda sl: sl["v"][0])
    return [sl["v"] for sl in keep][:4]


_FACE_CALLS = [0]

FACES_PROMPT = """Detect every human face visible in this image (front, side or partly turned, small or large).
Return ONLY JSON: {"faces": [{"box_2d": [ymin, xmin, ymax, xmax]}]} with coordinates normalized to 0-1000."""


def gemini_faces(frame):
    """بديل: Gemini يحدد أماكن الوجوه إذا فشل الكاشف المحلي (حد أقصى 30 طلباً لكل تشغيل)."""
    if not GEMINI_KEY or _FACE_CALLS[0] >= 30:
        return []
    _FACE_CALLS[0] += 1
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = base64.b64encode(buf.tobytes()).decode()
        res = gemini_generate([{"text": FACES_PROMPT},
                               {"inline_data": {"mime_type": "image/jpeg", "data": data}}], temperature=0)
        out = []
        for f in res.get("faces", []):
            y0, x0, y1, x1 = [float(v) / 1000 for v in f["box_2d"]]
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
        return out
    except Exception as e:  # noqa
        print("gemini faces failed:", e, flush=True)
        return []


def detect_cuts(src, start, dur, thr=0.15):
    """أزمنة تبديل الكاميرا (بدقة الإطار) داخل المقطع، نسبةً لبدايته."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                            "-vf", f"scale=320:-2,select='gt(scene,{thr})',showinfo", "-an", "-f", "null", "-"],
                           capture_output=True, text=True)
        cuts = sorted(float(m.group(1)) for m in re.finditer(r"pts_time:([0-9.]+)", r.stderr))
    except Exception as e:  # noqa
        print("cut detection failed:", e, flush=True)
        return []
    out = []
    for c in cuts:
        if 0.15 < c < dur - 0.15 and (not out or c - out[-1] > 0.3):
            out.append(c)
    return out


def read_frame(cap, t, width=DET_W):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    if not ok:
        return None
    h, w = fr.shape[:2]
    if w > width:
        fr = cv2.resize(fr, (width, int(h * width / w)))
    return fr


def motion_people(cap, t0, t1):
    """بديل أخير بدون كشف وجوه: يحدد مكان شخصين من الحركة (تنفّس، إيماءات) في لقطة واسعة ثابتة."""
    d = t1 - t0
    if d < 0.8:
        return []
    n = 12
    grays = []
    for i in range(n):
        fr = read_frame(cap, t0 + 0.2 + (d - 0.4) * i / (n - 1), 480)
        if fr is not None:
            grays.append(cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32))
    if len(grays) < 6:
        return []
    h, w = grays[0].shape
    sd = np.std(np.stack(grays), axis=0)
    mask = (sd > max(3.0, float(np.percentile(sd, 80)))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    col = mask.sum(axis=0).astype(np.float32)
    col = cv2.GaussianBlur(col.reshape(1, -1), (0, 0), sigmaX=max(w * 0.02, 1.0)).ravel()
    if col.max() < 0.15 * h:
        return []
    idx = np.where(col > 0.3 * col.max())[0]
    groups = np.split(idx, np.where(np.diff(idx) > 0.1 * w)[0] + 1)      # كتل حركة منفصلة أفقياً
    clusters = [g for g in groups if len(g) and col[g].max() >= 0.15 * h]
    clusters.sort(key=lambda g: -float(col[g].sum()))
    if len(clusters) < 2:
        return []
    centers = sorted(float((g * col[g]).sum() / col[g].sum()) for g in clusters[:2])
    if centers[1] - centers[0] < 0.3 * w:
        return []
    faces = []
    for p in centers:
        lo, hi = max(int(p - 0.08 * w), 0), min(int(p + 0.08 * w), w)
        rows = np.where(mask[:, lo:hi].sum(axis=1) >= 3)[0]
        if len(rows) == 0:
            return []
        faces.append((p / w, min(rows[0] / h + 0.09, 0.9), 0.08, 0.14))
    return faces


def shot_faces(cap, finder, t0, t1):
    """وجوه لقطة كاملة (نسبية 0-1): ([(cx, cy, w, h)], مصدر الكشف). يجرب أكثر من طريقة قبل الاستسلام."""
    d = t1 - t0
    if d > 0.6:
        times = sorted({min(max(t0 + d * f, t0 + 0.15), t1 - 0.1) for f in (0.2, 0.5, 0.8)})
    else:
        times = [(t0 + t1) / 2]

    def clear(faces):       # شخصان، أو وجه كبير واضح (لقطة قريبة)
        return len(faces) >= 2 or (faces and max(f[3] for f in faces) >= 0.20)

    faces, src_tag, mid = [], "none", None
    if finder.ok:
        for width, thr in ((DET_W, 0.6), (1280, 0.4)):          # تمريرة ثانية بدقة أعلى وعتبة أخفض
            frames = [f for f in (read_frame(cap, t, width) for t in times) if f is not None]
            if not frames:
                return [], "none"
            if mid is None:
                mid = frames[len(frames) // 2]
            hh, ww = frames[0].shape[:2]
            found = [(v[0] / ww, v[1] / hh, v[2] / ww, v[3] / hh)
                     for v in merge_slots([finder.detect(f, thr) for f in frames])]
            if len(found) > len(faces):
                faces, src_tag = found, f"finder@{width}"
            if clear(faces):
                return faces, src_tag
    else:
        mid = read_frame(cap, times[len(times) // 2])
    if mid is not None:
        g = gemini_faces(mid)                                    # Gemini يرى الصورة ويحدد الوجوه
        if len(g) > len(faces):
            faces, src_tag = g, "gemini"
        if clear(faces):
            return faces, src_tag
    m = motion_people(cap, t0, t1)                               # بديل أخير: الحركة
    if len(m) == 2 and len(faces) < 2:
        return m, "motion"
    return faces, src_tag


MIN_TURN = 2.5     # كلام شخص أقصر من هذا (مقاطعة قصيرة) لا يحرّك الكاميرا
PAN = 0.45         # مدة انتقال الكاميرا (ثانية) بحركة ناعمة


def classify_layout(faces, sw, sh):
    """single = وجه واحد (أو شخصان قريبان يكفيهما إطار عمودي)، follow = شخصان متباعدان: الكاميرا تتبع المتحدث."""
    if not faces:
        return {"kind": "single", "face": None}
    faces = sorted(faces, key=lambda f: -f[2] * f[3])[:2]
    big = max(f[2] * f[3] for f in faces)
    faces = sorted([f for f in faces if f[2] * f[3] >= 0.25 * big], key=lambda f: f[0])
    if len(faces) == 1:
        return {"kind": "single", "face": faces[0]}
    crop_w = min(sh * 9 / 16, sw) / sw
    left, right = faces[0][0] - faces[0][2] / 2, faces[1][0] + faces[1][2] / 2
    if (right - left) * 1.3 <= crop_w:
        return {"kind": "single", "face": ((left + right) / 2, (faces[0][1] + faces[1][1]) / 2,
                                             max(f[2] for f in faces), max(f[3] for f in faces))}
    return {"kind": "follow", "faces": faces}


def fill_unknown(flat):
    last = None
    for iv in flat:
        if iv[2] == -1 and last is not None:
            iv[2] = last
        elif iv[2] != -1:
            last = iv[2]
    nxt = None
    for iv in reversed(flat):
        if iv[2] == -1 and nxt is not None:
            iv[2] = nxt
        elif iv[2] != -1:
            nxt = iv[2]
    for iv in flat:
        if iv[2] == -1:
            iv[2] = 0


def smooth_runs(flat, min_turn):
    """كلام أقصر من min_turn لا يحرّك الكاميرا: يُدمج مع من قبله."""
    while True:
        runs = []
        for idx, iv in enumerate(flat):
            if runs and runs[-1][0] == iv[2]:
                runs[-1][2] = idx
                runs[-1][3] += iv[1] - iv[0]
            else:
                runs.append([iv[2], idx, idx, iv[1] - iv[0]])
        short = [r for r in runs if r[3] < min_turn]
        if len(runs) < 2 or not short:
            return
        r = min(short, key=lambda r: r[3])
        ri = runs.index(r)
        new = runs[ri - 1][0] if ri > 0 else runs[ri + 1][0]
        for idx in range(r[1], r[2] + 1):
            flat[idx][2] = new


def sample_gray(cap, t0, t1, fps=8, width=480):
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native / fps)), 1)
    times, frames = [], []
    for i in range(int((t1 - t0) * native)):
        if not cap.grab():
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            h, w = fr.shape[:2]
            if w > width:
                fr = cv2.resize(fr, (width, int(h * width / w)))
            frames.append(cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32))
            times.append(t0 + i / native)
    return times, frames


def speaker_runs(cap, faces, t0, t1):
    """من يتكلم في كل لحظة (حسب حركة الوجه والفم) مع تجاهل المقاطعات القصيرة. يرجع [(من, إلى, رقم الوجه)]."""
    if len(faces) < 2:
        return [(t0, t1, 0)]
    times, frames = sample_gray(cap, t0, t1)
    if len(frames) < 3:
        return [(t0, t1, 0)]
    h, w = frames[0].shape
    regs = []
    for cx, cy, fw, fh in faces:
        regs.append((int(max((cx - 0.5 * fw) * w, 0)), int(min((cx + 0.5 * fw) * w, w)),
                     int(max((cy - 0.05 * fh) * h, 0)), int(min((cy + 0.65 * fh) * h, h))))
    en = np.zeros((len(frames) - 1, len(faces)))
    for k, (x0, x1, y0, y1) in enumerate(regs):
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        for j in range(len(frames) - 1):
            en[j, k] = float(np.abs(frames[j + 1][y0:y1, x0:x1] - frames[j][y0:y1, x0:x1]).mean())
    flat = []
    for j in range(len(frames) - 1):
        order = np.argsort(en[j])[::-1]
        best, second = en[j][order[0]], en[j][order[1]]
        flat.append([times[j], times[j + 1], int(order[0]) if (best > 0.8 and best > 1.2 * second) else -1])
    flat[0][0], flat[-1][1] = t0, t1
    fill_unknown(flat)
    smooth_runs(flat, MIN_TURN)
    runs = []
    for iv in flat:
        if runs and runs[-1][2] == iv[2]:
            runs[-1][1] = iv[1]
        else:
            runs.append([iv[0], iv[1], iv[2]])
    return [tuple(r) for r in runs]


def plan_layouts(src, start, end, finder, cap, sw, sh):
    """يقسّم المقطع عند كل تبديل كاميرا، ويحدد لكل لقطة: وجه ثابت أو كاميرا تتبع المتحدث."""
    cuts = detect_cuts(src, start, end - start)
    bounds = [0.0] + cuts + [end - start]
    parts = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 0.05:
            continue
        faces, tag = shot_faces(cap, finder, start + a, start + b)
        lay = classify_layout(faces, sw, sh)
        lay["t0"], lay["t1"], lay["src"] = a, b, tag
        if lay["kind"] == "follow":
            runs = speaker_runs(cap, lay["faces"], start + a, start + b)
            lay["runs"] = [(ra - start, rb - start, k) for ra, rb, k in runs]
        parts.append(lay)
    return parts or [{"kind": "single", "face": None, "t0": 0.0, "t1": end - start}]


# ---------------------------------------------------------------- captions + hook
def ts(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def esc(s):
    return s.replace("{", "(").replace("}", ")").replace("\n", " ")


def wrap_hook(text, maxc, maxlines=2):
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


def is_latin(words):
    txt = "".join(w["w"] for w in words)
    ar = sum(1 for c in txt if "\u0600" <= c <= "\u06ff")
    la = sum(1 for c in txt if c.isascii() and c.isalpha())
    return la > ar


def pick_fonts():
    """أفضل خط عربي ولاتيني متوفر على السيرفر."""
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True).stdout
    except Exception:  # noqa
        out = ""
    fams = {f.strip().lower() for line in out.splitlines() for f in line.split(",")}

    def first(cands, default):
        for c in cands:
            if c.lower() in fams:
                return c
        return default

    ar = first(["Tajawal", "Vazirmatn", "IBM Plex Sans Arabic", "Cairo", "Noto Kufi Arabic",
                "Noto Sans Arabic"], "Noto Sans Arabic")
    la = first(["Poppins", "Montserrat", "Inter", "Noto Sans"], "Noto Sans")
    print("fonts:", ar, "|", la, flush=True)
    return ar, la


FONT_AR, FONT_LATIN = pick_fonts()


def phrases(ws, maxc):
    """يجمع الكلمات في عبارات قصيرة (سطران كحد أقصى) تظهر دفعة واحدة."""
    out, cur = [], []

    def n_chars(c):
        return sum(len(w["w"]) for w in c) + max(len(c) - 1, 0)

    for w in ws:
        if cur:
            gap = w["nt"] - cur[-1]["ne"]
            if (n_chars(cur) + 1 + len(w["w"]) > 2 * maxc or gap > 0.55 or len(cur) >= 7
                    or re.search(r"[.!?؟,،;:]$", cur[-1]["w"])):
                out.append(cur)
                cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def two_lines(texts, maxc):
    total = sum(len(t) for t in texts) + len(texts) - 1
    if total <= maxc or len(texts) < 2:
        return " ".join(texts)
    best, best_i = None, 1
    for i in range(1, len(texts)):
        a = sum(len(t) for t in texts[:i]) + i - 1
        b = sum(len(t) for t in texts[i:]) + len(texts) - i - 1
        if best is None or max(a, b) < best:
            best, best_i = max(a, b), i
    return " ".join(texts[:best_i]) + "\\N" + " ".join(texts[best_i:])


def ass_head(latin):
    font = FONT_LATIN if latin else FONT_AR
    size = 70 if latin else 80
    spacing = 1 if latin else 0
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,{spacing},0,1,4,2,2,70,70,400,1
Style: Hook,{font},{size + 4},&H00FFFFFF,&H00FFFFFF,&H4A000000,&H00000000,-1,0,0,0,100,100,{spacing},0,3,26,0,8,60,60,250,1
Style: Bar,{font},40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def build_ass(segs, hook, total, path, layouts=None):
    ws, off = [], 0.0
    for sg in segs:
        for w in sg["words"]:
            ws.append({"w": w["w"], "nt": off + (w["s"] - sg["s"]), "ne": off + (w["e"] - sg["s"])})
        off += sg["e"] - sg["s"]

    layouts = layouts or [{"kind": "single", "t0": 0.0, "t1": total}]
    latin = is_latin(ws)
    if latin:
        for x in ws:
            x["w"] = x["w"].upper()
    size = 70 if latin else 80
    maxc = 17 if latin else 20
    lines = [ass_head(latin)]
    ps = phrases(ws, maxc)
    base_tags = "\\fad(70,50)\\fscx94\\fscy94\\t(0,110,\\fscx100\\fscy100)"
    for i, ph in enumerate(ps):
        start = max(ph[0]["nt"] - 0.04, 0)
        end = ph[-1]["ne"] + 0.18
        if i + 1 < len(ps):
            end = min(end, ps[i + 1][0]["nt"] - 0.02)
        end = max(end, start + 0.3)
        text = two_lines([esc(w["w"]) for w in ph], maxc)
        for lay in layouts:                 # عبارة تعبر حدود لقطتين تُقسم عندهما
            s2, e2 = max(start, lay["t0"]), min(end, lay["t1"])
            if e2 - s2 < 0.08:
                continue
            lines.append(f"Dialogue: 0,{ts(s2)},{ts(e2)},Cap,,0,0,0,,{{{base_tags}}}{text}")

    if hook:
        y = 250
        tags = f"\\an8\\move(540,{y + 36},540,{y},0,280)\\fad(180,320)"
        text = wrap_hook(hook.upper() if latin else hook, 20 if latin else 17)
        lines.append(f"Dialogue: 2,{ts(0.1)},{ts(min(4.4, total))},Hook,,0,0,0,,{{{tags}}}{text}")

    ms = int(total * 1000)
    lines.append(f"Dialogue: 5,{ts(0)},{ts(total)},Bar,,0,0,0,,"
                 f"{{\\an7\\pos(0,1890)\\fscx0\\t(0,{ms},\\fscx100)\\1c&HFFFFFF&\\1a&H55&\\bord0\\shad0\\p1}}"
                 f"m 0 0 l {W} 0 {W} 20 0 20{{\\p0}}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- render
HOOK_PROMPT = """Here is the transcript of a short vertical video clip (a podcast, talk or interview).
Write ONE on-screen hook line that appears in the first seconds.
Goal: in 4-9 words, tell the viewer exactly what this clip is about, or what surprising point or promise they will get, so they keep watching.
Rules:
- Same language as the speakers. If Arabic: simple, clear Arabic that Algerian, Gulf and Egyptian viewers all understand (no heavy formal wording). If English: plain, punchy English.
- Specific to THIS clip (use its real topic), never generic, never misleading. No emojis, no hashtags, no quotation marks, no final dot.
- Prefer a strong question, a bold claim, or a clear benefit.
Return ONLY JSON: {{"hook": "..."}}

TRANSCRIPT:
{text}"""


def make_hook(words, start, end):
    """يكتب هوكاً يشرح فكرة المقطع من النص المصحَّح."""
    if not GEMINI_KEY:
        return ""
    text = " ".join(w["w"] for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.2)
    if len(text.split()) < 8:
        return ""
    try:
        res = gemini_generate([{"text": HOOK_PROMPT.format(text=text[:6000])}], temperature=0.5)
        h = str(res.get("hook", "")).strip().strip('"\u201c\u201d\u00ab\u00bb').rstrip(".\u060c")
        return h if 2 <= len(h.split()) <= 14 else ""
    except Exception as e:  # noqa
        print("hook failed:", e, flush=True)
        return ""


def show_hook(clip):
    hook = (clip.get("hook") or "").strip()
    ctype = clip.get("content_type", "")
    if HOOK_MODE == "never":
        return ""
    if HOOK_MODE == "always":
        return hook or (clip.get("title") or "").strip()
    return hook if ctype in ("educational", "important") else ""


def single_crop(face, sw, sh):
    if sw / sh >= 9 / 16:
        base_h, base_w = sh, sh * 9 / 16
    else:
        base_w, base_h = sw, sw * 16 / 9
    fx, fy, fw, fh = face if face else (0.5, 0.45, 0.0, 0.0)
    f = min(1.0, max(0.75, fh / 0.16)) if fh > 0 else 1.0     # تقريب خفيف إذا كان الوجه صغيراً
    ch_, cw_ = int(base_h * f) // 2 * 2, int(base_w * f) // 2 * 2
    x = int(min(max(fx * sw - cw_ / 2, 0), sw - cw_)) // 2 * 2
    y = int(min(max(fy * sh - 0.38 * ch_, 0), sh - ch_)) // 2 * 2
    return cw_, ch_, x, y


def render(idx, clip, src, words, sw, sh, finder, cap):
    start, end = clip["start"], clip["end"]
    dur = end - start
    cw_words = [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.2]
    if not cw_words:
        return None

    # الفيديو يبقى متصلاً (بدون حذف صمت أو قص). الكاميرا تتبع المتحدث وتتبدل مع تبديل المصوّر الأصلي
    layouts = plan_layouts(src, start, end, finder, cap, sw, sh)
    print("layouts:", [(round(l["t0"], 1), round(l["t1"], 1), l["kind"], l.get("src", "")) for l in layouts], flush=True)

    hook = show_hook(clip)
    ass = WORK / f"c{idx}.ass"
    build_ass([{"s": start, "e": end, "words": cw_words}], hook, dur, ass, layouts)

    # حدود اللقطات بالإطارات (30 إطار/ثانية) حتى لا يضيع أي إطار ويبقى الصوت متزامناً تماماً
    nf = max(int(round(dur * 30)), 1)
    edges = [0]
    for lay in layouts[:-1]:
        edges.append(min(max(int(round(lay["t1"] * 30)), edges[-1] + 1), nf - 1))
    edges.append(nf)
    n = len(layouts)
    parts = ["[0:v]fps=30" + (f",split={n}" + "".join(f"[vs{i}]" for i in range(n)) if n > 1 else "[vs0]")]
    for i, lay in enumerate(layouts):
        fa, fb = edges[i], edges[i + 1]
        head = f"[vs{i}]trim=start_frame={fa}:end_frame={fb},setpts=PTS-STARTPTS"
        if lay["kind"] == "follow":
            faces = lay["faces"]
            fh = max(f[3] for f in faces)
            zoom = min(1.0, max(0.75, fh / 0.16))
            if sw / sh >= 9 / 16:
                base_h, base_w = sh, sh * 9 / 16
            else:
                base_w, base_h = sw, sw * 16 / 9
            ch_, cw_ = int(base_h * zoom) // 2 * 2, int(base_w * zoom) // 2 * 2
            cy = sum(f[1] for f in faces) / len(faces)
            y = int(min(max(cy * sh - 0.38 * ch_, 0), sh - ch_)) // 2 * 2

            def xpx(k):
                return int(min(max(faces[k][0] * sw - cw_ / 2, 0), sw - cw_))

            runs = lay["runs"]
            kfs = [(0.0, xpx(runs[0][2]))]
            for r in runs[1:]:
                kfs.append((max(r[0] - lay["t0"] - 0.1, 0.05), xpx(r[2])))
            expr = str(kfs[0][1])
            for (_, xp), (tc, xc) in zip(kfs, kfs[1:]):          # انتقال ناعم (smoothstep) نحو المتحدث الجديد
                u = f"clip((t-{tc:.3f})/{PAN},0,1)"
                expr += f"+({xc - xp})*({u}*{u}*(3-2*{u}))"
            parts.append(f"{head},crop=w={cw_}:h={ch_}:x='{expr}':y={y},scale={W}:{H}:flags=lanczos,"
                         f"setsar=1,format=yuv420p,fps=30[v{i}]")
        else:
            cw_, ch_, x, y = single_crop(lay["face"], sw, sh)
            parts.append(f"{head},crop={cw_}:{ch_}:{x}:{y},scale={W}:{H}:flags=lanczos,"
                         f"setsar=1,format=yuv420p,fps=30[v{i}]")
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vc]")
    fontsdir = os.path.expanduser("~/.fonts")
    fd = f":fontsdir={fontsdir}" if os.path.isdir(fontsdir) else ""
    parts.append(f"[vc]ass=filename={ass}{fd}[vo]")
    parts.append("[0:a]aresample=48000,highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11[ao]")

    script = WORK / f"c{idx}.txt"
    script.write_text(";\n".join(parts), encoding="utf-8")

    out = OUT / f"short_{idx}.mp4"
    run_cmd(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{dur + 0.5:.3f}", "-i", WORK / "src.mp4",
             "-filter_complex_script", script, "-map", "[vo]", "-map", "[ao]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{dur:.2f}",
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

        finder = FaceFinder()
        cap = cv2.VideoCapture(str(src))
        for i, clip in enumerate(clips, 1):
            say(f"🎬 مونتاج المقطع {i}/{len(clips)}...")
            clip_words = fix_text(i, clip, words, src)
            hook = make_hook(clip_words, clip["start"], clip["end"])
            if hook:
                clip["hook"] = hook
            out = render(i, clip, src, clip_words, sw, sh_, finder, cap)
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
