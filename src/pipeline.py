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
HOOK_MODE = (os.environ.get("HOOK_MODE") or "auto").lower()    # auto | always | never
TEXT_FIX = (os.environ.get("TEXT_FIX") or "on").lower()        # on | off
MIN_D, MAX_D = 25, 60
GAP = 0.40  # أي صمت أطول من هذا (بالثواني) يُحذف
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


# ---------------------------------------------------------------- faces + who is speaking
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DET_W = 800        # عرض الصورة المستخدمة للكشف (أسرع وكافٍ)
MIN_TURN = 2.5     # أقل مدة لكلام شخص حتى تنتقل الكاميرا إليه (المقاطعات القصيرة لا تحرّك الكاميرا)
SPLIT_MIN = 1.5    # لا نقسم اللقطة إلى جزأين أقصر من هذا


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

    def detect(self, img):
        """وجوه الصورة: [(cx, cy, w, h, mouth_x, mouth_y)] بإحداثيات الصورة."""
        h, w = img.shape[:2]
        out = []
        try:
            if self.yunet is not None:
                self.yunet.setInputSize((w, h))
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


def sample_frames(cap, t0, t1, fps=8):
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
            hh, ww = fr.shape[:2]
            if ww > DET_W:
                fr = cv2.resize(fr, (DET_W, int(hh * DET_W / ww)))
            times.append(t0 + i / native)
            frames.append(fr)
    return times, frames


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


def analyze_segment(sg, finder, cap):
    """وجوه المقطع + حركة الفم لكل وجه، لمعرفة من يتكلم."""
    times, frames = sample_frames(cap, sg["s"], sg["e"])
    if len(frames) < 2:
        return None
    pick = sorted({0, len(frames) // 2, len(frames) - 1})
    slots = merge_slots([finder.detect(frames[i]) for i in pick])
    if not slots:
        return None
    hh, ww = frames[0].shape[:2]
    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) for f in frames]
    energy = np.zeros((len(frames) - 1, len(slots)))
    for k, v in enumerate(slots):
        cx, cy, fw, fh, mx, my = v
        x0, x1 = int(max(mx - 0.32 * fw, 0)), int(min(mx + 0.32 * fw, ww))
        y0, y1 = int(max(my - 0.22 * fw, 0)), int(min(my + 0.22 * fw, hh))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        for j in range(len(frames) - 1):
            energy[j, k] = float(np.abs(grays[j + 1][y0:y1, x0:x1] - grays[j][y0:y1, x0:x1]).mean())
    norm = [(v[0] / ww, v[1] / hh, v[2] / ww, v[3] / hh) for v in slots]
    return {"slots": [{"cx": a, "cy": b, "w": c, "h": d} for a, b, c, d in norm],
            "times": times, "energy": energy}


def same_layout(a, b):
    if a is None or b is None or len(a["slots"]) != len(b["slots"]):
        return False
    return all(abs(x["cx"] - y["cx"]) < 0.08 for x, y in zip(a["slots"], b["slots"]))


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


def decide(segs, infos):
    """قرار الكاميرا لكل مقطع: [{'sg': مقطع (قد يُقسم), 'face': (x, y, ارتفاع الوجه) أو None}]."""
    ivs = []
    for sg, info in zip(segs, infos):
        lst = []
        if info is not None:
            if len(info["slots"]) == 1:
                lst = [[sg["s"], sg["e"], 0]]
            else:
                times, en = info["times"], info["energy"]
                for j in range(len(times) - 1):
                    order = np.argsort(en[j])[::-1]
                    best, second = en[j][order[0]], en[j][order[1]]
                    lab = int(order[0]) if (best > 1.0 and best > 1.2 * second) else -1
                    lst.append([times[j], times[j + 1], lab])
                if lst:
                    lst[0][0], lst[-1][1] = sg["s"], sg["e"]
        ivs.append(lst)

    i = 0
    while i < len(segs):          # مجموعات لقطات متتالية بنفس تكوين الكاميرا (شخصان أو أكثر)
        if infos[i] is None or len(infos[i]["slots"]) < 2 or not ivs[i]:
            i += 1
            continue
        j = i + 1
        while j < len(segs) and same_layout(infos[i], infos[j]) and ivs[j]:
            j += 1
        flat = [iv for k in range(i, j) for iv in ivs[k]]
        fill_unknown(flat)
        smooth_runs(flat, MIN_TURN)
        i = j

    plan = []
    for sg, info, lst in zip(segs, infos, ivs):
        if info is None or not lst:
            plan.append({"sg": sg, "face": None})
            continue
        runs = []
        for iv in lst:
            if runs and runs[-1][2] == iv[2]:
                runs[-1][1] = iv[1]
            else:
                runs.append([iv[0], iv[1], iv[2]])
        merged = True
        while merged and len(runs) > 1:      # لا نقسم إلى أجزاء قصيرة جداً
            merged = False
            for ri, r in enumerate(runs):
                if r[1] - r[0] < SPLIT_MIN:
                    tgt = runs[ri - 1] if ri > 0 else runs[ri + 1]
                    tgt[0], tgt[1] = min(tgt[0], r[0]), max(tgt[1], r[1])
                    runs.pop(ri)
                    merged = True
                    break
            for ri in range(len(runs) - 1):  # نفس الشخص متتالياً
                if runs[ri][2] == runs[ri + 1][2]:
                    runs[ri][1] = runs[ri + 1][1]
                    runs.pop(ri + 1)
                    merged = True
                    break
        for r in runs:
            part = {"s": r[0], "e": r[1],
                    "words": [w for w in sg["words"] if r[0] <= (w["s"] + w["e"]) / 2 < r[1]
                              or (r is runs[-1] and (w["s"] + w["e"]) / 2 >= r[1])]}
            if not part["words"]:
                part["words"] = list(sg["words"]) if len(runs) == 1 else []
            sl = info["slots"][min(r[2], len(info["slots"]) - 1)]
            plan.append({"sg": part, "face": (sl["cx"], sl["cy"], sl["h"])})
    return [p for p in plan if p["sg"]["words"]]


def plan_crops(segs, finder, cap):
    infos = []
    for sg in segs:
        try:
            infos.append(analyze_segment(sg, finder, cap))
        except Exception as e:  # noqa
            print("segment analysis failed:", e, flush=True)
            infos.append(None)
    return decide(segs, infos)


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
    maxc = 17 if latin else 20
    lines = [ass_head(latin)]
    ps = phrases(ws, maxc)
    for i, ph in enumerate(ps):
        start = max(ph[0]["nt"] - 0.04, 0)
        end = ph[-1]["ne"] + 0.18
        if i + 1 < len(ps):
            end = min(end, ps[i + 1][0]["nt"] - 0.02)
        end = max(end, start + 0.3)
        text = two_lines([esc(w["w"]) for w in ph], maxc)
        lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},Cap,,0,0,0,,"
                     f"{{\\fad(70,50)\\fscx94\\fscy94\\t(0,110,\\fscx100\\fscy100)}}{text}")

    if hook:
        text = wrap_hook(hook.upper() if latin else hook, 20 if latin else 17)
        lines.append(f"Dialogue: 2,{ts(0.1)},{ts(min(3.8, total))},Hook,,0,0,0,,"
                     f"{{\\fad(90,260)\\fscx55\\fscy55\\t(0,150,\\fscx110\\fscy110)"
                     f"\\t(150,270,\\fscx100\\fscy100)}}{text}")

    ms = int(total * 1000)
    lines.append(f"Dialogue: 5,{ts(0)},{ts(total)},Bar,,0,0,0,,"
                 f"{{\\an7\\pos(0,1890)\\fscx0\\t(0,{ms},\\fscx100)\\1c&HFFFFFF&\\1a&H55&\\bord0\\shad0\\p1}}"
                 f"m 0 0 l {W} 0 {W} 20 0 20{{\\p0}}")
    path.write_text("\n".join(lines), encoding="utf-8")


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


def render(idx, clip, src, words, sw, sh, finder, cap):
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

    # الكاميرا الذكية: تتبع وجه المتحدث (شخصان أو أكثر) وتنتقل فقط عند كلام طويل
    if finder.ok:
        plan = plan_crops(segs, finder, cap)
    else:
        plan = [{"sg": sg, "face": None} for sg in segs]
    if not plan:
        plan = [{"sg": sg, "face": None} for sg in segs]
    segs = [p["sg"] for p in plan]
    total = sum(sg["e"] - sg["s"] for sg in segs)

    if sw / sh >= 9 / 16:
        base_h, base_w = sh, sh * 9 / 16
    else:
        base_w, base_h = sw, sw * 16 / 9

    hook = show_hook(clip)
    ass = WORK / f"c{idx}.ass"
    build_ass(segs, hook, total, ass)

    parts, prev = [], (0.5, 0.45, 0.0)
    for i, p in enumerate(plan):
        sg = p["sg"]
        fx, fy, fh = p["face"] or prev
        prev = (fx, fy, fh)
        # تقريب خفيف إذا كان الوجه صغيراً (لقطة واسعة)، بدون تكبير مبالغ فيه
        f = min(1.0, max(0.75, fh / 0.16)) if fh > 0 else 1.0
        ch_, cw_ = int(base_h * f) // 2 * 2, int(base_w * f) // 2 * 2
        x = int(min(max(fx * sw - cw_ / 2, 0), sw - cw_)) // 2 * 2
        y = int(min(max(fy * sh - 0.38 * ch_, 0), sh - ch_)) // 2 * 2
        a, b = sg["s"] - start, sg["e"] - start
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,"
                     f"crop={cw_}:{ch_}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1,fps=30[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,aresample=48000[a{i}]")
    n = len(plan)
    parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    fontsdir = os.path.expanduser("~/.fonts")
    fd = f":fontsdir={fontsdir}" if os.path.isdir(fontsdir) else ""
    parts.append(f"[vc]ass=filename={ass}{fd}[vo]")
    parts.append("[ac]highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11[ao]")

    script = WORK / f"c{idx}.txt"
    script.write_text(";\n".join(parts), encoding="utf-8")

    out = OUT / f"short_{idx}.mp4"
    run_cmd(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start + 1:.3f}", "-i", WORK / "src.mp4",
             "-filter_complex_script", script, "-map", "[vo]", "-map", "[ao]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.2f}",
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
