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
_MODELS = []


def gemini_models():
    if GEMINI_MODEL:
        return [GEMINI_MODEL]
    if _MODELS:
        return _MODELS
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
    _MODELS.extend((stable + prev)[:4] or ["gemini-2.5-flash"])
    return _MODELS


def gemini_generate(parts, temperature=0.4):
    last = None
    for model in gemini_models():
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
                if r.status_code == 429 and attempt == 0:
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
Rules:
- Keep the original language(s). Arabic: Arabic script, keep the speaker's own dialect wording (do not convert to formal Arabic, do not translate), no diacritics. English: normal spelling.
- One or more people may speak. Write everything every speaker says, in order. Do not label speakers.
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
                              temperature=0)
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


# ---------------------------------------------------------------- face tracking
class FaceTracker:
    def __init__(self, src):
        self.ok = False
        try:
            self.cap = cv2.VideoCapture(str(src))
            self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.ok = not self.det.empty()
        except Exception as e:  # noqa
            print("face detection unavailable:", e)

    def center(self, t0, t1):
        """(x, y, ارتفاع الوجه نسبةً للإطار) أو None."""
        if not self.ok:
            return None
        xs, ys, hs = [], [], []
        try:
            for f in (0.15, 0.5, 0.85):
                self.cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * f) * 1000)
                ok, frame = self.cap.read()
                if not ok:
                    continue
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = g.shape
                faces = self.det.detectMultiScale(g, 1.2, 5, minSize=(w // 20, w // 20))
                if len(faces):
                    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    xs.append((x + fw / 2) / w)
                    ys.append((y + fh / 2) / h)
                    hs.append(fh / h)
        except Exception as e:  # noqa
            print("face detection error:", e)
            return None
        if not xs:
            return None
        return float(np.median(xs)), float(np.median(ys)), float(np.median(hs))


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
Style: Hook,{font},92,&H00101010,&H00101010,&H0000D7FF,&H96000000,-1,0,0,0,100,100,0,0,3,26,6,8,50,50,250,1
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
        text = wrap_hook(hook.upper() if latin else hook, 20 if latin else 17)
        lines.append(f"Dialogue: 2,{ts(0.1)},{ts(min(3.8, total))},Hook,,0,0,0,,"
                     f"{{\\fad(90,260)\\fscx55\\fscy55\\t(0,150,\\fscx110\\fscy110)"
                     f"\\t(150,270,\\fscx100\\fscy100)}}{text}")

    ms = int(total * 1000)
    lines.append(f"Dialogue: 5,{ts(0)},{ts(total)},Bar,,0,0,0,,"
                 f"{{\\an7\\pos(0,1890)\\fscx0\\t(0,{ms},\\fscx100)\\1c&H00D7FF&\\bord0\\shad0\\p1}}"
                 f"m 0 0 l {W} 0 {W} 26 0 26{{\\p0}}")
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


def render(idx, clip, src, words, sw, sh, tracker):
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

    if sw / sh >= 9 / 16:
        base_h, base_w = sh, sh * 9 / 16
    else:
        base_w, base_h = sw, sw * 16 / 9

    hook = show_hook(clip)
    ass = WORK / f"c{idx}.ass"
    build_ass(segs, hook, total, ass)

    parts, prev = [], (0.5, 0.45, 0.0)
    for i, sg in enumerate(segs):
        found = tracker.center(sg["s"], sg["e"])
        fx, fy, fh = found if found else prev
        prev = (fx, fy, fh)
        # تكبير خفيف فقط في اللقطات الواسعة؛ لا تكبير في اللقطات القريبة أو إذا لم نعرف مكان الوجه
        f = 0.85 if (found and i % 2 == 1 and fh < 0.22) else 1.0
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

        tracker = FaceTracker(src)
        for i, clip in enumerate(clips, 1):
            say(f"🎬 مونتاج المقطع {i}/{len(clips)}...")
            clip_words = fix_text(i, clip, words, src)
            out = render(i, clip, src, clip_words, sw, sh_, tracker)
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
