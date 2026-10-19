"""وضع كرة القدم: يلخّص مباراة كاملة أو فيديو ملخص إلى أبرز اللقطات والأهداف بالترتيب. بدون أي نص أو موسيقى."""
import base64
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import numpy as np
import requests

WORK, OUT = pathlib.Path("work"), pathlib.Path("out")
URL = os.environ["VIDEO_URL"]
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or ""
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or ""
FORMAT = (os.environ.get("FOOTBALL_FORMAT") or "vertical").lower()   # vertical (تيك توك/ريلز) | wide
TARGET_MIN = min(max(float(os.environ.get("TARGET_MIN") or ("3" if FORMAT == "vertical" else "12")), 1.0), 25.0)
API = f"https://api.telegram.org/bot{TOKEN}"

# مدة كل لقطة (بناء الهجمة قبل اللحظة + الاحتفال والإعادة بعدها)
LEAD = {"goal": 28, "penalty": 24, "big_chance": 20, "red_card": 18, "foul": 14, "other": 16, "unknown": 22}
TAIL = {"goal": 34, "penalty": 26, "big_chance": 16, "red_card": 16, "foul": 10, "other": 12, "unknown": 20}
WEIGHT = {"goal": 100, "penalty": 70, "big_chance": 55, "red_card": 50, "foul": 5, "other": 0, "unknown": 20}


def run_cmd(cmd):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def say(text):
    try:
        requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": text}, timeout=30)
    except Exception as e:  # noqa
        print("telegram error:", e)


def download():
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    base = ["yt-dlp", "--no-playlist", "--merge-output-format", "mp4",
            "-o", str(WORK / "src.%(ext)s")]
    if pathlib.Path("cookies.txt").exists():
        base += ["--cookies", "cookies.txt"]
    attempts = [
        ["-f", "bv*[height<=720]+ba/b[height<=720]/b"],
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
                         "-crf", "23", "-c:a", "aac", WORK / "src.mp4"])
            return WORK / "src.mp4"
        errs = [l for l in r.stderr.splitlines() if l.startswith("ERROR")]
        last = errs[-1] if errs else last
    raise RuntimeError(last)


# ---------------------------------------------------------------- 1) find loud, sustained moments
def audio_db(src):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "8000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    sec = len(x) // 8000
    rms = np.sqrt((x[:sec * 8000].reshape(sec, 8000) ** 2).mean(axis=1)) + 1e-6
    return 20 * np.log10(rms)


def find_candidates(db, max_n=40, min_gap=60, skip=45):
    """يلتقط اللحظات التي يرتفع فيها صوت الملعب بشكل مستمر (الأهداف تعطي أطول وأعلى زئير)."""
    n = len(db)
    win = 121
    pad = np.pad(db, (win // 2, win // 2), mode="edge")
    base = np.median(np.lib.stride_tricks.sliding_window_view(pad, win), axis=1)
    excess = db - base
    k = 10
    csum = np.concatenate([[0], np.cumsum(excess)])
    sustained = np.array([(csum[min(i + k, n)] - csum[i]) / max(min(i + k, n) - i, 1) for i in range(n)])
    sustained[:skip] = -99        # تجاهل بداية البث
    cands = []
    for _ in range(max_n):
        i = int(np.argmax(sustained))
        if sustained[i] < 3.0:
            break
        peak = float(excess[i:i + k].max())
        onset = i
        while onset > 0 and i - onset < 15 and excess[onset - 1] > 0.4 * peak:
            onset -= 1
        cands.append({"t": onset, "score": float(sustained[i])})
        sustained[max(0, i - min_gap):i + min_gap] = -99
    return cands


# ---------------------------------------------------------------- 2) ask Gemini what happened (optional)
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


def gemini_generate(parts, config=None):
    last = None
    for model in gemini_models():
        for attempt in range(2):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": GEMINI_KEY},
                    json={"contents": [{"parts": parts}],
                          "generationConfig": {"responseMimeType": "application/json", "temperature": 0,
                                               **(config or {})}},
                    timeout=300)
                if r.status_code == 429 and attempt == 0:
                    time.sleep(25)
                    continue
                r.raise_for_status()
                return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            except Exception as e:  # noqa
                print("Gemini model", model, "failed:", e)
                last = e
                break
    raise last or RuntimeError("no Gemini model")


CLASSIFY = """You are a football (soccer) analyst. This is the audio (commentary + stadium crowd) from just before to just after a very loud moment of a match. Decide what happened:
- "goal": a goal was scored (commentator screams, long roar, celebration)
- "penalty": a penalty was awarded or taken
- "big_chance": shot saved, hit the post or bar, narrowly missed, a great save
- "red_card": a player was sent off
- "foul": an ordinary foul, yellow card, or argument
- "other": anthem, chanting, ordinary build-up, kick-off, half-time, nothing important
The commentary may be in any language. Return ONLY JSON: {"event": "...", "excitement": 0-10}"""


def classify(src, cand):
    a, b = max(cand["t"] - 25, 0), cand["t"] + 35
    audio = WORK / f"cand_{cand['t']}.mp3"
    run_cmd(["ffmpeg", "-y", "-ss", a, "-t", b - a, "-i", src, "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "48k", audio])
    data = base64.b64encode(audio.read_bytes()).decode()
    res = gemini_generate([{"text": CLASSIFY}, {"inline_data": {"mime_type": "audio/mp3", "data": data}}])
    ev = str(res.get("event", "unknown")).lower()
    cand["event"] = ev if ev in WEIGHT else "unknown"
    cand["excitement"] = float(res.get("excitement", 5))


# ---------------------------------------------------------------- 3) choose moments to reach the target length
def clip_window(c, total, target_s=720.0):
    """اللقطات تقصر تلقائياً كلما كان الملخص المطلوب أقصر."""
    ev = c.get("event", "unknown")
    scale = min(1.0, max(0.35, target_s / 600.0))
    lead = max(LEAD[ev] * scale, 8)
    tail = max(TAIL[ev] * scale, 8)
    return max(c["t"] - lead, 0), min(c["t"] + tail, total)


def choose(cands, total, target_s):
    for c in cands:
        c.setdefault("event", "unknown")
        c["rank"] = WEIGHT[c["event"]] + c.get("excitement", 0) * 2 + c["score"]
    chosen, length = [], 0.0
    for c in sorted(cands, key=lambda c: -c["rank"]):
        if c["event"] in ("other", "foul") and length >= 0.6 * target_s:
            continue
        if c["event"] == "other":
            continue
        a, b = clip_window(c, total, target_s)
        if length + (b - a) > target_s * 1.15 and chosen:
            continue
        chosen.append(c)
        length += b - a
        if length >= target_s:
            break
    if length < 0.5 * target_s:          # لا يوجد ما يكفي من اللقطات المؤكدة: أكمل بالأقوى صوتاً
        for c in sorted(cands, key=lambda c: -c["rank"]):
            if c in chosen or c["event"] == "other" or length >= 0.5 * target_s:
                continue
            a, b = clip_window(c, total, target_s)
            chosen.append(c)
            length += b - a
    return sorted(chosen, key=lambda c: c["t"]), length


# ---------------------------------------------------------------- 4) cut, fade, join (no text, no music)
def clip_filter(d):
    """عمودي 1080x1920: اللقطة كاملة في الوسط وخلفها نسخة مموهة منها (بدون أي نص)."""
    fade = f"fade=t=in:st=0:d=0.3,fade=t=out:st={d - 0.3:.2f}:d=0.3"
    if FORMAT == "vertical":
        return ("[0:v]split=2[bg][fg];"
                "[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                "boxblur=10:2,scale=1080:1920,eq=brightness=-0.08[b];"
                "[fg]scale=1080:-2[f];"
                f"[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30,{fade}[v]")
    return ("[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,{fade}[v]")


def build_windows(src, windows):
    clips = []
    for i, (a, b) in enumerate(windows, 1):
        d = b - a
        out = WORK / f"clip{i:02d}.mp4"
        af = f"aresample=48000,afade=t=in:st=0:d=0.3,afade=t=out:st={d - 0.3:.2f}:d=0.3"
        run_cmd(["ffmpeg", "-y", "-ss", f"{a:.2f}", "-t", f"{d:.2f}", "-i", src,
                 "-filter_complex", clip_filter(d), "-map", "[v]", "-map", "0:a?", "-af", af,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", out])
        clips.append(out)
    lst = WORK / "concat.txt"
    lst.write_text("\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8")
    final = OUT / "football_summary.mp4"
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
             "-movflags", "+faststart", final])
    return final


def build(src, chosen, total, target_s=720.0):
    return build_windows(src, [clip_window(c, total, target_s) for c in chosen])


# ---------------------------------------------------------------- highlights video -> shorter summary (Gemini watches it)
def to_sec(v):
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(p) for p in str(v).strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def make_proxy(src):
    proxy = WORK / "proxy.mp4"
    run_cmd(["ffmpeg", "-y", "-i", src, "-vf", "scale=-2:360,fps=2", "-c:v", "libx264", "-crf", "30",
             "-preset", "veryfast", "-c:a", "aac", "-b:a", "48k", "-ac", "1", proxy])
    return proxy


def upload_to_gemini(path):
    size = path.stat().st_size
    r = requests.post("https://generativelanguage.googleapis.com/upload/v1beta/files",
                      params={"key": GEMINI_KEY},
                      headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                               "X-Goog-Upload-Header-Content-Length": str(size),
                               "X-Goog-Upload-Header-Content-Type": "video/mp4",
                               "Content-Type": "application/json"},
                      json={"file": {"display_name": "match"}}, timeout=60)
    r.raise_for_status()
    url = r.headers["x-goog-upload-url"]
    with open(path, "rb") as f:
        r2 = requests.post(url, headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                         "X-Goog-Upload-Command": "upload, finalize"},
                           data=f, timeout=900)
    r2.raise_for_status()
    info = r2.json()["file"]
    for _ in range(72):
        if info.get("state") == "ACTIVE":
            return info
        if info.get("state") == "FAILED":
            raise RuntimeError("Gemini could not process the video")
        time.sleep(5)
        info = requests.get(f"https://generativelanguage.googleapis.com/v1beta/{info['name']}",
                            params={"key": GEMINI_KEY}, timeout=30).json()
    raise RuntimeError("Gemini video processing timeout")


SUMMARY_PROMPT = """You are a football (soccer) video editor. This video is a match highlights compilation of about {total_min:.0f} minutes. Make a SHORT summary of about {target} seconds with only the best moments.

Pick, in this priority: all goals (include 4-6 seconds of the build-up before the ball goes in, and about 3 seconds of the celebration), then the best saves, big chances (post, bar, narrow misses), skills and dribbles, red cards.
Rules:
- If the same goal appears again as a replay, keep only the first live version (or the clearest angle), not both.
- Skip: intro, logos, graphics, scoreboards-only shots, studio talk, adverts, crowd shots, slow replays of things already chosen.
- Each segment is 5-25 seconds. Total of all segments about {target} seconds.
- Use exact timestamps from the video, format MM:SS.
- importance: 1-10 (goals 9-10).
Return ONLY a JSON array in chronological order:
[{{"start":"MM:SS","end":"MM:SS","type":"goal|save|chance|skill|card|other","importance":0}}]"""


def summarize_with_gemini(src, total, target_s):
    proxy = make_proxy(src)
    info = upload_to_gemini(proxy)
    prompt = SUMMARY_PROMPT.format(total_min=total / 60, target=int(target_s))
    parts = [{"file_data": {"mime_type": "video/mp4", "file_uri": info["uri"]}}, {"text": prompt}]
    try:
        data = gemini_generate(parts, {"mediaResolution": "MEDIA_RESOLUTION_LOW"})
    except Exception as e:  # noqa
        print("retrying without mediaResolution:", e)
        data = gemini_generate(parts)
    try:
        requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{info['name']}",
                        params={"key": GEMINI_KEY}, timeout=30)
    except Exception:  # noqa
        pass
    return segments_from(data, total, target_s)


def segments_from(data, total, target_s):
    if isinstance(data, dict):
        data = data.get("segments") or data.get("moments") or []
    segs = []
    for d in data:
        try:
            a, b = to_sec(d["start"]), to_sec(d["end"])
        except (KeyError, ValueError, TypeError):
            continue
        a, b = max(a, 0.0), min(b, total)
        if b - a < 3:
            continue
        if b - a > 30:
            b = a + 30
        segs.append({"a": a, "b": b, "imp": float(d.get("importance") or 5), "type": d.get("type", "")})
    chosen, length = [], 0.0
    for sg in sorted(segs, key=lambda x: -x["imp"]):
        if any(sg["a"] < o["b"] and sg["b"] > o["a"] for o in chosen):
            continue
        if length + (sg["b"] - sg["a"]) > target_s * 1.15 and chosen:
            continue
        chosen.append(sg)
        length += sg["b"] - sg["a"]
        if length >= target_s:
            break
    chosen.sort(key=lambda x: x["a"])
    return [(sg["a"], sg["b"]) for sg in chosen]


def audio_windows(src, total, target_s):
    """بديل بدون ذكاء اصطناعي: أعلى اللحظات صوتاً في الفيديو."""
    cands = find_candidates(audio_db(src), max_n=30, min_gap=20, skip=5)
    chosen, _ = choose(cands, total, target_s)
    return [clip_window(c, total, target_s) for c in chosen]


def shrink_for_telegram(final, dur):
    """نسخة أخف (أقل من 50MB) لإرسالها في تلغرام، والأصلية تبقى في Artifacts."""
    vb = max(int(46 * 8 * 1024 / dur - 128), 500)      # kbps
    small = OUT / "football_summary_telegram.mp4"
    scale = "scale=720:1280" if FORMAT == "vertical" else "scale=960:540"
    run_cmd(["ffmpeg", "-y", "-i", final, "-vf", scale, "-c:v", "libx264", "-preset", "veryfast",
             "-b:v", f"{vb}k", "-maxrate", f"{int(vb * 1.3)}k", "-bufsize", f"{vb * 2}k",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", small])
    return small


def duration_of(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                                check=True).stdout.strip())


def main():
    try:
        say("⬇️ جاري تحميل المباراة (قد يطول حسب حجمها)...")
        src = download()
        total = duration_of(src)

        target_s = TARGET_MIN * 60
        if total <= 30 * 60:
            # فيديو ملخص جاهز: Gemini يشاهده ويختار أبرز اللقطات
            windows = []
            if GEMINI_KEY:
                say("🧠 أشاهد الفيديو بالذكاء الاصطناعي لاختيار الأهداف وأبرز اللقطات (قد يأخذ دقائق)...")
                try:
                    windows = summarize_with_gemini(src, total, target_s)
                except Exception as e:  # noqa
                    print("Gemini summary failed:", e)
                    say("⚠️ تعذّر تحليل الفيديو بالذكاء الاصطناعي، سأختار حسب ارتفاع الصوت.")
            if not windows:
                windows = audio_windows(src, total, target_s)
            if not windows:
                say("❌ لم أستطع اختيار لقطات مناسبة.")
                sys.exit(1)
            say(f"🎬 أركّب الملخص من {len(windows)} لقطة...")
            final = build_windows(src, windows)
        else:
            say("🔊 تحليل صوت الملعب لاكتشاف اللحظات القوية...")
            cands = find_candidates(audio_db(src))
            if len(cands) < 2:
                say("❌ لم أجد لحظات قوية واضحة. ربما الفيديو ملخص جاهز أو صوته ضعيف.")
                sys.exit(1)

            if GEMINI_KEY:
                say(f"🧠 وجدت {len(cands)} لحظة، أتحقق أيها أهداف...")
                for c in cands[:(16 if TARGET_MIN <= 4 else 30)]:
                    try:
                        classify(src, c)
                    except Exception as e:  # noqa
                        print("classify failed:", e)
                    time.sleep(6)   # حدود الطبقة المجانية

            chosen, length = choose(cands, total, target_s)
            if not chosen:
                say("❌ لم أستطع اختيار لقطات مناسبة.")
                sys.exit(1)
            goals = sum(1 for c in chosen if c.get("event") == "goal")
            say(f"🎬 أركّب الملخص: {len(chosen)} لقطة (أهداف مؤكدة: {goals}) بمدة تقارب {length / 60:.1f} دقيقة...")
            final = build(src, chosen, total, target_s)

        dur = duration_of(final)
        size = final.stat().st_size / 1024 / 1024
        run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                   f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}")
        to_send, note = final, ""
        if size > 49:
            to_send = shrink_for_telegram(final, dur)
            note = (f"\nهذه نسخة أخف لتلغرام. النسخة الأصلية بجودة أعلى ({size:.0f}MB) "
                    f"في قسم Artifacts:\n{run_url}")
        with open(to_send, "rb") as f:
            requests.post(f"{API}/sendVideo", data={"chat_id": CHAT, "supports_streaming": "true"},
                          files={"video": f}, timeout=900)
        say(f"✅ ملخص المباراة جاهز ({dur / 60:.1f} دقيقة)." + note)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        say("❌ حدث خطأ: " + str(sys.exc_info()[1])[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()
