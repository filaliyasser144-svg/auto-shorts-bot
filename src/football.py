"""وضع كرة القدم: يلخّص مباراة كاملة إلى أبرز اللقطات والأهداف بالترتيب. بدون أي نص أو موسيقى."""
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
TARGET_MIN = min(max(float(os.environ.get("TARGET_MIN") or "12"), 5.0), 25.0)
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


def find_candidates(db, max_n=40, min_gap=60):
    """يلتقط اللحظات التي يرتفع فيها صوت الملعب بشكل مستمر (الأهداف تعطي أطول وأعلى زئير)."""
    n = len(db)
    win = 121
    pad = np.pad(db, (win // 2, win // 2), mode="edge")
    base = np.median(np.lib.stride_tricks.sliding_window_view(pad, win), axis=1)
    excess = db - base
    k = 10
    csum = np.concatenate([[0], np.cumsum(excess)])
    sustained = np.array([(csum[min(i + k, n)] - csum[i]) / max(min(i + k, n) - i, 1) for i in range(n)])
    sustained[:45] = -99          # تجاهل بداية البث
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


def gemini_generate(parts):
    last = None
    for model in gemini_models():
        for attempt in range(2):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": GEMINI_KEY},
                    json={"contents": [{"parts": parts}],
                          "generationConfig": {"responseMimeType": "application/json", "temperature": 0}},
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
def clip_window(c, total):
    ev = c.get("event", "unknown")
    return max(c["t"] - LEAD[ev], 0), min(c["t"] + TAIL[ev], total)


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
        a, b = clip_window(c, total)
        if length + (b - a) > target_s * 1.15 and chosen:
            continue
        chosen.append(c)
        length += b - a
        if length >= target_s:
            break
    if length < 0.5 * target_s:          # لا يوجد ما يكفي من اللقطات المؤكدة: أكمل بالأقوى صوتاً
        for c in sorted(cands, key=lambda c: -c["rank"]):
            if c in chosen or length >= 0.5 * target_s:
                continue
            a, b = clip_window(c, total)
            chosen.append(c)
            length += b - a
    return sorted(chosen, key=lambda c: c["t"]), length


# ---------------------------------------------------------------- 4) cut, fade, join (no text, no music)
def build(src, chosen, total):
    clips = []
    for i, c in enumerate(chosen, 1):
        a, b = clip_window(c, total)
        d = b - a
        out = WORK / f"clip{i:02d}.mp4"
        vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
              "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,"
              f"fade=t=in:st=0:d=0.3,fade=t=out:st={d - 0.3:.2f}:d=0.3")
        af = f"aresample=48000,afade=t=in:st=0:d=0.3,afade=t=out:st={d - 0.3:.2f}:d=0.3"
        run_cmd(["ffmpeg", "-y", "-ss", f"{a:.2f}", "-t", f"{d:.2f}", "-i", src,
                 "-vf", vf, "-af", af, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", out])
        clips.append(out)
    lst = WORK / "concat.txt"
    lst.write_text("\n".join(f"file '{c.resolve()}'" for c in clips), encoding="utf-8")
    final = OUT / "football_summary.mp4"
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
             "-movflags", "+faststart", final])
    return final


def duration_of(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                                check=True).stdout.strip())


def main():
    try:
        say("⬇️ جاري تحميل المباراة (قد يطول حسب حجمها)...")
        src = download()
        total = duration_of(src)

        say("🔊 تحليل صوت الملعب لاكتشاف اللحظات القوية...")
        cands = find_candidates(audio_db(src))
        if len(cands) < 2:
            say("❌ لم أجد لحظات قوية واضحة. ربما الفيديو ملخص جاهز أو صوته ضعيف.")
            sys.exit(1)

        if GEMINI_KEY:
            say(f"🧠 وجدت {len(cands)} لحظة، أتحقق أيها أهداف...")
            for c in cands[:30]:
                try:
                    classify(src, c)
                except Exception as e:  # noqa
                    print("classify failed:", e)
                time.sleep(6)   # حدود الطبقة المجانية

        chosen, length = choose(cands, total, TARGET_MIN * 60)
        if not chosen:
            say("❌ لم أستطع اختيار لقطات مناسبة.")
            sys.exit(1)
        goals = sum(1 for c in chosen if c.get("event") == "goal")

        say(f"🎬 أركّب الملخص: {len(chosen)} لقطة (أهداف مؤكدة: {goals}) بمدة تقارب {length / 60:.0f} دقيقة...")
        final = build(src, chosen, total)

        mins = duration_of(final) / 60
        size = final.stat().st_size / 1024 / 1024
        run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                   f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}")
        if size <= 49:
            with open(final, "rb") as f:
                requests.post(f"{API}/sendVideo", data={"chat_id": CHAT, "supports_streaming": "true"},
                              files={"video": f}, timeout=900)
            say(f"✅ ملخص المباراة جاهز ({mins:.0f} دقيقة).")
        else:
            say(f"✅ الملخص جاهز ({mins:.0f} دقيقة، {size:.0f}MB) وهو أكبر من حد تلغرام (50MB).\n"
                f"حمّله من قسم Artifacts في هذه الصفحة:\n{run_url}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        say("❌ حدث خطأ: " + str(sys.exc_info()[1])[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()
