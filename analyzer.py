import os
import re
import json
import time
import glob
from google import genai
from google.genai import types
from ytdl import run_ytdlp

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT = """
أنت خبير مونتاج وتسويق عبر الفيديوهات القصيرة (Reels & TikTok).
استمع إلى هذا الصوت بعناية (بودكاست باللهجة الجزائرية)، واستخرج أفضل 3 مقاطع حماسية أو مثيرة للجدل أو ذات فائدة عالية جداً.
شروط كل مقطع:
- مدته بين 30 و50 ثانية.
- يبدأ من أول الجملة ويُنهي عند اكتمال الفكرة، دون قطع في منتصف الكلام.
- الأزمنة بالثواني الصحيحة من بداية الملف الصوتي (start_time و end_time).
أعطني النتيجة حصرياً بصيغة JSON على شكل قائمة بدون أي نص إضافي:
[
  {
    "start_time": 45,
    "end_time": 90,
    "title": "عنوان جذاب جداً ومثير للشورت (Hook)",
    "reason": "لماذا سينتشر هذا المقطع؟"
  }
]
"""


def _cleanup():
    for f in glob.glob("temp_audio*"):
        try:
            os.remove(f)
        except OSError:
            pass


def _parse_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
        else:
            data = [data]
    return data


def _validate(clips):
    valid = []
    for c in clips:
        try:
            s = int(float(c["start_time"]))
            e = int(float(c["end_time"]))
        except (KeyError, TypeError, ValueError):
            continue
        if s < 0 or e <= s or (e - s) < 10 or (e - s) > 90:
            continue
        valid.append({
            "start_time": s,
            "end_time": e,
            "title": str(c.get("title", "شورت بودكاست جديد")).strip(),
            "reason": str(c.get("reason", "")).strip(),
        })
    return valid


def analyze_podcast_from_url(video_url):
    uploaded = None
    client = None
    try:
        _cleanup()
        print("جاري تنزيل ملف الصوت...")
        ok = run_ytdlp([
            "-f", "bestaudio/best",
            "-x", "--audio-format", "mp3",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -b:a 48k",
            "-o", "temp_audio.%(ext)s",
            video_url,
        ])
        files = glob.glob("temp_audio*.mp3") or glob.glob("temp_audio*")
        if not ok or not files:
            print("تعذر تنزيل الصوت من يوتيوب.")
            return []
        audio_path = files[0]

        print("جاري رفع الصوت إلى Gemini...")
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded = client.files.upload(file=audio_path)
        while uploaded.state and uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state and uploaded.state.name == "FAILED":
            print("فشلت معالجة الملف الصوتي في Gemini.")
            return []

        print("جاري تحليل البودكاست...")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        return _validate(_parse_json(response.text))
    except Exception as e:
        print(f"خطأ في التحليل الصوتي عبر Gemini: {e}")
        return []
    finally:
        _cleanup()
        try:
            if client and uploaded:
                client.files.delete(name=uploaded.name)
        except Exception:
            pass
