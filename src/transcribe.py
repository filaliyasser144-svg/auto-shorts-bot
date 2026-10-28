"""Transcription with Groq's Whisper API (free tier). Long audio is cut into
10-minute chunks; word timestamps are shifted back to the full timeline."""
import glob
import json
import os
import subprocess
import time

import requests

from config import CHUNK_SEC, GROQ_API_KEY, WHISPER_MODEL

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# small hint that helps Whisper with dialect + mixed languages
PROMPTS = {"ar": "حوار بالدارجة الجزائرية، أحياناً مع كلمات فرنسية وإنجليزية."}


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def _chunk(audio, folder):
    for f in glob.glob(os.path.join(folder, "chunk_*.mp3")):
        os.remove(f)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", audio, "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "32k", "-f", "segment", "-segment_time", str(CHUNK_SEC),
         "-reset_timestamps", "1", os.path.join(folder, "chunk_%03d.mp3")], check=True)
    return sorted(glob.glob(os.path.join(folder, "chunk_*.mp3")))


def _call_groq(path, language):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    data = [("model", WHISPER_MODEL), ("response_format", "verbose_json"),
            ("temperature", "0"), ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment")]
    if language and language != "auto":
        data.append(("language", language))
    if PROMPTS.get(language):
        data.append(("prompt", PROMPTS[language]))
    for attempt in range(8):
        with open(path, "rb") as f:
            r = requests.post(GROQ_URL, headers=headers, data=data,
                              files={"file": (os.path.basename(path), f, "audio/mpeg")},
                              timeout=300)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = float(r.headers.get("retry-after", 30 * (attempt + 1)))
            except ValueError:
                wait = 30 * (attempt + 1)
            print(f"  groq {r.status_code}: waiting {wait:.0f}s (free-tier limit)")
            time.sleep(min(wait, 900))
            continue
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Groq: too many retries")


def transcribe(audio, language, cache):
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    chunks = _chunk(audio, os.path.dirname(cache))
    segments, words, offset = [], [], 0.0
    for k, c in enumerate(chunks, 1):
        print(f"  chunk {k}/{len(chunks)}")
        res = _call_groq(c, language)
        for s in res.get("segments", []):
            if s.get("no_speech_prob", 0) > 0.85:      # whisper "hallucinations" on silence
                continue
            segments.append(dict(start=s["start"] + offset, end=s["end"] + offset,
                                 text=s["text"].strip()))
        for w in res.get("words", []):
            words.append(dict(start=w["start"] + offset, end=w["end"] + offset,
                              word=w["word"].strip()))
        offset += _duration(c)
    data = dict(language=language, segments=segments, words=words)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data
