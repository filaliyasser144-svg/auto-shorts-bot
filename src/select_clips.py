"""The "editor brain": Gemini reads the timestamped transcript and picks the best
moments. Boundaries are then snapped to real word timings so a clip never starts
or ends in the middle of a word."""
import json
import math
import re
import time

import requests

from config import (CLIP_MAX, CLIP_MIN, GEMINI_API_KEY, GEMINI_MODEL,
                    WINDOW_OVERLAP, WINDOW_SEC)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}, "score": {"type": "INTEGER"},
    "hook": {"type": "STRING"}, "title": {"type": "STRING"},
    "description": {"type": "STRING"},
    "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}},
    "reason": {"type": "STRING"}},
    "required": ["start", "end", "score", "hook", "title"]}}

PROMPT = """You are a top short-form video editor (TikTok / Reels / YouTube Shorts).
Below is a timestamped transcript of part of a long video ({lang}; it may contain Algerian Arabic dialect, French and English words).
Pick the {n} best moments to turn into standalone vertical shorts.

Rules:
- Each clip must be {cmin}-{cmax} seconds long and understandable WITHOUT any other context.
- The first 3 seconds must hook: a bold claim, a question, a surprise, a strong emotion or a story opening.
- Start at the beginning of a sentence and end after a complete thought (punchline, conclusion, cliffhanger).
- Prefer: strong opinions, funny moments, emotional stories, surprising facts, practical advice, heated debate.
- Avoid: greetings, intros/outros, sponsor reads, small talk, reading chat messages, repeated content.
- "start" and "end" are seconds, taken from the [seconds] markers in the transcript.
- "hook": a punchy on-screen opening line (max 8 words) in the speaker's language.
- "title" and "description": in the speaker's language, written to maximise views. "hashtags": 5 relevant ones without '#'.
- "score": 1-100 viral potential. Be strict; most clips should score below 80.
- If nothing is good, return fewer clips (or an empty list).

TRANSCRIPT:
{transcript}"""

LANG_NAMES = {"ar": "Arabic", "en": "English", "auto": "unknown language"}


def _lines(segments, t0, t1):
    return "\n".join(f"[{int(s['start'])}] {s['text']}"
                     for s in segments if t0 <= s["start"] < t1)


def _ask_gemini(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4,
                                 "responseMimeType": "application/json",
                                 "responseSchema": SCHEMA}}
    for attempt in range(6):
        r = requests.post(GEMINI_URL.format(model=GEMINI_MODEL),
                          headers={"x-goog-api-key": GEMINI_API_KEY},
                          json=body, timeout=180)
        if r.status_code == 200:
            try:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
                return json.loads(txt)
            except (KeyError, IndexError, json.JSONDecodeError):
                return []
        if r.status_code in (429, 500, 503):
            print(f"  gemini {r.status_code}: waiting (free-tier limit)")
            time.sleep(20 * (attempt + 1))
            continue
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Gemini: too many retries")


def _snap(c, words):
    i = min(range(len(words)), key=lambda k: abs(words[k]["start"] - c["start"]))
    j = min(range(len(words)), key=lambda k: abs(words[k]["end"] - c["end"]))
    c["start"] = max(0.0, words[i]["start"] - 0.15)
    c["end"] = words[j]["end"] + 0.35
    return c


def select_clips(tr, n, lang):
    segs, words = tr["segments"], tr["words"]
    if not segs or not words:
        return []
    total = segs[-1]["end"]
    cands, t = [], 0.0
    while t < total:
        text = _lines(segs, t, t + WINDOW_SEC)
        if len(text) > 300:
            print(f"  asking Gemini about {int(t // 60)}-{int((t + WINDOW_SEC) // 60)} min")
            prompt = PROMPT.format(lang=LANG_NAMES.get(lang, lang), n=min(6, max(3, n)),
                                   cmin=CLIP_MIN, cmax=CLIP_MAX, transcript=text)
            cands += _ask_gemini(prompt)
            time.sleep(6)
        t += WINDOW_SEC - WINDOW_OVERLAP

    good = []
    for c in cands:
        try:
            c = _snap(dict(c), words)
        except (KeyError, TypeError, ValueError):
            continue
        dur = c["end"] - c["start"]
        if CLIP_MIN - 8 <= dur <= CLIP_MAX + 15:
            good.append(c)

    picked = []
    for c in sorted(good, key=lambda c: -c.get("score", 0)):
        if all(c["end"] + 3 < p["start"] or c["start"] - 3 > p["end"] for p in picked):
            picked.append(c)
        if len(picked) >= n:
            break
    return picked
