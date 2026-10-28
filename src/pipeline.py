"""python src/pipeline.py --url <youtube url> --lang ar --clips 5"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import download
import select_clips as sel
import telegram_send as tg
import transcribe as tr
from config import GEMINI_API_KEY, GROQ_API_KEY, OUT_DIR, WORK_DIR
from virtual_cam import reframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--lang", default="ar", help="ar / en / auto")
    ap.add_argument("--clips", type=int, default=5)
    a = ap.parse_args()

    for name, val in (("GROQ_API_KEY", GROQ_API_KEY), ("GEMINI_API_KEY", GEMINI_API_KEY)):
        if not val:
            sys.exit(f"missing secret: {name}")

    info = download.video_info(a.url)
    vid = info["id"]
    folder = os.path.join(WORK_DIR, vid)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Video: {info['title']} ({int(info['duration'] // 60)} min)")

    print("1/4 audio")
    audio = download.download_audio(a.url, folder)

    print("2/4 transcription")
    transcript = tr.transcribe(audio, a.lang, os.path.join(folder, "transcript.json"))

    print("3/4 choosing clips")
    clips = sel.select_clips(transcript, a.clips, a.lang)
    if not clips:
        tg.send_text(f"No strong clips found in: {info['title']}")
        print("no clips found")
        return
    with open(os.path.join(folder, "clips.json"), "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    done = 0
    for n, c in enumerate(clips, 1):
        print(f"4/4 clip {n}/{len(clips)}  [{c['start']:.0f}s-{c['end']:.0f}s]  {c.get('title', '')}")
        try:
            src, file_start = download.download_clip_source(a.url, folder, c["start"], c["end"])
            out = os.path.join(OUT_DIR, f"{vid}_{n:02d}.mp4")
            reframe(src, out, start=c["start"] - file_start, end=c["end"] - file_start)

            meta = dict(c, source=a.url, source_title=info["title"])
            with open(out.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            tags = " ".join("#" + h.lstrip("#") for h in c.get("hashtags", []))
            caption = (f"{n}/{len(clips)}  score {c.get('score', '?')}/100\n"
                       f"{c.get('hook', '')}\n\n{c.get('title', '')}\n\n"
                       f"{c.get('description', '')}\n\n{tags}")
            preview = tg.make_preview(out, os.path.join(folder, f"preview_{n:02d}.mp4"))
            tg.send_video(preview, caption)
            done += 1
        except Exception as e:                        # one bad clip must not stop the rest
            print(f"  clip {n} failed: {e}")
    tg.send_text(f"Done: {done}/{len(clips)} shorts from \"{info['title']}\"")


if __name__ == "__main__":
    main()
