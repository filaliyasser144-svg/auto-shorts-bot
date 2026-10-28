"""Sends each finished short to your Telegram for review. Bots can only upload
files up to 50 MB, so a light preview is sent; the full-quality file is saved in
the workflow's artifacts."""
import subprocess

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def make_preview(src, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", "scale=540:960",
                    "-c:v", "libx264", "-crf", "29", "-preset", "veryfast",
                    "-c:a", "aac", "-b:a", "96k", dst], check=True)
    return dst


def _configured():
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        return True
    print("  telegram not configured, skipping")
    return False


def send_text(text):
    if not _configured():
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": text[:4000]}, timeout=60)
        return r.ok
    except requests.RequestException:
        print("  telegram request failed")
        return False


def send_video(path, caption):
    if not _configured():
        return False
    try:
        with open(path, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000],
                      "supports_streaming": "true", "width": 540, "height": 960},
                files={"video": f}, timeout=300)
        if not r.ok:
            print("  telegram error:", r.text[:200])
        return r.ok
    except requests.RequestException:
        print("  telegram request failed")
        return False
