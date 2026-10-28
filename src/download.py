"""YouTube access through yt-dlp. For long lives we download the audio first,
then only the few video sections we actually need."""
import glob
import json
import math
import os
import subprocess

FMT = "bv*[height<=1080]+ba/b[height<=1080]/b"


def _base():
    cmd = ["yt-dlp", "--no-playlist", "--no-warnings", "--retries", "5"]
    if os.path.exists("cookies.txt"):
        cmd += ["--cookies", "cookies.txt"]
    return cmd


def _find(pattern):
    files = [f for f in glob.glob(pattern) if not f.endswith((".part", ".ytdl"))]
    return files[0] if files else None


def video_info(url):
    out = subprocess.run(_base() + ["-J", url], capture_output=True,
                         text=True, check=True).stdout
    j = json.loads(out)
    return dict(id=j["id"], title=j.get("title", ""),
                duration=j.get("duration") or 0, channel=j.get("channel", ""))


def download_audio(url, folder):
    found = _find(os.path.join(folder, "audio_raw.*"))
    if found:
        return found
    subprocess.run(_base() + ["-f", "bestaudio/best", "-o",
                              os.path.join(folder, "audio_raw.%(ext)s"), url], check=True)
    return _find(os.path.join(folder, "audio_raw.*"))


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def download_clip_source(url, folder, start, end, pad=2.0):
    """Returns (path, file_start): file_start = source time (seconds) at t=0 of the file."""
    s0 = max(0, int(start - pad))
    e0 = int(math.ceil(end + pad))
    base = os.path.join(folder, f"src_{s0}_{e0}")
    found = _find(base + ".*")
    if found:
        return found, float(s0)
    try:
        subprocess.run(_base() + [
            "-f", FMT, "--merge-output-format", "mp4",
            "--download-sections", f"*{_hms(s0)}-{_hms(e0)}",
            "--force-keyframes-at-cuts", "-o", base + ".%(ext)s", url], check=True)
        found = _find(base + ".*")
        if found:
            return found, float(s0)
    except subprocess.CalledProcessError:
        print("  section download failed, falling back to the full video")
    full = os.path.join(folder, "full.mp4")
    if not os.path.exists(full):
        subprocess.run(_base() + ["-f", FMT, "--merge-output-format", "mp4",
                                  "-o", full, url], check=True)
    return full, 0.0
