import os
import glob
import subprocess
from ytdl import run_ytdlp


def _cleanup():
    for f in glob.glob("temp_source*"):
        try:
            os.remove(f)
        except OSError:
            pass


def download_and_cut_clip(video_url, start_time, end_time, output_filename="output_short.mp4"):
    try:
        start = int(start_time)
        end = int(end_time)
        print(f"جاري تنزيل وقص المقطع من {start} إلى {end}...")
        _cleanup()

        ok = run_ytdlp([
            "-f", "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "--merge-output-format", "mp4",
            "-o", "temp_source.%(ext)s",
            video_url,
        ])
        sources = glob.glob("temp_source.*")
        if not ok or not sources:
            print("فشل تنزيل المقطع.")
            return None
        source = sources[0]

        ffmpeg_command = [
            "ffmpeg", "-y", "-i", source,
            "-vf", "crop=trunc(ih*9/16/2)*2:ih,scale=1080:1920:flags=lanczos,setsar=1",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_filename,
        ]
        subprocess.run(ffmpeg_command, check=True)
        return output_filename
    except Exception as e:
        print(f"حدث خطأ أثناء معالجة الفيديو: {e}")
        return None
    finally:
        _cleanup()
