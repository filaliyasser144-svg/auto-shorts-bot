import os
import subprocess

def download_and_cut_clip(video_url, start_time, end_time, output_filename="output_short.mp4"):
    try:
        print(f"جاري قص المقطع باستخدام الكوكيز...")
        download_command = [
            "yt-dlp",
            "--cookies", "cookies.txt",
            "--extractor-args", "youtube:player_client=android",
            "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--download-sections", f"*{int(start_time)}-{int(end_time)}",
            "-o", "temp_source.mp4",
            video_url
        ]
        subprocess.run(download_command, check=True)
        
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", "temp_source.mp4",
            "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_filename
        ]
        subprocess.run(ffmpeg_command, check=True)
        
        if os.path.exists("temp_source.mp4"):
            os.remove("temp_source.mp4")
            
        return output_filename
    except Exception as e:
        print(f"حدث خطأ أثناء معالجة الفيديو: {e}")
        return None
