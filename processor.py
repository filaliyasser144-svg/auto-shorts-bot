import os
import subprocess

def download_and_cut_clip(video_url, start_time, end_time, output_filename="output_short.mp4"):
    try:
        print(f"جاري قص المقطع من الزمن {start_time} إلى {end_time}...")
        
        # 1. تحميل الجزء المحدد فقط من يوتيوب باستخدام yt-dlp لتوفير الوقت والمساحة
        download_command = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--download-sections", f"*{int(start_time)}-{int(end_time)}",
            "-o", "temp_source.mp4",
            video_url
        ]
        subprocess.run(download_command, check=True)
        
        # 2. معالجة الفيديو عبر FFmpeg: تحويله إلى 9:16 بملء الشاشة تماماً (بدون خطوط سوداء)
        # هذا الفلتر يقوم بقص منتصف الفيديو الأفقي 16:9 ليصبح عمودياً 1080x1920 بالكامل
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", "temp_source.mp4",
            "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_filename
        ]
        
        subprocess.run(ffmpeg_command, check=True)
        print(f"تم إنشاء الشورت بنجاح وبدون مساحات سوداء: {output_filename}")
        
        # تنظيف الملف المؤقت
        if os.path.exists("temp_source.mp4"):
            os.remove("temp_source.mp4")
            
        return output_filename
        
    except Exception as e:
        print(f"حدث خطأ أثناء معالجة الفيديو: {e}")
        return None

if __name__ == "__main__":
    # تجربة سريعة عند التشغيل المباشر
    test_url = input("أدخل رابط يوتيوب: ")
    download_and_cut_clip(test_url, 10, 40, "test_result.mp4")
