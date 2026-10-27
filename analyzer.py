import os
import json
import subprocess
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def analyze_podcast_from_url(video_url):
    try:
        print("جاري تحميل صوت البودكاست بطريقة مضمونة...")
        
        # التحقق من وجود الكوكيز بشكل صحيح
        cookie_arg = []
        if os.path.exists("cookies.txt") and os.path.getsize("cookies.txt") > 10:
            cookie_arg = ["--cookies", "cookies.txt"]

        # استخدام صيغة مرنة ومضمونة لتحميل الصوت وتجنب أي خطأ في الصيغ
        cmd = [
            "yt-dlp",
        ] + cookie_arg + [
            "-f", "bestaudio/best",
            "-x", "--audio-format", "mp3",
            "-o", "temp_audio.mp3",
            video_url
        ]
        subprocess.run(cmd, check=True)
        
        audio_path = "temp_audio.mp3"
        if not os.path.exists(audio_path):
            for f in os.listdir("."):
                if f.startswith("temp_audio"):
                    audio_path = f
                    break
        
        print("جاري رفع الصوت إلى Gemini وتحليله استماعاً لاختيار أفضل المقاطع...")
        audio_file = genai.upload_file(audio_path)
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = """
        أنت خبير مونتاج وتسويق عبر الفيديوهات القصيرة (Reels & TikTok). 
        استمع إلى هذا الصوت بعناية (فهو بودكاست باللهجة الجزائرية)، واستخرج أفضل 3 مقاطع حماسية، مثيرة للجدل، أو تحمل فائدة عالية جداً.
        يجب أن يكون كل مقطع مدته ما بين 30 إلى 50 ثانية.
        
        أعطني النتيجة حصرياً بصيغة JSON تحتوي على قائمة بالشكل التالي بدون أي نصوص إضافية، مع تحديد الثواني بدقة (start_time و end_time بأرقام صحيحة بالثواني):
        [
          {
            "start_time": 45,
            "end_time": 90,
            "title": "عنوان جذاب جداً ومثير للشورت (Hook)",
            "reason": "لماذا هذا المقطع سينتشر؟"
          }
        ]
        """
        
        response = model.generate_content([audio_file, prompt])
        
        # تنظيف الملف الصوتي المؤقت
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        clips_data = json.loads(clean_text)
        return clips_data
        
    except Exception as e:
        print(f"خطأ في التحليل الصوتي عبر Gemini: {e}")
        return []
