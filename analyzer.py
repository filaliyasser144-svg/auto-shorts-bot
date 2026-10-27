import os
import json
import subprocess
import google.generativeai as genai

# إعداد مفتاح Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def get_youtube_transcript(video_url):
    try:
        print("جاري استخراج الترجمة باستخدام yt-dlp...")
        
        # استخدام yt-dlp لتحميل الترجمة التلقائية أو اليدوية
        cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--skip-download",
            "--sub-format", "json3",
            "--sub-lang", "ar,en",
            "-o", "temp_sub",
            video_url
        ]
        subprocess.run(cmd, check=True)
        
        # البحث عن ملف الترجمة الناتج
        sub_file = None
        for f in os.listdir("."):
            if f.startswith("temp_sub") and f.endswith(".json3"):
                sub_file = f
                break
                
        if not sub_file:
            print("لم يتم العثور على ملف ترجمة جاهز.")
            return None, None
            
        # قراءة وتحليل ملف الترجمة واستخراج النصوص
        with open(sub_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            events = data.get("events", [])
            text_list = []
            for event in events:
                segs = event.get("segs", [])
                for seg in segs:
                    text_list.append(seg.get("utf8", ""))
            full_text = " ".join(text_list)
            
        # تنظيف الملفات المؤقتة
        for f in os.listdir("."):
            if f.startswith("temp_sub"):
                os.remove(f)
                
        return full_text, []
        
    except Exception as e:
        print(f"خطأ في استخراج النص: {e}")
        return None, None

def analyze_podcast_with_gemini(transcript_text):
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    أنت خبير مونتاج وتسويق عبر الفيديوهات القصيرة (Reels & TikTok). 
    أمامك نص بودكاست بالهجة الجزائرية. أريدك أن تحلل النص وتستخرج أفضل 3 مقاطع حماسية، مثيرة للجدل، أو تحمل فائدة عالية جداً.
    يجب أن يكون كل مقطع مدته ما بين 30 إلى 50 ثانية.
    
    أعطني النتيجة حصرياً بصيغة JSON تحتوي على قائمة بالشكل التالي بدون أي نصوص إضافية:
    [
      {{
        "start_time": "الوقت بالثواني لبداية المقطع مثلاً 45",
        "end_time": "الوقت بالثواني لنهاية المقطع مثلاً 90",
        "title": "عنوان جذاب جداً ومثير للشورت (Hook)",
        "reason": "لماذا هذا المقطع سينتشر؟"
      }}
    ]

    إليك النص:
    {transcript_text[:30000]}
    """
    
    response = model.generate_content(prompt)
    try:
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        clips_data = json.loads(clean_text)
        return clips_data
    except Exception as e:
        print(f"خطأ في تحليل الذكاء الاصطناعي: {e}")
        return []
