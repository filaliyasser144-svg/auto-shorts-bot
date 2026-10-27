import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

# إعداد مفتاح Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def get_youtube_transcript(video_url):
    try:
        # استخراج معرف الفيديو من الرابط بدقة
        if "watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = video_url.strip()
            
        # الطريقة الأبسط والأكثر استقراراً لجلب النصوص
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item['text'] for item in transcript_list])
        return full_text, transcript_list
    except Exception as e:
        print(f"خطأ في جلب النص: {e}")
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

if __name__ == "__main__":
    test_url = input("أدخل رابط يوتيوب للبودكاست: ")
    text, raw_list = get_youtube_transcript(test_url)
    if text:
        print("جاري تحليل النص عبر Gemini...")
        clips = analyze_podcast_with_gemini(text)
        print(json.dumps(clips, indent=2, ensure_ascii=False))
