from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def suggest_caption(analysis):
    prompt = f"""
    You're an expert TikTok content strategist.

    Here's the extracted text shown inside the video (captions, overlays, etc.):
    \"\"\"
    {analysis.get('visible_text', '')}
    \"\"\"

    Here are the video stats:
    - Average brightness: {analysis['average_brightness']}
    - Focus score: {analysis['average_focus']}
    - Face presence: {analysis['face_frame_percentage']}%
    - Scene changes: {analysis['scene_changes']}

    🎯 Create a catchy, viral-style caption based on what’s happening in the video.

    Rules:
    - Must be 20 words or fewer
    - Use strong hook phrasing or emotional bait (e.g. “POV:...”, “She had no idea…”, “When you…”)
    - Include 1–2 trending hashtags if they feel natural
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
