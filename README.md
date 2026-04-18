# 🔥 AttentionX — Automated Content Repurposing Engine

> Turn a 60-minute lecture into a week of viral TikTok/Reels content — automatically.

## 🎬 Demo Video
📹 [Watch Live Demo](https://drive.google.com/file/d/1Uy2yY7ozbqgFXUbkC4HlwQuUeHn-nMlh/view?usp=sharing)

## 🌐 Live App
🚀 [attentionx.streamlit.app](https://attentionx.streamlit.app)

---

## 🎯 What It Does

AttentionX analyses any long-form video — lectures, podcasts, workshops — and automatically produces scroll-stopping short-form content ready for TikTok, Instagram Reels, and YouTube Shorts.

- 🔍 **Emotional Peak Detection** — Finds the most impactful 60-second moments using audio energy, sentiment analysis, and power-word density
- 📱 **Smart Vertical Crop** — MediaPipe face tracking keeps the speaker centered while converting 16:9 to 9:16
- ✍️ **Karaoke Captions** — Word-by-word animated captions burned directly into every clip
- 💡 **AI Hook Headlines** — Google Gemini 1.5 Flash generates scroll-stopping titles for each clip
- 🎞️ **Highlight Reel** — All top clips stitched into one shareable video

---

## 🧠 How the Scoring Works

Every segment of the video gets an Emotional Score out of 100:

|      Signal       | Weight|               Source              |
|-------------------|-------|-----------------------------------|
| Audio Energy      |  35%  | RMS amplitude via librosa         |
| Emotion Intensity |  30%  | TextBlob sentiment analysis       |
| Power Keywords    |  20%  | Viral trigger word detection      |
| Speech Density    |  15%  | Words/sec from Whisper timestamps |

---

## 🛠️ Tech Stack

|      Layer      |        Technology       |
|-----------------|-------------------------|
| Speech-to-Text  |     OpenAI Whisper      |
| Audio Analysis  |         librosa         |
|  Face Tracking  |   MediaPipe + OpenCV    |
| Hook Generation | Google Gemini 1.5 Flash |
|  Sentiment NLP  |        TextBlob         |
| Video Rendering |         ffmpeg          |
|     Web UI      |        Streamlit        |
|     Charts      |         Plotly          |

---

## 🚀 Run Locally

```bash
# Install system dependency
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html

# Install Python dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 📁 Project Structure

```
attentionx/
├── app.py              # Streamlit web UI
├── run.py              # CLI runner
├── src/
│   └── pipeline.py     # Core engine
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud
├── Dockerfile          # For Docker deployment
└── .streamlit/
    └── config.toml     # Theme and server config
```

---

## 📦 Output

Each run produces:
- 9:16 vertical clips with captions and hook headline
- Highlight reel combining all top clips
- JSON report with scores and transcripts

---

*Built for the UnsaidTalks AttentionX AI Hackathon 2026*
