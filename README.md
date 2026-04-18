<div align="center">

# 🔥 AttentionX
### Automated Content Repurposing Engine

**Turn a 60-minute lecture into a week of viral TikTok/Reels content — fully automatically.**

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-attentionx.streamlit.app-ff9000?style=for-the-badge)](https://attentionx.streamlit.app)
[![Demo Video](https://img.shields.io/badge/🎬%20Demo%20Video-Google%20Drive-ff4444?style=for-the-badge)](https://drive.google.com/file/d/1Uy2yY7ozbqgFXUbkC4HlwQuUeHn-nMlh/view?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-maribsidd%2Fattentionx-181717?style=for-the-badge&logo=github)](https://github.com/maribsidd/attentionx)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=flat-square&logo=openai)
![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-4285F4?style=flat-square&logo=google)
![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-4285F4?style=flat-square&logo=google)

</div>

---

## 📌 The Problem

The creator economy is worth **$250 billion** — yet most high-value educational content never reaches its audience.

Mentors, educators, and coaches record hours of lectures, podcasts, and workshops. But today's audiences consume content in **60-second bursts** on TikTok, Instagram Reels, and YouTube Shorts. The result? Thousands of hours of genuine wisdom sit buried in long-form videos that nobody watches to the end.

The manual alternative is painful:
- ⏱️ Watching hours of footage to find the best moments
- ✂️ Hand-editing clips and reformatting for vertical screens
- ✍️ Manually writing captions and timing them word by word
- 💡 Coming up with a catchy hook for every single clip

**AttentionX eliminates all of that in one click.**

---

## 💡 The Solution

AttentionX is an end-to-end AI pipeline that takes any long-form video as input and automatically outputs ready-to-post viral short-form clips — complete with vertical crop, animated captions, and AI-generated hook headlines.

```
Upload Video → AI Analysis → Viral Clips → Post to TikTok/Reels/Shorts
     ↑                                              ↑
  1 click                                    fully automated
```

---

## 🎬 Demo

> 📹 **[Watch Full Demo Video](https://drive.google.com/file/d/1Uy2yY7ozbqgFXUbkC4HlwQuUeHn-nMlh/view?usp=sharing)**

> 🚀 **[Try the Live App](https://attentionx.streamlit.app)**

---

## ✨ Features

### 🔍 Emotional Peak Detection
The core engine slides a window across the video and scores every segment using four multimodal signals combined into a single **Emotional Score out of 100**. Only the highest-scoring, non-overlapping windows are selected — so every clip is genuinely the best moment in its part of the video.

### 📱 Smart Vertical Crop (16:9 → 9:16)
Using **Google MediaPipe** face detection, the engine tracks where the speaker's face is in every frame and calculates the optimal crop position. The result is a perfectly framed 9:16 vertical video where the speaker is always centered — ready for TikTok, Instagram Reels, and YouTube Shorts without any manual editing.

### ✍️ Karaoke-Style Animated Captions
Every word from the **OpenAI Whisper** transcription comes with a precise timestamp. AttentionX burns word-by-word animated captions directly into the video using ffmpeg's drawtext filter with `enable='between(t,...)'` — the same technique used by professional short-form video editors. Captions appear and disappear in sync with the speaker's voice.

### 💡 AI Hook Headlines (Gemini 1.5 Flash)
For every clip, **Google Gemini 1.5 Flash** reads the transcript and generates a punchy, scroll-stopping hook headline — the kind that makes someone stop mid-scroll. The headline is burned into the top of the video as a persistent banner. Without a Gemini key, a rule-based fallback system generates hooks based on sentiment and energy signals.

### 🎞️ Highlight Reel
All extracted clips are automatically stitched together into a single highlight reel using ffmpeg's concat demuxer — giving creators one ready-to-post video that showcases the best moments of their entire session.

### 📊 Visual Analytics Dashboard
The Streamlit UI includes an interactive Plotly timeline showing the Emotional Score of every segment across the video's full duration, a radar chart breaking down each signal for the best clip, and a detailed segment table with all metrics.

---

## 🧠 Emotional Peak Scoring — How It Works

Every 60-second window of the video gets scored using four signals:

| Signal | Weight | Method |
|---|---|---|
| 🎙️ Audio Energy | **35%** | RMS amplitude calculated by librosa — louder, more passionate speech scores higher |
| 😤 Emotion Intensity | **30%** | TextBlob sentiment polarity (absolute value) — both joy and frustration are viral |
| 🔑 Power Keywords | **20%** | Detection of 40+ proven viral trigger words: "secret", "truth", "never", "hack", "mistake" |
| 🗣️ Speech Density | **15%** | Words per second from Whisper word-level timestamps — faster speech = higher energy |

**Score Bands:**

| Score | Label | What It Means |
|---|---|---|
| ≥ 70 | 🔥 Viral Gem | Post immediately |
| ≥ 50 | ⚡ High Energy | Strong candidate |
| ≥ 30 | 📈 Golden Nugget | Worth including |
| < 30 | 💡 Decent Insight | Filler content |

A **Non-Maximum Suppression** algorithm then removes overlapping windows — so if two high-scoring moments are close together, only the best one is kept, ensuring maximum diversity across clips.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 INPUT: Long-form Video                   │
│           (lecture / podcast / workshop / talk)          │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌───────────┐ ┌───────────┐ ┌───────────┐
   │  WHISPER  │ │  LIBROSA  │ │ MEDIAPIPE │
   │ Speech-to │ │   Audio   │ │   Face    │
   │   Text    │ │  Energy   │ │ Tracking  │
   │+Timestamps│ │ Timeline  │ │ Center-X  │
   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         │             │             │
         └─────────────┼─────────────┘
                       ▼
          ┌────────────────────────┐
          │  EMOTIONAL PEAK ENGINE │
          │                        │
          │  Sliding window scan   │
          │  Score every segment   │
          │  NMS deduplication     │
          │  Select Top N clips    │
          └────────────┬───────────┘
                       │
          ┌────────────▼───────────┐
          │   GEMINI 1.5 FLASH     │
          │  Hook Headline per     │
          │  clip from transcript  │
          └────────────┬───────────┘
                       │
          ┌────────────▼───────────┐
          │    FFMPEG RENDERER     │
          │                        │
          │  • Smart 9:16 crop     │
          │  • Scale to 1080x1920  │
          │  • Karaoke captions    │
          │  • Hook headline banner│
          │  • Score badge overlay │
          └────────────┬───────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  OUTPUT: Viral 9:16 Shorts + Highlight Reel + JSON Report│
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Speech-to-Text | **OpenAI Whisper** | Industry-standard accuracy with word-level timestamps |
| Audio Analysis | **librosa** | Precise RMS energy calculation for passion detection |
| Face Tracking | **MediaPipe** + OpenCV | Real-time face detection for smart vertical crop |
| Hook Generation | **Google Gemini 1.5 Flash** | 1M token context window, excellent at creative writing |
| Sentiment NLP | **TextBlob** | Fast, accurate polarity scoring for emotion detection |
| Video Rendering | **ffmpeg** | Professional video processing with caption burn-in |
| Web UI | **Streamlit** | Rapid, beautiful Python web apps |
| Visualizations | **Plotly** | Interactive charts for the analytics dashboard |

---

## 🚀 Run Locally

### 1. Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> First run downloads Whisper base model (~140MB, automatic, one-time only)

### 3. Launch

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### 4. Optional — Gemini API Key

Get a free key at [aistudio.google.com](https://aistudio.google.com) and paste it in the sidebar for AI-generated hook headlines.

---

## 🖥️ CLI Usage

```bash
# Basic
python run.py --video lecture.mp4

# Full options
python run.py --video podcast.mp4 \
  --gemini-key YOUR_KEY \
  --top-n 5 \
  --clip-sec 60 \
  --output-dir ./clips
```

---

## 📁 Project Structure

```
attentionx/
├── app.py                   # Streamlit web UI
├── run.py                   # CLI runner
├── src/
│   ├── pipeline.py          # Core AI pipeline
│   └── __init__.py
├── requirements.txt         # Python dependencies
├── packages.txt             # System packages (Streamlit Cloud)
├── Dockerfile               # Docker deployment
├── railway.toml             # Railway deployment config
├── render.yaml              # Render.com deployment config
└── .streamlit/
    └── config.toml          # Theme and server settings
```

---

## 📦 Output Files

| File | Description |
|---|---|
| `clip_01_Xs-Ys_scoreN.mp4` | 9:16 vertical short with captions and hook |
| `highlight_reel.mp4` | All top clips concatenated |
| `attentionx_report.json` | Full analysis: scores, transcripts, signals |

---

## 🌍 Deployment

Deployed on **Streamlit Community Cloud** with ffmpeg installed via `packages.txt`.

Also supports **Railway** and **Render** via included `Dockerfile`, `railway.toml`, and `render.yaml`.

---

## 📈 Impact

- A single 60-minute mentorship session becomes **5 ready-to-post viral clips**
- Reduces content repurposing time from **hours to minutes**
- No video editing skills required
- Works on any spoken-word video in any language Whisper supports

---

<div align="center">

**Built for the UnsaidTalks AttentionX AI Hackathon 2026**

*Inspired by the $250B Creator Economy*

</div>
