# 🔥 AttentionX — Automated Content Repurposing Engine

> **UnsaidTalks · AttentionX AI Hackathon 2026**
> *Turn a 60-minute lecture into a week of viral TikTok/Reels content — automatically.*

---

## 🎬 Demo Video

> 📹 **[Watch Live Demo → PASTE YOUR GOOGLE DRIVE LINK HERE]**
>
> *(Record a screen capture of the app running, upload to Google Drive with "Anyone with link can view", paste the URL above)*

---

## 🌐 Live Hosted App

> 🚀 **[Try it live → PASTE YOUR DEPLOYED URL HERE]**
>
> *(See hosting guide below — Streamlit Cloud is free and takes 5 minutes)*

---

## 🎯 Problem Solved

Mentors, educators, and creators produce hours of high-value content. But modern audiences consume in 60-second bursts. AttentionX solves this automatically:

1. 🔍 **Emotional Peak Detection** — Whisper + librosa audio energy + sentiment finds the most impactful moments
2. 📱 **Smart 9:16 Crop** — MediaPipe face tracking keeps the speaker centered for TikTok/Reels/Shorts
3. ✍️ **Karaoke Captions** — Word-by-word animated captions burned into every clip
4. 💡 **AI Hook Headlines** — Gemini 1.5 Flash generates scroll-stopping titles
5. 🎞️ **Highlight Reel** — All top clips stitched into one shareable video

---

## 🚀 Local Setup (Run on Your Computer)

### Step 1 — Install ffmpeg

**Windows:**
1. Download from https://ffmpeg.org/download.html (get the "essentials" build)
2. Extract and add the `bin/` folder to your system PATH
3. Test: open Command Prompt and type `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

### Step 2 — Install Python packages

```bash
pip install -r requirements.txt
```

> First run downloads the Whisper model (~140MB automatically, one time only).

### Step 3 — Launch the app

```bash
streamlit run app.py
```

Your browser opens **http://localhost:8501** automatically.

### Step 4 (Optional) — Gemini API Key for better hook headlines

1. Go to https://aistudio.google.com → click "Get API Key" (free)
2. Paste your key in the app sidebar, OR set an environment variable:

```bash
# Mac/Linux
GEMINI_API_KEY="your_key_here" streamlit run app.py

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"; streamlit run app.py
```

---

## 🌍 Hosting Guide — Get a Public URL (Free!)

### Option A: Streamlit Community Cloud — RECOMMENDED (Free, ~5 minutes)

1. Push this repo to **GitHub** (repo must be public)
2. Go to **https://share.streamlit.io** and sign in with GitHub
3. Click **"New app"**
4. Set:
   - Repository: `your-github-username/attentionx`
   - Branch: `main`
   - Main file: `app.py`
5. Click **"Advanced settings"** → Secrets → paste:
   ```
   GEMINI_API_KEY = "your_gemini_key_here"
   ```
6. Click **Deploy** — you get a free `*.streamlit.app` public URL!

> `packages.txt` is already in the repo so ffmpeg installs automatically on Streamlit Cloud.

### Option B: Railway (Docker-based, free tier)

1. Go to **https://railway.app** → New Project → Deploy from GitHub repo
2. Railway auto-detects the `Dockerfile`
3. Add env variable: `GEMINI_API_KEY=your_key`
4. Deploy → public URL in ~3 minutes

### Option C: Render.com (free tier)

1. Go to **https://render.com** → New → Web Service → connect GitHub repo
2. Render reads `render.yaml` automatically
3. Add `GEMINI_API_KEY` in the Environment tab
4. Deploy

---

## 🧠 How It Works

```
INPUT: Long-form Video (lecture / podcast / workshop)
          |
    +-----+------+---------------+
    |            |               |
 WHISPER      LIBROSA        MEDIAPIPE
 Transcribe   Audio RMS      Face Track
 + Timestamps Energy         Center-X
    |            |               |
    +-----------+++---------------+
                 |
    +------------+------------------+
    |   EMOTIONAL PEAK ENGINE       |
    |  Score = 0.35 x Energy        |
    |        + 0.30 x Emotion       |
    |        + 0.20 x Keywords      |
    |        + 0.15 x Speech Rate   |
    |  -> Top N windows selected    |
    +------------+------------------+
                 |
    +------------+------------------+
    |  GEMINI 1.5 FLASH             |
    |  -> Hook Headline per clip    |
    +------------+------------------+
                 |
    +------------+------------------+
    |  FFMPEG RENDERER              |
    |  - Smart crop 16:9 to 9:16   |
    |  - Karaoke word captions      |
    |  - Hook banner (top)          |
    |  - Score badge (corner)       |
    +------------+------------------+
                 |
OUTPUT: 9:16 Viral Shorts + Highlight Reel + JSON Report
```

---

## 📊 Emotional Peak Scoring

| Signal | Weight | Method |
|---|---|---|
| Audio Energy | **35%** | RMS amplitude peaks via librosa |
| Emotion Intensity | **30%** | TextBlob sentiment abs value — joy and frustration both go viral |
| Power Keywords | **20%** | 40+ viral trigger words: "secret", "never", "truth", "hack" |
| Speech Density | **15%** | Words per second from Whisper word-level timestamps |

**Score Thresholds:**
| Score | Label |
|---|---|
| >= 70 | Viral Gem |
| >= 50 | High Energy |
| >= 30 | Golden Nugget |
| < 30 | Decent Insight |

---

## 📁 Project Structure

```
attentionx/
├── app.py                    # Streamlit web UI (main entry point)
├── run.py                    # CLI runner
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages for Streamlit Cloud (ffmpeg)
├── Dockerfile                # For Railway / Render / self-hosting
├── railway.toml              # Railway deployment config
├── render.yaml               # Render.com deployment config
├── setup.sh                  # One-shot local setup script
├── .env.example              # Environment variable template
├── .streamlit/
│   ├── config.toml           # Streamlit theme and server settings
│   └── secrets.toml.example  # API key template for Streamlit Cloud
├── src/
│   ├── pipeline.py           # Full engine: audio to peaks to face to render
│   └── __init__.py
└── outputs/                  # Generated clips (auto-created, git-ignored)
```

---

## 📦 Output Files

| File | Description |
|---|---|
| `clip_01_Xs-Ys_scoreN.mp4` | 9:16 vertical short with captions and headline |
| `highlight_reel.mp4` | All top clips concatenated |
| `attentionx_report.json` | Full analysis data: scores, transcripts, hooks |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Speech-to-Text | OpenAI Whisper (runs locally, no API cost) |
| Audio Analysis | librosa (RMS energy timeline) |
| Face Tracking | MediaPipe + OpenCV Haar Cascade fallback |
| Hook Generation | Google Gemini 1.5 Flash (optional, free tier) |
| Sentiment NLP | TextBlob |
| Video Rendering | ffmpeg (crop + scale + karaoke drawtext) |
| Web UI | Streamlit |
| Charts | Plotly |

---

## 🔧 CLI Usage

```bash
# Basic
python run.py --video lecture.mp4

# Full options
python run.py --video podcast.mp4 --gemini-key YOUR_KEY --top-n 5 --clip-sec 60 --output-dir ./clips
```

---

## ❓ Troubleshooting

**"ffmpeg not found"** — Follow Step 1 above. On Windows, restart terminal after PATH change.

**Slow first run** — Whisper downloads ~140MB model once, then it's cached.

**No captions visible** — Video needs speech. Silent/music-only videos won't transcribe.

**Hook looks generic** — Add a Gemini API key. Without it, rule-based fallback is used.

---

## 📝 Submission Checklist

- [x] Public GitHub repository with full source code
- [ ] Record demo video → upload to Google Drive → update link at top of this README
- [ ] Deploy to Streamlit Cloud → update live URL at top of this README
- [ ] Submit GitHub repo link on Unstop portal

---

*Built for the $250B Creator Economy · UnsaidTalks AttentionX AI Hackathon 2026*
