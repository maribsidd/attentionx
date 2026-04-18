"""
AttentionX — Automated Content Repurposing Engine
Streamlit Web UI
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import AttentionXPipeline, AnalysisResult, Clip

# ─────────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttentionX | Content Repurposing Engine",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080810; color: #e8e8f0; }

.hero { text-align: center; padding: 2rem 0 1rem; }
.hero h1 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 5rem; letter-spacing: 0.08em;
  background: linear-gradient(135deg, #ff3c3c 0%, #ff9000 45%, #ffe000 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0; line-height: 1;
}
.hero p { color: #666; font-size: 1rem; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 6px; }

.stat-box {
  background: #10101c; border: 1px solid #1e1e3a;
  border-radius: 14px; padding: 20px; text-align: center;
}
.stat-val { font-family: 'Bebas Neue', sans-serif; font-size: 3.2rem; color: #ff9000; line-height: 1; }
.stat-lbl { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

.clip-card {
  background: #0e0e1a; border: 1px solid #1e1e3a;
  border-radius: 16px; padding: 22px; margin-bottom: 16px;
  border-left: 5px solid #ff3c3c;
  transition: border-color 0.2s;
}
.clip-rank { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: #ff9000; line-height: 1; }
.hook { font-size: 1.1rem; font-weight: 700; color: #ffe000; margin: 6px 0; }
.transcript-preview { font-size: 0.85rem; color: #555; font-family: 'JetBrains Mono', monospace; line-height: 1.6; }

.badge {
  display: inline-block; padding: 3px 14px; border-radius: 20px;
  font-size: 0.78rem; font-weight: 700; margin-bottom: 8px;
}
.badge-viral    { background: #ff3c3c22; color: #ff3c3c; border: 1px solid #ff3c3c55; }
.badge-high     { background: #ff900022; color: #ff9000; border: 1px solid #ff900055; }
.badge-nugget   { background: #44bb4422; color: #44bb44; border: 1px solid #44bb4455; }
.badge-default  { background: #55555522; color: #888;    border: 1px solid #55555555; }

.feature-card {
  background: #0e0e1a; border: 1px solid #1a1a30;
  border-radius: 12px; padding: 28px 22px; text-align: center;
}
.feature-icon { font-size: 2.8rem; margin-bottom: 12px; }
.feature-title { font-weight: 700; color: #ccc; margin-bottom: 6px; }
.feature-desc { font-size: 0.83rem; color: #555; line-height: 1.6; }

.stProgress > div > div { background: linear-gradient(90deg, #ff3c3c, #ff9000, #ffe000) !important; }
.stButton > button { background: linear-gradient(135deg, #ff3c3c, #ff9000) !important;
  color: white !important; border: none !important; font-weight: 700 !important;
  border-radius: 10px !important; }
section[data-testid="stSidebar"] { background: #0a0a14 !important; border-right: 1px solid #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────
for k, v in [("result", None), ("output_dir", ""), ("reel_path", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    # Auto-load from Streamlit Cloud secrets if available
    _default_gemini = ""
    try:
        _default_gemini = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        _default_gemini = os.environ.get("GEMINI_API_KEY", "")

    gemini_key = _default_gemini
    st.success("✅ Gemini API configured")
    clip_sec = st.slider("Clip Duration (sec)", 15, 120, 60, 15,
                         help="Length of each extracted short")
    top_n = st.slider("Number of Clips", 1, 8, 5)
    padding = st.slider("Clip Padding (sec)", 0.0, 3.0, 0.5, 0.5)

    st.markdown("---")
    st.markdown("### 📊 Emotional Peak Weights")
    weights = {
        "Audio Energy": 35,
        "Emotion Intensity": 30,
        "Power Keywords": 20,
        "Speech Density": 15,
    }
    for k, v in weights.items():
        st.markdown(f"<div style='display:flex;justify-content:space-between'>"
                    f"<span style='color:#888;font-size:0.82rem'>{k}</span>"
                    f"<span style='color:#ff9000;font-weight:700;font-size:0.82rem'>{v}%</span>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#333;font-size:0.72rem;line-height:1.6'>"
        "AttentionX · UnsaidTalks Hackathon 2026<br>"
        "Whisper · MediaPipe · Gemini · OpenCV · ffmpeg"
        "</div>", unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ATTENTIONX</h1>
  <p>🔥 Automated Content Repurposing Engine · Long-form → Viral Shorts</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your long-form video here",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Upload a lecture, podcast, or workshop recording"
)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(uploaded.read())
        video_path = tf.name

    st.video(video_path)

    if st.button("🚀 Extract Viral Shorts", use_container_width=True, type="primary"):

        out_dir = tempfile.mkdtemp(prefix="attentionx_")
        st.session_state.output_dir = out_dir

        progress = st.progress(0)
        status = st.empty()

        def cb(msg, pct):
            progress.progress(pct / 100)
            status.markdown(f"**{msg}**")

        pipeline = AttentionXPipeline(
            clip_sec=clip_sec,
            top_n=top_n,
            gemini_key=gemini_key or None,
            output_dir=out_dir,
        )

        with st.spinner(""):
            result = pipeline.run(video_path, cb=cb)

        st.session_state.result = result
        reel = Path(out_dir) / "highlight_reel.mp4"
        st.session_state.reel_path = str(reel) if reel.exists() else ""

        progress.progress(1.0)
        status.success("✅ Done! Your viral shorts are ready.")
        time.sleep(0.5)
        status.empty()
        progress.empty()
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────
result: AnalysisResult = st.session_state.result

if result:
    st.markdown("---")

    # ── Stats Row ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{result.duration_sec:.0f}s", "Video Duration"),
        (str(len(result.clips)), "Viral Clips Found"),
        (f"{result.clips[0].emotional_score:.0f}" if result.clips else "—", "Peak Score /100"),
        (f"{len(result.all_words)}", "Words Transcribed"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-box">
              <div class="stat-val">{val}</div>
              <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Timeline chart ────────────────────────────────────────────
    if result.clips:
        st.markdown("### 📈 Emotional Peak Timeline")
        df = pd.DataFrame([{
            "Start (s)": c.start_sec,
            "Score": c.emotional_score,
            "Label": c.label,
            "Hook": c.hook_headline,
            "Energy": c.audio_energy,
        } for c in result.clips])

        colors = []
        for s in df["Score"]:
            colors.append(
                "#ff3c3c" if s >= 70 else
                "#ff9000" if s >= 50 else
                "#44bb44" if s >= 30 else "#555"
            )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Start (s)"], y=df["Score"],
            marker_color=colors,
            hovertext=[f"#{i+1} {h}" for i, h in enumerate(df["Hook"])],
            hovertemplate="<b>%{x:.0f}s</b><br>Score: %{y:.1f}<br>%{hovertext}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df["Start (s)"], y=df["Score"],
            mode="markers+text",
            marker=dict(size=14, color="#ffe000", symbol="star"),
            text=[f"#{i+1}" for i in range(len(df))],
            textposition="top center",
            textfont=dict(color="#ffe000", size=11),
            name="Selected Clips",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080810",
            plot_bgcolor="#0e0e1a",
            font=dict(color="#aaa"),
            xaxis_title="Video Position (seconds)",
            yaxis_title="Emotional Score",
            yaxis_range=[0, 110],
            margin=dict(l=40, r=20, t=20, b=40),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Highlight Reel ─────────────────────────────────────────────
    if st.session_state.reel_path and Path(st.session_state.reel_path).exists():
        st.markdown("### 🎞️ Highlight Reel (All Top Clips)")
        st.video(st.session_state.reel_path)
        with open(st.session_state.reel_path, "rb") as f:
            st.download_button("⬇️ Download Highlight Reel",
                               f.read(), file_name="highlight_reel.mp4",
                               mime="video/mp4")

    # ── Individual Clips ──────────────────────────────────────────
    st.markdown("### ✂️ Extracted Viral Shorts (9:16)")

    for clip in result.clips:
        badge_cls = (
            "badge-viral"   if clip.emotional_score >= 70 else
            "badge-high"    if clip.emotional_score >= 50 else
            "badge-nugget"  if clip.emotional_score >= 30 else
            "badge-default"
        )
        with st.container():
            st.markdown(f"""
            <div class="clip-card">
              <div class="clip-rank">#{clip.rank}</div>
              <span class="badge {badge_cls}">{clip.label}</span>
              <div class="hook">"{clip.hook_headline}"</div>
              <div style="color:#888;font-size:0.82rem;margin-bottom:8px">
                ⏱ {clip.start_sec:.1f}s – {clip.end_sec:.1f}s &nbsp;·&nbsp;
                🔥 Score: <b style="color:#ff9000">{clip.emotional_score:.1f}/100</b> &nbsp;·&nbsp;
                😊 Sentiment: {'+' if clip.sentiment>=0 else ''}{clip.sentiment:.2f}
              </div>
              <div class="transcript-preview">"{clip.transcript[:200]}…"</div>
            </div>""", unsafe_allow_html=True)

            col_vid, col_dl = st.columns([3, 1])
            with col_vid:
                if clip.output_path and Path(clip.output_path).exists():
                    st.video(clip.output_path)
                else:
                    st.info("Video rendering in progress or failed — check outputs/ folder")
            with col_dl:
                if clip.output_path and Path(clip.output_path).exists():
                    with open(clip.output_path, "rb") as f:
                        st.download_button(
                            f"⬇️ Download Short #{clip.rank}",
                            f.read(),
                            file_name=Path(clip.output_path).name,
                            mime="video/mp4",
                            key=f"dl_clip_{clip.rank}"
                        )

    # ── Transcript ────────────────────────────────────────────────
    if result.full_transcript:
        with st.expander("📝 Full Transcript (Whisper)"):
            st.text_area("", result.full_transcript, height=250, label_visibility="collapsed")

    # ── JSON Export ───────────────────────────────────────────────
    export_data = {
        "video": result.video_path,
        "duration_sec": result.duration_sec,
        "clips": [
            {
                "rank": c.rank,
                "start": c.start_sec,
                "end": c.end_sec,
                "score": c.emotional_score,
                "label": c.label,
                "hook": c.hook_headline,
                "transcript": c.transcript[:300],
                "sentiment": c.sentiment,
                "audio_energy": c.audio_energy,
            }
            for c in result.clips
        ],
    }
    st.download_button(
        "📄 Download JSON Report",
        json.dumps(export_data, indent=2),
        file_name="attentionx_report.json",
        mime="application/json",
    )

# ─────────────────────────────────────────────────────────────────
# Landing Features (no result yet)
# ─────────────────────────────────────────────────────────────────
else:
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    features = [
        ("🎙️", "Emotional Peak Detection",
         "Whisper transcription + RMS audio energy + sentiment analysis identifies the most impactful 60-second windows in your video."),
        ("📱", "Smart Vertical Crop",
         "MediaPipe face tracking keeps your speaker centered as we convert 16:9 horizontal to 9:16 vertical for TikTok, Reels & Shorts."),
        ("✍️", "Karaoke Captions + Hook",
         "Word-by-word animated captions keep viewers engaged. Gemini AI generates a scroll-stopping hook headline for every clip."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
              <div class="feature-icon">{icon}</div>
              <div class="feature-title">{title}</div>
              <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div style="text-align:center;color:#333;font-size:0.9rem;padding:20px">
      Upload a long-form video above to get started.
      Works with lectures, podcasts, workshops, interviews, and more.
    </div>""", unsafe_allow_html=True)
