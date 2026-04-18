"""
AttentionX – Core Pipeline
Automated Content Repurposing Engine
Whisper → Emotional Peak Detection → Smart Vertical Crop → Karaoke Captions
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger("AttentionX")


# ══════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class Clip:
    rank: int
    start_sec: float
    end_sec: float
    emotional_score: float          # 0–100
    audio_energy: float             # 0–1
    sentiment: float                # –1 to 1
    transcript: str
    words: List[WordTimestamp]      # for karaoke
    hook_headline: str = ""
    label: str = ""
    output_path: str = ""           # final 9:16 mp4
    thumbnail_path: str = ""


@dataclass
class AnalysisResult:
    video_path: str
    duration_sec: float
    full_transcript: str
    all_words: List[WordTimestamp]
    clips: List[Clip] = field(default_factory=list)
    summary: str = ""


# ══════════════════════════════════════════════════════════════════
# 1. AUDIO – Whisper transcription + energy
# ══════════════════════════════════════════════════════════════════

class AudioProcessor:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            import whisper
            logger.info("Loading Whisper (base)…")
            self._model = whisper.load_model("base")
        return self._model

    def extract_wav(self, video_path: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = ["ffmpeg", "-y", "-i", video_path,
               "-ac", "1", "-ar", "16000", "-vn", tmp.name]
        subprocess.run(cmd, capture_output=True, timeout=180)
        return tmp.name

    def transcribe(self, wav_path: str) -> dict:
        model = self._load()
        logger.info("Transcribing…")
        return model.transcribe(wav_path, word_timestamps=True, verbose=False)

    def rms_timeline(self, wav_path: str, hop_sec: float = 1.0) -> List[Tuple[float, float]]:
        import librosa
        y, sr = librosa.load(wav_path, sr=16000)
        hop = int(sr * hop_sec)
        energies = []
        for i in range(0, len(y), hop):
            chunk = y[i:i+hop]
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
            energies.append((i / sr, rms))
        mx = max(e for _, e in energies) if energies else 1.0
        if mx == 0: mx = 1.0
        return [(t, e / mx) for t, e in energies]


# ══════════════════════════════════════════════════════════════════
# 2. EMOTIONAL PEAK DETECTION
# ══════════════════════════════════════════════════════════════════

POWER_WORDS = {
    "secret","hack","truth","reveal","viral","incredible","never","always",
    "must","you","free","now","discover","proven","best","worst","shocking",
    "amazing","insane","genius","mistake","myth","actually","honestly",
    "literally","breaking","urgent","exclusive","finally","simple","easy",
    "instantly","guaranteed","real","raw","remember","change","lesson",
    "biggest","failure","success","mindset","transform","never","thought",
}


def sentiment_score(text: str) -> float:
    try:
        from textblob import TextBlob
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def keyword_density(text: str) -> float:
    words = [w.strip(".,!?\"'").lower() for w in text.split()]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in POWER_WORDS)
    return min(hits / max(len(words), 1) * 8, 1.0)


def extract_peaks(
    whisper_result: dict,
    energy_timeline: List[Tuple[float, float]],
    duration: float,
    clip_sec: float = 60.0,
    top_n: int = 5,
) -> List[Clip]:
    """
    Slide a window over the video and score each window.
    Score = 0.35*audio_energy + 0.30*|sentiment| + 0.20*keywords + 0.15*speech_density
    """
    whisper_segs = whisper_result.get("segments", [])

    def words_in_range(start, end) -> List[WordTimestamp]:
        out = []
        for seg in whisper_segs:
            for w in seg.get("words", []):
                ws, we = w.get("start", 0), w.get("end", 0)
                word = w.get("word", "").strip()
                if we < start or ws > end or not word:
                    continue
                out.append(WordTimestamp(word=word, start=ws, end=we))
        return out

    def text_in_range(start, end) -> str:
        return " ".join(w.word for w in words_in_range(start, end))

    def energy_in_range(start, end) -> float:
        vals = [e for t, e in energy_timeline if start <= t <= end]
        return float(np.mean(vals)) if vals else 0.0

    step = clip_sec / 2  # 50% overlap for better coverage
    candidates = []
    t = 0.0
    while t + clip_sec <= duration:
        end = t + clip_sec
        text = text_in_range(t, end)
        words = words_in_range(t, end)
        energy = energy_in_range(t, end)
        sent = sentiment_score(text)
        kd = keyword_density(text)
        speech_density = min(len(words) / (clip_sec * 2.5), 1.0)

        raw = (0.35 * energy +
               0.30 * abs(sent) +       # both positive & negative emotion = viral
               0.20 * kd +
               0.15 * speech_density)
        score = round(raw * 100, 2)

        candidates.append(Clip(
            rank=0,
            start_sec=round(t, 2),
            end_sec=round(end, 2),
            emotional_score=score,
            audio_energy=energy,
            sentiment=sent,
            transcript=text,
            words=words,
        ))
        t += step

    # De-overlap: greedy NMS – keep highest score, suppress overlapping windows
    candidates.sort(key=lambda c: c.emotional_score, reverse=True)
    selected: List[Clip] = []
    used_ranges = []
    for c in candidates:
        overlap = any(
            not (c.end_sec <= s or c.start_sec >= e)
            for s, e in used_ranges
        )
        if not overlap:
            selected.append(c)
            used_ranges.append((c.start_sec, c.end_sec))
        if len(selected) >= top_n:
            break

    # Assign rank & label
    for i, clip in enumerate(selected):
        clip.rank = i + 1
        score = clip.emotional_score
        clip.label = (
            "🔥 Viral Gem"       if score >= 70 else
            "⚡ High Energy"     if score >= 50 else
            "📈 Golden Nugget"   if score >= 30 else
            "💡 Decent Insight"
        )

    return selected


# ══════════════════════════════════════════════════════════════════
# 3. HOOK HEADLINE GENERATOR  (Gemini → fallback rule-based)
# ══════════════════════════════════════════════════════════════════

def generate_hook(clip: Clip, gemini_key: Optional[str] = None) -> str:
    """Generate a catchy hook headline for a clip."""

    # Try Gemini first
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are a viral content strategist for TikTok/Reels/Shorts.\n"
                "Given this transcript excerpt, write ONE punchy hook headline (max 10 words).\n"
                "It must be attention-grabbing, curiosity-inducing, and scroll-stopping.\n"
                "Only output the headline, nothing else.\n\n"
                f"Transcript: {clip.transcript[:500]}"
            )
            resp = model.generate_content(prompt)
            hook = resp.text.strip().strip('"').strip("'")
            if hook:
                return hook
        except Exception as e:
            logger.warning(f"Gemini hook failed: {e}")

    # Rule-based fallback
    text = clip.transcript.lower()
    templates = [
        "Nobody Talks About This 🤫",
        "This Changed Everything For Me",
        "The Truth They Don't Tell You",
        "Stop Wasting Time — Watch This",
        "This 60-Second Lesson Is Gold",
        "Most People Get This Wrong",
        "The Secret No One Shares",
        "I Wish I Knew This Earlier",
    ]

    # Pick based on sentiment
    if clip.sentiment > 0.2:
        return "This Will Change How You Think 🧠"
    elif clip.sentiment < -0.2:
        return "The Hard Truth Nobody Tells You 🔥"
    elif clip.audio_energy > 0.7:
        return "This Is The Most Intense Moment 🚨"
    else:
        import random
        random.seed(int(clip.start_sec))
        return random.choice(templates)


# ══════════════════════════════════════════════════════════════════
# 4. FACE TRACKER – MediaPipe + fallback OpenCV
# ══════════════════════════════════════════════════════════════════

class FaceTracker:
    """
    Tracks speaker face across a video segment.
    Returns bounding box centres for smart vertical crop.
    """

    def __init__(self):
        self._mp = None
        self._cascade = None

    def _get_mp(self):
        if self._mp is None:
            try:
                import mediapipe as mp
                self._mp = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except Exception as e:
                logger.warning(f"MediaPipe unavailable: {e}")
        return self._mp

    def _get_cascade(self):
        if self._cascade is None:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        return self._cascade

    def track_center_x(self, video_path: str, start_sec: float, end_sec: float,
                        sample_fps: float = 2.0) -> float:
        """
        Returns the average normalised center-x (0–1) of the face in the segment.
        0.5 means perfectly centered, <0.5 = left, >0.5 = right.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
        step = max(1, int(fps / sample_fps))
        cx_values = []
        frame_idx = 0
        mp_detector = self._get_mp()
        cascade = self._get_cascade()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos_sec > end_sec:
                break
            if frame_idx % step == 0:
                h, w = frame.shape[:2]
                cx = 0.5  # default: centered
                detected = False

                # Try MediaPipe
                if mp_detector:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = mp_detector.process(rgb)
                        if result.detections:
                            det = result.detections[0]
                            bb = det.location_data.relative_bounding_box
                            cx = bb.xmin + bb.width / 2
                            detected = True
                    except Exception:
                        pass

                # Fallback: OpenCV cascade
                if not detected:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    if len(faces) > 0:
                        x, _, fw, _ = faces[0]
                        cx = (x + fw / 2) / w

                cx_values.append(cx)
            frame_idx += 1

        cap.release()
        return float(np.mean(cx_values)) if cx_values else 0.5


# ══════════════════════════════════════════════════════════════════
# 5. VERTICAL CROP RENDERER (9:16 + face-centered)
# ══════════════════════════════════════════════════════════════════

class VerticalCropRenderer:
    """
    Produces a 9:16 vertical video with:
      - Face-centered smart crop
      - Karaoke-style animated word captions
      - Hook headline banner at top
      - Score badge
    """

    TARGET_W = 1080
    TARGET_H = 1920

    def render(
        self,
        source_video: str,
        clip: Clip,
        face_cx: float,
        output_path: str,
        padding_sec: float = 0.3,
    ) -> bool:
        start = max(0.0, clip.start_sec - padding_sec)
        duration = (clip.end_sec - clip.start_sec) + padding_sec * 2

        # Get source dimensions
        cap = cv2.VideoCapture(source_video)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if src_w == 0 or src_h == 0:
            return False

        # Compute crop box: preserve height, crop width to achieve 9:16
        # Scale source to fill 9:16
        target_ratio = self.TARGET_W / self.TARGET_H   # 9/16

        # If source is landscape (wider than 9:16), crop width
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            # Crop width
            crop_h = src_h
            crop_w = int(src_h * target_ratio)
        else:
            # Source is portrait or square: crop height
            crop_w = src_w
            crop_h = int(src_w / target_ratio)

        # Face-center the crop horizontally
        ideal_cx = int(face_cx * src_w)
        crop_x = ideal_cx - crop_w // 2
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, (src_h - crop_h) // 2)

        # Build ffmpeg crop + scale + caption filter
        # Escape text for ffmpeg drawtext
        def esc(t: str) -> str:
            return (t.replace("\\", "\\\\")
                      .replace("'", "\\'")
                      .replace(":", "\\:")
                      .replace("%", "\\%"))

        hook_safe = esc(clip.hook_headline[:50])
        score_safe = esc(f"Score {clip.emotional_score:.0f}/100")

        # Build karaoke caption using enable= time conditions
        caption_filters = []
        font_size = 72
        for w in clip.words[:120]:  # cap to avoid huge filter chains
            ws = max(0.0, w.start - start)
            we = max(ws + 0.1, w.end - start)
            word_esc = esc(w.word.upper())
            caption_filters.append(
                f"drawtext=text='{word_esc}':"
                f"fontsize={font_size}:fontcolor=white:"
                f"x=(w-text_w)/2:"
                f"y=h*0.75:"
                f"box=1:boxcolor=black@0.75:boxborderw=12:"
                f"enable='between(t,{ws:.3f},{we:.3f})'"
            )

        # Hook headline (always visible)
        headline_filter = (
            f"drawtext=text='{hook_safe}':"
            f"fontsize=52:fontcolor=yellow:"
            f"x=(w-text_w)/2:y=60:"
            f"box=1:boxcolor=black@0.8:boxborderw=10"
        )

        # Score badge (top-right)
        score_filter = (
            f"drawtext=text='{score_safe}':"
            f"fontsize=36:fontcolor=white:"
            f"x=w-text_w-20:y=20:"
            f"box=1:boxcolor=#ff4444@0.9:boxborderw=8"
        )

        all_filters = [headline_filter, score_filter] + caption_filters
        vf = (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={self.TARGET_W}:{self.TARGET_H},"
        ) + ",".join(all_filters)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", source_video,
            "-t", str(duration),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        logger.info(f"Rendering 9:16 clip #{clip.rank} → {Path(output_path).name}")
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                err = result.stderr.decode()[-800:]
                logger.error(f"ffmpeg error: {err}")
                # Fallback: simple crop without captions
                return self._render_simple(source_video, clip, crop_w, crop_h,
                                           crop_x, crop_y, start, duration, output_path)
            return True
        except Exception as e:
            logger.error(f"Render failed: {e}")
            return False

    def _render_simple(self, source, clip, cw, ch, cx, cy, start, dur, out):
        """Fallback: crop + scale only, no text overlays."""
        vf = (
            f"crop={cw}:{ch}:{cx}:{cy},"
            f"scale=1080:1920"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", source,
            "-t", str(dur),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            out,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            return r.returncode == 0
        except Exception:
            return False

    def thumbnail(self, video_path: str, out_path: str, at_sec: float = 1.0) -> bool:
        cmd = [
            "ffmpeg", "-y", "-ss", str(at_sec),
            "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            return r.returncode == 0
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════
# 6. HIGHLIGHT REEL
# ══════════════════════════════════════════════════════════════════

def build_highlight_reel(clip_paths: List[str], output_path: str) -> bool:
    if not clip_paths:
        return False
    concat_file = output_path.replace(".mp4", "_concat.txt")
    with open(concat_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

class AttentionXPipeline:
    def __init__(
        self,
        clip_sec: float = 60.0,
        top_n: int = 5,
        gemini_key: Optional[str] = None,
        output_dir: str = "outputs",
    ):
        self.clip_sec = clip_sec
        self.top_n = top_n
        self.gemini_key = gemini_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio = AudioProcessor()
        self.tracker = FaceTracker()
        self.renderer = VerticalCropRenderer()

    def run(self, video_path: str, cb=None) -> AnalysisResult:
        def _cb(msg, pct):
            logger.info(f"[{pct:3d}%] {msg}")
            if cb: cb(msg, pct)

        _cb("Reading video metadata…", 3)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = frames / fps

        _cb("Extracting audio…", 8)
        wav = self.audio.extract_wav(video_path)

        _cb("Transcribing with Whisper…", 15)
        whisper_result = self.audio.transcribe(wav)
        full_text = whisper_result.get("text", "")

        _cb("Analysing audio energy…", 30)
        energy_tl = self.audio.rms_timeline(wav)

        _cb("Detecting emotional peaks…", 40)
        clips = extract_peaks(whisper_result, energy_tl, duration,
                              self.clip_sec, self.top_n)

        # Build all_words list
        all_words: List[WordTimestamp] = []
        for seg in whisper_result.get("segments", []):
            for w in seg.get("words", []):
                word = w.get("word", "").strip()
                if word:
                    all_words.append(WordTimestamp(
                        word=word, start=w.get("start", 0), end=w.get("end", 0)
                    ))

        result = AnalysisResult(
            video_path=video_path,
            duration_sec=duration,
            full_transcript=full_text,
            all_words=all_words,
            clips=clips,
        )

        n = len(clips)
        for i, clip in enumerate(clips):
            pct = 50 + int(i / max(n, 1) * 45)

            _cb(f"Generating hook for clip #{clip.rank}…", pct)
            clip.hook_headline = generate_hook(clip, self.gemini_key)

            _cb(f"Face tracking clip #{clip.rank}…", pct + 2)
            face_cx = self.tracker.track_center_x(
                video_path, clip.start_sec, clip.end_sec
            )

            _cb(f"Rendering 9:16 vertical clip #{clip.rank}…", pct + 4)
            out_name = (
                f"clip_{clip.rank:02d}_"
                f"{int(clip.start_sec)}s-{int(clip.end_sec)}s_"
                f"score{int(clip.emotional_score)}.mp4"
            )
            out_path = str(self.output_dir / out_name)
            ok = self.renderer.render(video_path, clip, face_cx, out_path)
            if ok:
                clip.output_path = out_path
                thumb = out_path.replace(".mp4", "_thumb.jpg")
                self.renderer.thumbnail(out_path, thumb, at_sec=2.0)
                if Path(thumb).exists():
                    clip.thumbnail_path = thumb

        # Highlight reel
        valid_clips = [c for c in clips if c.output_path]
        if len(valid_clips) > 1:
            _cb("Building highlight reel…", 96)
            reel_path = str(self.output_dir / "highlight_reel.mp4")
            build_highlight_reel([c.output_path for c in valid_clips], reel_path)

        result.summary = (
            f"Duration: {duration:.0f}s · "
            f"Clips found: {len(clips)} · "
            f"Top score: {clips[0].emotional_score:.1f}/100" if clips else "No clips"
        )

        # Cleanup wav
        try: os.unlink(wav)
        except: pass

        _cb("✅ Done!", 100)
        return result
