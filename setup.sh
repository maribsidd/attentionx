#!/usr/bin/env bash
set -e
echo "═══════════════════════════════════════"
echo "  🔥 AttentionX — Setup"
echo "═══════════════════════════════════════"

# ffmpeg check
if ! command -v ffmpeg &>/dev/null; then
  echo "⚠️  Installing ffmpeg..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -qq && sudo apt-get install -y ffmpeg
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
  else
    echo "❌ Install ffmpeg from https://ffmpeg.org"
    exit 1
  fi
fi
echo "✅ ffmpeg OK"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "═══════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Web UI:  streamlit run app.py"
echo "  CLI:     python run.py --video file.mp4"
echo "═══════════════════════════════════════"
