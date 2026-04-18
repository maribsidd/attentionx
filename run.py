"""
AttentionX — CLI Runner
python run.py --video my_lecture.mp4 [--gemini-key KEY] [--top-n 5] [--clip-sec 60]
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import AttentionXPipeline


def main():
    p = argparse.ArgumentParser(description="AttentionX – Automated Content Repurposing Engine")
    p.add_argument("--video", required=True)
    p.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY", ""))
    p.add_argument("--clip-sec", type=float, default=60.0)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--padding", type=float, default=0.5)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    if not Path(args.video).exists():
        print(f"❌ File not found: {args.video}")
        sys.exit(1)

    print("=" * 58)
    print("  🔥 AttentionX — Automated Content Repurposing Engine")
    print("=" * 58)

    def cb(msg, pct):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r[{bar}] {pct:3d}%  {msg:<40}", end="", flush=True)

    pipeline = AttentionXPipeline(
        clip_sec=args.clip_sec,
        top_n=args.top_n,
        gemini_key=args.gemini_key or None,
        output_dir=args.output_dir,
    )

    result = pipeline.run(args.video, cb=cb)
    print()
    print(f"\n✅ Done! Duration: {result.duration_sec:.0f}s · {len(result.clips)} clips")
    print()

    for clip in result.clips:
        print(f"#{clip.rank:02d} [{clip.start_sec:.0f}s–{clip.end_sec:.0f}s] "
              f"Score:{clip.emotional_score:.1f}  {clip.label}")
        print(f"    Hook: \"{clip.hook_headline}\"")
        print(f"    Transcript: \"{clip.transcript[:100]}…\"")
        if clip.output_path:
            print(f"    → {clip.output_path}")
        print()

    reel = Path(args.output_dir) / "highlight_reel.mp4"
    if reel.exists():
        print(f"🎞️  Highlight Reel → {reel}")
    print("\n🏁 All done!")


if __name__ == "__main__":
    main()
