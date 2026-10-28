from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "doctor":
        dp = argparse.ArgumentParser(prog="titl doctor", description="check ffmpeg and run a real GPU inference")
        dp.add_argument("--model", default="large-v3")
        from .doctor import run
        return run(dp.parse_args(argv[1:]).model)
    ap = argparse.ArgumentParser(prog="titl", description="Subtitles for full-length films, locally.")
    ap.add_argument("video", nargs="?", help="film file; writes <film>.<lang>.srt next to it")
    ap.add_argument("-l", "--language", default="en", help="en, sr or bs (default en)")
    ap.add_argument("-o", "--output", help="output .srt path")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--audio-track", type=int, help="Nth audio stream, 0 based (default: the one tagged with the language)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--workdir", default=".titl-work", help="where checkpoints are kept (default ./.titl-work)")
    ap.add_argument("--chunk-minutes", type=float, default=10.0)
    ap.epilog = "also: titl doctor   (checks ffmpeg and runs a real GPU inference)"
    args = ap.parse_args(argv)
    if not args.video:
        ap.print_help()
        return 0

    from .lang import profile_for
    from .media import MediaError
    from .qc import check, report
    from .segment import build_cues
    from .srt import to_srt
    from .transcribe import transcribe_film

    video = Path(args.video)
    if not video.is_file():
        print(f"titl: no such file: {video}", file=sys.stderr)
        return 2
    try:
        prof = profile_for(args.language)
        words = transcribe_film(video, Path(args.workdir), args.language, args.model, args.audio_track,
                                args.device, None, args.chunk_minutes * 60)
    except (MediaError, ValueError, RuntimeError) as e:
        print(f"titl: {e}", file=sys.stderr)
        return 1
    cues = build_cues(words, prof)
    if not cues:
        print("titl: no speech found, nothing written. Wrong --audio-track or --language?", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else video.with_suffix(f".{prof.code}.srt")
    out.write_text(to_srt(cues), encoding="utf-8")
    print(f"wrote {out}\n{report(cues, check(cues, prof))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
