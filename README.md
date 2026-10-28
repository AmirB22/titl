# titl

Makes subtitles for full-length movies. Runs locally on your GPU (or CPU), no accounts etc fully local

```sh
titl movie.mkv -l en     # writes movie.en.srt next to the movie
titl movie.mkv -l sr     # Serbian, in Latin script
titl movie.mkv -l bs     # Bosnian
titl doctor              # checks ffmpeg and GPU setup
```

## What it does

- Transcribes with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v3`).
- Works in ~10 minute chunks and saves each one. If it crashes or you hit Ctrl+C, run the same command and it
  carries on. Changing the subtitle layout reuses the saved transcript, so that takes seconds.
- On 5.1 tracks it uses the center channel, where the dialogue is, instead of a plain downmix. It picks the audio
  track that matches your language, or use `--audio-track N`.
- Drops the stuff Whisper makes up over silence and music ("Thanks for watching" and so on).
- Splits text into proper subtitles: 42 characters per line, 2 lines, reading speed and duration limits.
  Lines don't break after words like "the" or "Mr.", and the two lines are kept balanced.
- Prints a report at the end listing cues that are too fast, too short, overlapping, and so on.

## Install

Linux with an NVIDIA GPU is what I use and test on.

```sh
sudo pacman -S --needed ffmpeg      # or: sudo apt install ffmpeg
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
titl doctor
```

The CUDA libraries (cuBLAS, cuDNN) come from pip, so you don't need the CUDA toolkit installed. First run downloads
the `large-v3` model, about 3 GB.

No GPU? It falls back to the CPU on its own (or force it with `--device cpu`).

## Speed

A 3 hour movie took 7.5 minutes on an 8 GB card, with peak VRAM around 4.4 GB.

On CPU it works too, just slower.

## Options

```
-l, --language     en, sr or bs (default en)
-o, --output       where to write the .srt (default: next to the movie)
--audio-track N    audio stream to use, counting from 0
--model            Whisper model (default large-v3)
--device           auto, cuda or cpu
--workdir          where chunks are saved (default ./.titl-work, ~400 MB for a 3 hour movie)
```

## Not great yet

- Short interjections like "Mm-hmm" or "Yeah" that overlap someone else talking can end up on screen for well
  under half a second. There were about 120 of them in the 3 hour test movie.
- No speaker detection, so overlapping speakers aren't marked with dashes. to be done
- No burned-in subtitles, translation or UI. to be done

## Tests

```sh
uv pip install -e '.[dev]'
pytest
```

They use made up data, no GPU needed.

## Thanks

Ideas and bits of logic from [stable-ts](https://github.com/jianfch/stable-ts), [srt_equalizer](https://github.com/peterk/srt_equalizer)
and [WhisperX](https://github.com/m-bain/whisperX), see [NOTICE](NOTICE).

Use it on movies you have the right to caption.

MIT 
