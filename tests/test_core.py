import numpy as np

from titl import clean, media
from titl.lang import profile_for, to_latin
from titl.qc import check
from titl.segment import Word, build_cues, break_lines, split_units
from titl.srt import to_srt, ts


def words(text, start=0.0, wps=3.0, pause_after=None):
    out, t = [], start
    for w in text.split():
        out.append(Word(w, t, t + 0.8 / wps))
        t += 1 / wps
        if pause_after and w in pause_after:
            t += pause_after[w]
    return out


def test_ts():
    assert ts(3661.5) == "01:01:01,500"
    assert ts(-1) == "00:00:00,000"


def test_short_sentence_is_one_cue_one_line():
    cues = build_cues(words("Hello there."), profile_for("en"))
    assert len(cues) == 1 and cues[0].lines == ["Hello there."]
    assert cues[0].end - cues[0].start >= profile_for("en").min_dur


def test_long_sentence_splits_within_limits_and_no_qc_issues():
    p = profile_for("en")
    txt = ("I told you yesterday that if we left before the storm arrived, we would still have enough time to reach "
           "the village, but nobody listened to me, and now we are stuck here.")
    cues = build_cues(words(txt, wps=3.5), p)
    assert len(cues) >= 2
    assert all(len(l) <= p.max_chars for c in cues for l in c.lines)
    assert [i for i in check(cues, p) if i.kind in ("line-length", "overlap", "lines")] == []
    assert " ".join(c.text.replace("\n", " ") for c in cues) == txt


def test_cue_does_not_end_on_article_when_avoidable():
    p = profile_for("en")
    cues = build_cues(words("She walked all the way down to the old house near the river and knocked on the door.", wps=3), p)
    for c in cues[:-1]:
        assert c.lines[-1].split()[-1].lower() not in p.no_end


def test_pause_splits_units():
    ws = words("Wait. Okay.", pause_after={"Wait.": 1.5})
    assert len(split_units(ws)) == 2


def test_no_overlap_and_min_gap_between_fast_cues():
    p = profile_for("en")
    ws = words("Go. Now. Run. Hide.", wps=6)
    cues = build_cues(ws, p)
    for a, b in zip(cues, cues[1:]):
        assert b.start - a.end >= p.min_gap - 1e-6
        assert a.end > a.start


def test_two_lines_balanced():
    p = profile_for("en")
    ws = words("This is a sentence that needs two lines to fit properly")
    lines = break_lines(ws, p)
    assert len(lines) == 2 and abs(len(lines[0]) - len(lines[1])) < 15


def test_srt_format():
    cues = build_cues(words("Hello there."), profile_for("en"))
    s = to_srt(cues)
    assert s.startswith("1\n00:00:00,000 --> ") and s.endswith("\n\n")


def test_to_latin():
    assert to_latin("Здраво, ђаци! Љубав") == "Zdravo, đaci! Ljubav"
    assert to_latin("već latinica") == "već latinica"


def test_hallucinations():
    assert clean.is_hallucination({"text": " Thanks for watching!"})
    assert clean.is_hallucination({"text": "Titlovao: Neko"} ) is False
    assert clean.is_hallucination({"text": "Hvala na gledanju."})
    assert clean.is_hallucination({"text": "real words", "compression_ratio": 3.0})
    assert not clean.is_hallucination({"text": "I love you", "no_speech_prob": 0.1, "avg_logprob": -0.3})
    segs = [{"text": "La la la"}] * 5
    assert len(clean.drop_repeats(segs)) == 2


def test_pick_track_prefers_language_tag_and_forced():
    st = [media.AudioStream(1, 6, "5.1", "eng", "ac3", ""), media.AudioStream(2, 2, "stereo", "srp", "aac", "")]
    assert media.pick_track(st, "sr").index == 2
    assert media.pick_track(st, "en").index == 1
    assert media.pick_track(st, "bs").index == 1
    assert media.pick_track(st, "en", forced=1).index == 2


def test_audio_filter_uses_centre_for_surround():
    assert "FC" in media.audio_filter(media.AudioStream(1, 6, "5.1", "", "", ""))


def test_chunk_bounds_cut_at_silence():
    sr = media.SR
    x = (np.random.default_rng(0).standard_normal(sr * 100) * 8000).astype(np.int16)
    x[sr * 31: sr * 31 + sr // 5] = 0
    b = media.chunk_bounds(x, chunk_s=30, window_s=5)
    assert b[0] == 0 and b[-1] == len(x)
    assert sr * 31 - sr // 10 <= b[1] <= sr * 31 + sr // 5


def test_dp_avoids_stub_cue_when_sentence_must_split():
    p = profile_for("en")
    txt = "The first thing they did was pull out different ratios so I could try them side by side."
    cues = build_cues(words(txt, wps=4.5), p)
    assert len(cues) == 2
    assert all(len(c.text.replace("\n", " ")) > 25 for c in cues)


def test_cue_that_cannot_break_into_legal_lines_is_split_instead():
    p = profile_for("en")
    txt = "Then I got to try on a wax model before anything was final, and that's when I knew."
    cues = build_cues(words(txt, wps=3.5), p)
    assert all(len(l) <= p.max_chars for c in cues for l in c.lines)


def test_cyrillic_segment_survives_cleaning_after_latin_conversion():
    seg = {"text": to_latin("Где си био?"), "no_speech_prob": 0.01, "avg_logprob": -0.2}
    assert seg["text"] == "Gde si bio?"
    assert not clean.is_hallucination(seg)
    assert not clean.is_hallucination({"text": "Где си био?"})
    assert clean.is_hallucination({"text": "♪♪"})


def test_confident_thank_you_is_dialogue_but_unsure_one_is_not():
    assert not clean.is_hallucination({"text": "Thank you.", "no_speech_prob": 0.02, "avg_logprob": -0.2})
    assert clean.is_hallucination({"text": "Thank you.", "no_speech_prob": 0.5, "avg_logprob": -0.4})
    assert clean.is_hallucination({"text": "Thanks for watching!", "no_speech_prob": 0.0, "avg_logprob": 0.0})


def test_abbreviation_does_not_end_a_sentence():
    from titl.segment import is_sentence_end
    assert not is_sentence_end("Mr.") and not is_sentence_end("Dr.") and not is_sentence_end("U.S.")
    assert is_sentence_end("done.") and is_sentence_end("What?") and is_sentence_end("wait...")
    cues = build_cues(words("Okay, Mr. Belfort, sit down."), profile_for("en"))
    assert len(cues) == 1


def test_clamp_follows_speaker_pace():
    from titl.timing import clamp_segment
    ws = [Word("a", 0, 0.2), Word("b", 0.2, 0.4), Word("c", 0.4, 0.6), Word("end.", 0.6, 4.0)]
    clamp_segment(ws)
    assert abs(ws[-1].end - 1.1) < 1e-6
    assert ws[0].end - ws[0].start == 0.2
