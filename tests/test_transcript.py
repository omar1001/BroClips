"""transcript.py: lines for the AI, caption phrases (English + Arabic), the made-up-text filter, word times."""
from broclips import transcript as T


def words_from(text, start=0.0, step=0.4, dur=0.3, p=0.95, part=None):
    out = []
    for i, w in enumerate(text.split()):
        d = {"w": w, "s": round(start + i * step, 3), "e": round(start + i * step + dur, 3), "p": p}
        if part is not None:
            d["part"] = part
        out.append(d)
    return out


def texts(phr):
    return [" ".join(w["w"] for w in p["words"]) for p in phr]


def test_make_lines_breaks_on_pause_punctuation_and_length():
    ws = words_from("hello there friend.") + words_from("new line after a pause", start=3.0)
    ws += words_from(" ".join(["word"] * 40), start=10.0, step=0.5)          # 20 s without a pause
    lines = T.make_lines(ws)
    assert [ln["id"] for ln in lines[:3]] == ["L001", "L002", "L003"]
    assert lines[0]["text"] == "hello there friend."
    assert lines[1]["text"] == "new line after a pause"
    assert all(ln["e"] - ln["s"] <= 12.0 for ln in lines)
    assert lines[2]["words"][0]["s"] == 10.0


def test_english_phrases_are_sentence_aware_balanced_and_clean():
    ws = words_from("Hey! In the next two minutes you will finally understand for loops in Python. "
                    "Range stops before the end.", step=0.3, dur=0.25)
    ph = texts(T.phrases(ws))
    assert ph[0] == "Hey!"                                   # a one-word sentence stays alone
    assert all(2 <= len(p.split()) <= 6 for p in ph[1:])
    joined = " | ".join(ph)
    assert "Hey! In" not in joined and "Python Range" not in joined     # never across a sentence end
    assert not any(p.endswith(".") for p in ph)                          # no trailing full stops
    assert ph[-1] == "Range stops before the end"
    sizes = [len(p.split()) for p in ph[1:-1]]
    assert max(sizes) - min(sizes) <= 1                      # 13 words -> 5 + 4 + 4, not 6 + 6 + 1


def test_phrases_prefer_commas_and_respect_max_words():
    ws = words_from("first we open the file, then we read every line and print it", step=0.3)
    ph = texts(T.phrases(ws, max_words=6))
    assert ph[0] == "first we open the file"
    assert all(len(p.split()) <= 6 for p in ph)


def test_phrases_never_run_across_a_cut():
    ws = words_from("one two three", part=0) + words_from("four five six", start=1.2, part=1)
    ph = T.phrases(ws)
    assert texts(ph) == ["one two three", "four five six"]
    assert [p["part"] for p in ph] == [0, 1]


def test_arabic_phrases_break_at_pauses_and_keep_words():
    a = "تعالى اهو خدس الوحي اهو"
    b = "اموت بقى يا عم بقى خلاص"
    ws = words_from(a, step=0.35) + words_from(b, start=3.0, step=0.35)
    ph = texts(T.phrases(ws))
    assert ph == [a, b]
    assert T.is_arabic(ph[0]) and not T.is_arabic("Python loops")


def test_arabic_long_run_without_punctuation_is_split_evenly():
    ws = words_from("يا جماعة النهارده هنتكلم عن حاجة مهمة جدا في البرمجة وهي اللوب", step=0.3)
    ph = T.phrases(ws)
    sizes = [len(p["words"]) for p in ph]
    assert sum(sizes) == 12 and all(2 <= n <= 6 for n in sizes) and max(sizes) - min(sizes) <= 1


def test_zero_length_and_same_time_words_get_visible_times():
    ws = [{"w": "في", "s": 11.94, "e": 11.94}, {"w": "البيت", "s": 12.5, "e": 12.9},
          {"w": "a", "s": 20.0, "e": 20.0}, {"w": "b", "s": 20.0, "e": 20.0}, {"w": "c", "s": 20.01, "e": 20.3}]
    fx = T.fix_times(ws)
    assert fx[0]["e"] - fx[0]["s"] >= T.MIN_WORD_S - 1e-9
    starts = [w["s"] for w in fx[2:]]
    assert starts == sorted(starts) and len(set(starts)) == 3            # spread, no longer on top of each other
    assert all(w["e"] > w["s"] for w in fx)
    ph = T.phrases(ws)
    assert all(w["e"] > w["s"] for p in ph for w in p["words"])


def test_display_strips_trailing_punctuation_but_keeps_questions_and_numbers():
    assert T.display("end.") == "end" and T.display("وبس،") == "وبس" and T.display("why?") == "why?"
    assert T.display("3.5") == "3.5" and T.display("...") == ""


def test_hallucination_alone_at_the_end_is_dropped_but_real_speech_stays():
    talk = words_from("so that is how loops work in Python", step=0.4)
    fake = words_from("Thanks for watching!", start=12.0)
    kept, dropped = T.drop_hallucinations(talk + fake)
    assert [w["w"] for w in kept] == [w["w"] for w in talk]
    assert dropped and dropped[0]["text"] == "Thanks for watching!" and "alone" in dropped[0]["why"]
    # said for real, in the middle of loud speech: kept
    real = words_from("ok thanks for watching and see you tomorrow", step=0.4)
    per_s = [-20.0] * 10
    kept, dropped = T.drop_hallucinations(real, per_s=per_s)
    assert len(kept) == len(real) and not dropped


def test_hallucination_quiet_unsure_repeated_and_marks():
    talk = words_from("we play the game now and it is fun", step=0.4)        # 0.0 - 3.1 s, loud
    quiet = words_from("اشتركوا في القناة", start=6.0)                         # into silence
    per_s = [-18.0] * 4 + [-60.0] * 6
    kept, dropped = T.drop_hallucinations(talk + quiet, per_s=per_s)
    assert len(kept) == len(talk) and "silence" in dropped[0]["why"]
    # the "and" prefix (و) is matched too, low confidence -> dropped
    unsure = words_from("واشتركوا في القناة", start=1.0, p=0.3)
    kept, dropped = T.drop_hallucinations(unsure, per_s=[-18.0] * 5)
    assert kept == [] and "not sure" in dropped[0]["why"]
    # three times in one video -> a loop
    rep = []
    for k in range(3):
        rep += words_from("please subscribe", start=k * 20.0)
        rep += words_from("and some real talk here", start=k * 20.0 + 1.0)
    kept, dropped = T.drop_hallucinations(rep, per_s=[-18.0] * 70)
    assert len(dropped) == 3 and all("repeated" in d["why"] for d in dropped)
    assert len(kept) == 15
    # "Subtitles by <anyone>" and sound marks always go
    ws = words_from("[Music] hello friends") + words_from("Subtitles by the Amara.org community", start=5.0)
    kept, dropped = T.drop_hallucinations(ws)
    assert [w["w"] for w in kept] == ["hello", "friends"]
    assert {d["text"] for d in dropped} == {"[Music]", "Subtitles by the Amara.org community"}


def test_norm_unifies_arabic_letters_and_case():
    assert T.norm("القناة") == T.norm("القناه")
    assert T.norm("Don't Forget!") == "dont forget"
    assert T.words_of("أهلا يا جماعة") == T.words_of("اهلا يا جماعه")
