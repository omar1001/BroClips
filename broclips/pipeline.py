"""The per-video work (SPEC §6), as jobs of the work line (jobs.py):

  make        1 prepare (probe, 16 kHz sound) -> 2 listen (speech-to-text, made-up text dropped, lines, loudness)
              -> 3 find moments (normal search, + Marathon mode when the project has it on) -> 4 Shorts
              -> 5 the long video (jump cuts) -> 6 texts (titles, descriptions, hashtags, chapters) + thumbnails
  find_again  the creator's wishes -> new moments -> new Shorts (+ their texts and thumbnails); the old Shorts are
              replaced only after the new ones are all made (Shorts marked ✅ keep always stay)
  short_edit  make one Short again (or all of them) after an edit on the video page

Every stage is marked done in <video>/stages.json, so a job that was paused (a "pause while these programs run"
program started), stopped or crashed continues where it was. progress() is called often: it is where ⏹ Stop and the
pause take effect (it raises jobs.Cancelled / jobs.Paused — never caught here). One AI model in memory at a time:
the listening model is freed before the thinking model starts, and the thinking model before rendering; the work
line frees everything after the job. Source videos are only read."""
import shutil
import time
from datetime import datetime
from pathlib import Path

from . import config, jobs, longvideo, marathon, media, projects, providers, shorts, texts
from . import moments as M
from . import transcript as T
from .util import load_json, log, save_json

STEPS = 6
LABEL = {1: "Getting the video ready", 2: "Listening to what is said", 3: "Finding the best moments",
         4: "Making the Shorts", 5: "Making the long video", 6: "Writing titles and making thumbnails"}
MIN_FREE_GB = 3


def _now():
    return datetime.now().isoformat(timespec="seconds")


class Video:
    """One video of one project: its folder, its facts, the project's settings, its stages."""

    def __init__(self, pid, vid):
        if not (projects.valid_id(pid) and projects.valid_id(vid)):
            raise jobs.JobError("This video is not in the project any more.")
        self.pid, self.vid = pid, vid
        self.dir = projects.video_dir(pid, vid)
        self.meta = load_json(self.dir / "video.json")
        self.project = projects.get(pid)
        if not isinstance(self.meta, dict) or not self.project:
            raise jobs.JobError("This video is not in the project any more.")
        self.src = Path(str(self.meta.get("src") or ""))
        self.name = self.meta.get("name") or self.src.stem
        self.settings = config.settings()
        self.boxes = self.settings.get("blur_boxes") or []

    def f(self, name):
        return self.dir / name

    def stages(self):
        return load_json(self.f("stages.json"), {}) or {}

    def mark(self, stages, key, value=True):
        stages[key] = value
        save_json(stages, self.f("stages.json"))

    def info(self):
        info = load_json(self.f("info.json"))
        if not isinstance(info, dict) or not info.get("width"):
            info = media.probe(self.src)
            save_json(info, self.f("info.json"))
        return info

    def words(self):
        return (load_json(self.f("transcript.json"), {}) or {}).get("words") or []

    def lines(self):
        return load_json(self.f("lines.json"), []) or []

    def spikes(self):
        return (load_json(self.f("loudness.json"), {}) or {}).get("spikes") or []

    def language(self):
        lang = str(self.project.get("language") or "auto")
        if lang == "auto":
            lang = (load_json(self.f("transcript.json"), {}) or {}).get("language") or ""
        return lang

    def layout(self):
        return shorts.layout_for(self.project, self.meta)

    def brain(self, llm, step):
        return M.Brain(llm, self.project, language=self.language(), wishes=self.meta.get("wishes") or "",
                       log_path=self.f("ai_log.jsonl"), step=step)

    def log_note(self, kind, text, at=None, title=None):
        M.Brain(None, log_path=self.f("ai_log.jsonl")).note(kind, text, at=at, title=title)


def _check_source(v):
    if not v.src.is_file():
        raise jobs.JobError(f"The video file is not there any more: {v.src}. Was it moved or renamed? Remove the "
                            "video from the project and add it again from its new place.")


def _check_space(v):
    try:
        free = shutil.disk_usage(v.dir).free / 1e9
    except OSError:
        return
    if free < MIN_FREE_GB:
        raise jobs.JobError(f"The disk with BroClips' output folder is almost full ({free:.1f} GB free). Make some "
                            "room (at least 3 GB), then press ✨ Make again — it continues where it stopped.")


def _rm(path):
    p = Path(path)
    for _ in range(20):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            return
        except PermissionError:
            time.sleep(0.25)
        except FileNotFoundError:
            return


def _fresh_log(v, text):
    try:
        v.f("ai_log.jsonl").write_text("", encoding="utf-8")
    except OSError:
        pass
    v.log_note("step", text)


def _who(llm=None, stt=None):
    if llm is not None:
        return f"{llm.name}" + (f" ({llm.model})" if getattr(llm, "model", "") else "")
    return f"{stt.name}" + (f" ({stt.model_opt})" if getattr(stt, "model_opt", "") not in ("", "auto") else "")


# ---------- the steps ----------
def step_prepare(v, st, progress):
    progress(1, LABEL[1], "checking the file")
    _check_source(v)
    info = media.probe(v.src)
    if not info.get("has_video"):
        raise jobs.JobError("This file has no picture. BroClips needs a video.")
    if int(info.get("n_audio") or 0) < 1:
        raise jobs.JobError("This video has no sound. BroClips needs someone talking in it.")
    save_json(info, v.f("info.json"))
    wav = v.f("audio16k.wav")
    if not (st.get("prepare") and wav.is_file()):
        progress(1, LABEL[1], "taking out the sound", 0.3)
        media.extract_audio16k(v.src, wav, info["n_audio"])
        v.mark(st, "prepare")
    return info


def step_listen(v, st, progress):
    if st.get("listen") and v.f("transcript.json").is_file() and v.f("lines.json").is_file():
        return
    progress(2, LABEL[2], "starting the listening AI")
    stt = providers.get_stt(v.settings)
    lang = str(v.project.get("language") or "auto")
    t0 = time.monotonic()
    res = stt.transcribe(str(v.f("audio16k.wav")), None if lang == "auto" else lang, "",
                         on_progress=lambda f: progress(2, LABEL[2], f"{int(f * 100)} % listened", f))
    try:
        stt.unload()                              # one model at a time: the thinking AI comes next
    except Exception as ex:
        log.info("listening unload: %s", ex)
    words = T.clean_words(res.get("words"))
    progress(2, LABEL[2], "measuring the loudness", 0.98)
    try:
        per_s, spikes = media.loudness_spikes(v.f("audio16k.wav"))
    except Exception as ex:                      # only a hint: never fail the job for it
        log.warning("loudness: %s", ex)
        per_s, spikes = [], []
    kept, dropped = T.drop_hallucinations(words, per_s)
    language = str(res.get("language") or (lang if lang != "auto" else "")).split("-")[0]
    save_json({"language": language, "provider": stt.name, "words": kept, "dropped": dropped, "made": _now()},
              v.f("transcript.json"))
    save_json({"per_s": per_s, "spikes": spikes}, v.f("loudness.json"))
    lines = T.make_lines(kept)
    save_json(lines, v.f("lines.json"))
    v.log_note("listen", f"listened with {_who(stt=stt)} in {time.monotonic() - t0:.0f} s: {len(kept)} words, "
                         f"{len(lines)} lines, language {T.language_name(language)}")
    for d in dropped:
        v.log_note("clean", f"dropped «{d['text']}» from the transcript: {d['why']}", at=d["s"])
    if not kept:
        v.log_note("warn", "nothing was heard in this video (no speech?) - there will be no Shorts")
    v.mark(st, "listen")


def step_moments(v, st, progress, kept=()):
    """Find the moments -> moments.json. `kept` = Shorts the creator marked ✅ keep (new moments avoid them)."""
    if st.get("moments") and v.f("moments.json").is_file():
        return load_json(v.f("moments.json"), []) or []
    lines, spikes, info = v.lines(), v.spikes(), v.info()
    total = float(info.get("dur") or (lines[-1]["e"] if lines else 0))
    marathon_on = bool(v.project.get("marathon"))
    span = [0.0, 0.25 if marathon_on else 1.0]

    def step(detail="", frac=None):
        f = None if frac is None else span[0] + (span[1] - span[0]) * max(0.0, min(1.0, float(frac)))
        progress(3, LABEL[3], detail, f)
    progress(3, LABEL[3], "starting the thinking AI")
    llm = providers.get_llm(v.settings)
    brain = v.brain(llm, step)
    brain.note("step", f"thinking AI: {_who(llm=llm)}; {len(lines)} lines of talk in a {T.mmss(total)} video; "
                       f"Shorts of {brain.min_s}-{brain.max_s} s" + ("; Marathon mode is on" if marathon_on else ""))
    if brain.instructions:
        brain.note("you", "your wishes for every question: " + brain.instructions.replace("\n", " / "))
    for t, line in brain.hints:
        brain.note("you", f"you pointed at {T.mmss(t)}: \"{line}\"", at=t)
    found = M.find(brain, lines, spikes, total)
    if marathon_on and lines:
        span[:] = [0.25, 1.0]
        found = marathon.search(brain, lines, spikes, total, found, cache_path=v.f("marathon_cache.json"))
    n = int(v.project.get("max_shorts") or v.settings.get("max_shorts") or 8)
    if kept:
        before = len(found)
        found = [m for m in found if not any(M.overlap(m, k) > 0.5 for k in kept)]
        if before != len(found):
            brain.note("drop", f"{before - len(found)} moment(s) are already a Short you keep")
    n = max(0, n - len(kept))
    if not found and lines:
        found = M.fallback(brain, lines, spikes, total, max(1, n))
    for m in found[n:]:
        brain.note("drop", f"not in the top {n}", at=m["s"], title=m.get("title"))
    chosen = found[:n]
    brain.note("done", f"{len(chosen)} moment(s) chosen ({brain.questions} questions to the AI) - now making the videos")
    save_json(chosen, v.f("moments.json"))
    try:
        llm.unload()                              # free the graphics card for rendering
    except Exception as ex:
        log.info("thinking unload: %s", ex)
    v.mark(st, "moments")
    return chosen


def _render_one(v, info, data, mp4, ass_name, progress, step, label, detail, frac):
    def tick():
        progress(step, label, detail, frac)
    res = shorts.render(v.src, info, data, v.words(), mp4, config.cache_dir() / "ass" / ass_name,
                        data.get("layout") or v.layout(), v.boxes, tick=tick)
    data.update(phrases=res["phrases"], dur=res["dur"], layout=res["layout"], made=_now())
    return data


def step_shorts(v, st, progress, chosen):
    """Render the Shorts. When the video already had Shorts (make again / find again) the new ones are made in
    shorts_new/ and replace the old ones only at the end."""
    if st.get("shorts"):
        return
    _check_source(v)
    _check_space(v)
    info = v.info()
    staging = bool(st.get("staging"))
    folder = "shorts_new" if staging else "shorts"
    layout = v.layout()
    for i, m in enumerate(chosen, 1):
        mp4, js = shorts.paths(v.dir, i, folder)
        if mp4.is_file() and js.is_file():
            continue                                 # made before a pause
        detail = f"{i} of {len(chosen)}"
        progress(4, LABEL[4], detail, (i - 1) / max(1, len(chosen)))
        data = shorts.from_moment(m, i, layout)
        data = _render_one(v, info, data, mp4, f"{v.pid}-{v.vid}-{folder}-{i:02d}.ass", progress, 4, LABEL[4],
                           detail, (i - 1) / max(1, len(chosen)))
        data["version"] = 1
        shorts.save(v.dir, i, data, folder)
        v.log_note("render", f"Short {i} made ({data['dur']:.0f} s, layout {data['layout']})", at=m["s"],
                   title=data.get("title"))
    if staging:
        _swap_shorts(v)
    v.mark(st, "shorts")


def _swap_shorts(v):
    """shorts_new/ -> shorts/: the old Shorts go to the Recycle Bin, except the ones marked ✅ keep."""
    old = shorts.listing(v.dir)
    keep = {s["n"] for s in old if s.get("mark") == "keep"}
    trash = []
    for s in old:
        if s["n"] not in keep:
            trash += [p for p in shorts.paths(v.dir, s["n"]) if p.exists()]
    if trash:
        for attempt in range(6):
            if projects.to_recycle_bin(trash):
                break
            if attempt == 5:
                raise jobs.JobError("Could not replace the old Shorts: one of them is open in another program (a "
                                    "video player?). Close it, then start this again — the new Shorts are ready and "
                                    "are not made twice.")
            time.sleep(1.0)
    new = shorts.listing(v.dir, "shorts_new")
    free = (n for n in range(1, 1000) if n not in keep)
    (v.dir / "shorts").mkdir(exist_ok=True)
    for s in new:
        n = next(free)
        src_mp4, src_js = shorts.paths(v.dir, s["n"], "shorts_new")
        dst_mp4, dst_js = shorts.paths(v.dir, n)
        data = shorts.load(v.dir, s["n"], "shorts_new") or {}
        data["n"] = n
        shorts.save(v.dir, n, data)
        src_mp4.replace(dst_mp4)
        _rm(src_js)
    _rm(v.dir / "shorts_new")
    if keep:
        v.log_note("keep", f"{len(keep)} Short(s) you marked ✅ keep stayed; the others went to the Recycle Bin")


def step_long(v, st, progress):
    platforms = v.project.get("platforms") or []
    if platforms and "youtube" not in platforms:
        if not st.get("long_skipped"):
            v.log_note("step", "no long video: \"YouTube (long video)\" is not ticked in the project")
            v.mark(st, "long_skipped")
        return
    if st.get("long") and v.f("long.mp4").is_file():
        return
    _check_source(v)
    _check_space(v)
    info = v.info()
    words = v.words()
    jump = bool(v.project.get("jump_cuts", True))
    silence = float(v.project.get("silence_s") or 1.2)
    total = float(info.get("dur") or 0)
    ranges = longvideo.plan(words, total, silence, jump_cuts=jump, fps=shorts.out_fps(info.get("fps")))
    kept = sum(b - a for a, b in ranges)
    progress(5, LABEL[5], f"{len(ranges)} part(s) to join", 0.0)

    def step(k, n):
        progress(5, LABEL[5], f"part {k} of {n}", k / max(1, n))
    longvideo.render(v.src, info, ranges, v.f("long.mp4"), v.boxes,
                     tick=lambda: progress(5, LABEL[5], "encoding", None), step=step)
    save_json({"ranges": longvideo.mapping(ranges), "dur": round(kept, 2), "src_dur": round(total, 2),
               "jump_cuts": jump, "silence_s": silence, "cut_s": round(total - kept, 2), "made": _now()},
              v.f("long.json"))
    v.log_note("render", f"long video made: {T.mmss(kept)} of {T.mmss(total)}"
                         + (f" ({T.mmss(total - kept)} of silence cut in {max(0, len(ranges) - 1)} jump cuts)"
                            if jump else " (jump cuts are off)"))
    v.mark(st, "long")


def step_texts(v, st, progress):
    if not (st.get("texts") and v.f("texts.json").is_file()):
        progress(6, LABEL[6], "writing titles and descriptions", 0.05)
        llm = providers.get_llm(v.settings)
        brain = v.brain(llm, lambda d="", f=None: progress(6, LABEL[6], d or "writing", None))
        long_info = load_json(v.f("long.json")) if v.f("long.mp4").is_file() else None
        data = texts.write(brain, v.lines(), shorts.listing(v.dir), long_info, v.name, v.project.get("platforms"))
        save_json(data, v.f("texts.json"))
        try:
            llm.unload()
        except Exception as ex:
            log.info("thinking unload: %s", ex)
        v.mark(st, "texts")
    if not st.get("thumbs"):
        progress(6, LABEL[6], "making thumbnails", 0.6)
        try:
            from . import thumbs
            # thumbs.auto writes thumb_<first_n>.. : a video that already has thumbnails (maybe changed in the
            # studio) gets new ones AFTER them, so nothing the creator made is lost (M3 contract)
            first = thumbs.next_n(v.pid, v.vid) if thumbs.thumb_list(v.pid, v.vid) else 1
            thumbs.auto(v.pid, v.vid, first_n=first)
            v.log_note("text", "thumbnails made" + (f" (new ones from number {first}; yours stay)" if first > 1 else ""))
        except ImportError as ex:
            log.info("thumbnails: %s", ex)
            v.log_note("warn", "thumbnails are not in this version of BroClips yet")
        except Exception as ex:                      # a thumbnail problem never fails the video
            log.warning("thumbnails failed: %s", ex, exc_info=True)
            v.log_note("warn", f"the thumbnails could not be made ({type(ex).__name__}: {str(ex)[:160]}) - you can "
                               "make them in the 🎨 Thumbnail studio")
        v.mark(st, "thumbs")


def _summary(v):
    n = len([s for s in shorts.listing(v.dir) if s.get("has_file")])
    parts = [f"{n} Short{'s' if n != 1 else ''}"]
    li = load_json(v.f("long.json")) if v.f("long.mp4").is_file() else None
    if li:
        parts.append(f"the long video ({T.mmss(li.get('dur'))} of {T.mmss(li.get('src_dur'))})")
    if v.f("texts.json").is_file():
        parts.append("titles and texts")
    return "✅ " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + " ready"


# ---------- the jobs ----------
def make(job, progress, cancelled):
    v = Video(job["project"], job["video"])
    again = bool((job.get("args") or {}).get("again"))
    st = v.stages()
    if again and st.get("again_of") != job["id"]:
        # make everything again: the transcript stays (same video, same words), the rest is made again; the old
        # Shorts are replaced only when the new ones are ready
        keep = {k: st[k] for k in ("prepare", "listen") if st.get(k)}
        st = dict(keep, again_of=job["id"], staging=bool(shorts.listing(v.dir)))
        for f in ("moments.json", "marathon_cache.json", "texts.json", "shorts_new"):
            _rm(v.f(f))
        save_json(st, v.f("stages.json"))
        _fresh_log(v, "🔁 Make again: new moments, new Shorts, a new long video and new texts")
    elif not st:
        _fresh_log(v, f"▶️ Started: {v.name}")
    info = step_prepare(v, st, progress)
    step_listen(v, st, progress)
    kept = [s for s in shorts.listing(v.dir) if s.get("mark") == "keep"] if st.get("staging") else []
    chosen = step_moments(v, st, progress, kept)
    step_shorts(v, st, progress, chosen)
    step_long(v, st, progress)
    step_texts(v, st, progress)
    msg = _summary(v)
    projects.set_video(v.pid, v.vid, {"status": "done", "made": _now(), "dur": info.get("dur") or v.meta.get("dur")})
    v.log_note("done", msg)
    return msg


def find_again(job, progress, cancelled):
    """The wishes -> new moments -> new Shorts (+ their texts and thumbnails). The long video stays."""
    v = Video(job["project"], job["video"])
    st = v.stages()
    if not (st.get("listen") and v.f("transcript.json").is_file()):
        raise jobs.JobError("Make this video first (✨ Make). Then you can look for new moments.")
    if st.get("again_of") != job["id"]:
        st = dict({k: st[k] for k in ("prepare", "listen", "long", "long_skipped") if st.get(k)},
                  again_of=job["id"], staging=True)
        for f in ("moments.json", "marathon_cache.json", "shorts_new"):
            _rm(v.f(f))
        save_json(st, v.f("stages.json"))
        wishes = (v.meta.get("wishes") or "").strip()
        _fresh_log(v, "🔁 Find new moments" + (f" — your wishes: {wishes.replace(chr(10), ' / ')}" if wishes else ""))
    step_prepare(v, st, progress)
    kept = [s for s in shorts.listing(v.dir) if s.get("mark") == "keep"]
    chosen = step_moments(v, st, progress, kept)
    step_shorts(v, st, progress, chosen)
    step_texts(v, st, progress)
    n = len(shorts.listing(v.dir))
    msg = f"✅ {n} Short{'s' if n != 1 else ''} ready (new moments)"
    v.log_note("done", msg)
    return msg


def short_edit(job, progress, cancelled):
    """Make one Short again after an edit (args {"n": 3}), or every Short (args {"n": "all"}, e.g. a new layout)."""
    v = Video(job["project"], job["video"])
    _check_source(v)
    info = v.info()
    want = (job.get("args") or {}).get("n")
    nums = [s["n"] for s in shorts.listing(v.dir)] if want == "all" else [int(want)]
    done = 0
    for k, n in enumerate(nums, 1):
        data = shorts.load(v.dir, n)
        if not isinstance(data, dict):
            continue
        rev = data.get("rev", 0)
        detail = f"Short {n}" + (f" ({k} of {len(nums)})" if len(nums) > 1 else "")
        progress(1, "Making the Short again", detail, (k - 1) / len(nums))
        mp4, _ = shorts.paths(v.dir, n)
        try:
            data = _render_one(v, info, data, mp4, f"{v.pid}-{v.vid}-short-{n:02d}.ass", progress, 1,
                               "Making the Short again", detail, (k - 1) / len(nums))
        except PermissionError:
            raise jobs.JobError(f"Short {n} is open in another program (a video player?). Close it and try again.")
        with shorts.LOCK:
            cur = shorts.load(v.dir, n) or data
            for key in ("phrases", "dur", "layout", "made"):
                cur[key] = data[key]
            cur["version"] = int(cur.get("version") or 0) + 1
            if cur.get("rev", 0) == rev:
                cur["dirty"] = False                 # (a newer edit keeps it dirty: its own job follows)
            shorts.save(v.dir, n, cur)
        v.log_note("render", f"Short {n} made again ({data['dur']:.0f} s)", title=data.get("title"))
        done += 1
    if not done:
        raise jobs.JobError("That Short is not there any more.")
    return f"✅ Short {nums[0]} made again" if len(nums) == 1 else f"✅ {done} Shorts made again"


jobs.register("make", make, steps=STEPS, title="Make")
jobs.register("find_again", find_again, steps=STEPS, title="Find new moments")
jobs.register("short_edit", short_edit, steps=1, title="Re-make a Short")
