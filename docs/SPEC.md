# BroClips — specification (v1)

Owner: Omar (github `omar1001`). Written 2026-10-07 from his request and his answers. Tier: **normal** ("don't overkill
it": simple and robust beats complete). This file is the contract for everyone building BroClips (people or AI
agents). If something here is unclear, prefer the simplest choice that keeps the user's videos safe.

## 1. What BroClips is

A free, local-first desktop app: **you record a video, BroClips edits it for you.**

Input: any recording where someone talks — gameplay with commentary (any game), a lesson for school students, a
programming tutorial (screen recording), a talking-head video, a podcast. Any language (Egyptian Arabic and English
must work well; Arabic is right-to-left).

The user describes the videos once per **project** with a prompt (e.g. "Short Python lessons for beginners, in
English, for YouTube Shorts and TikTok, friendly tone" or "My funny League of Legends commentary in Egyptian Arabic").

Output per video:
1. **Shorts** (vertical 1080×1920, ≤ 3 min — YouTube Shorts / TikTok / Reels): the best self-contained moments, cut
   with their context, big word-by-word captions, a title on top.
2. **A clean long video** (16:9) for YouTube: the whole recording with long silences cut out ("jump cuts").
3. **Titles, descriptions, hashtags** per platform (in the project's language), YouTube chapters for the long video.
4. **Thumbnails** (1280×720) — automatic ones + a **Thumbnail studio** to change them with clicks, get more at random,
   or ask the AI for N new ones from a prompt.

**Marathon mode** (Omar's idea): instead of one AI answer, the AI works in many small rounds — lists every candidate
moment, then compares them two at a time (each pair asked in both orders) in a tournament — to pick better moments.

AI can be **local and free** (Ollama for thinking, faster-whisper for listening) or a **cloud API key** the user
pastes: Google Gemini, OpenAI or any OpenAI-compatible service (OpenRouter, Groq, LM Studio, …), Anthropic Claude.

Not in v1 (non-goals): making videos without a recording (AI voices, slides); writing scripts before recording;
uploading to YouTube/TikTok (the user uploads); multi-part "story" shorts; an AI editor that edits by chat (v1 has
simple manual edits); macOS/Linux installers (the code should run there, untested).

## 2. Hard rules (never break)

1. **Never modify, move or delete the user's source videos.** Open them read-only. All output goes to the data folder.
2. **One heavy job at a time** (speech-to-text, AI, rendering) — a home PC has limited RAM/VRAM. After an AI job,
   unload local models (Ollama `keep_alive: 0`).
3. **API keys** live only in `<data>/secrets.json`. Never logged, never shown again in full (UI shows `…abcd`), never
   committed, never sent anywhere except the provider they belong to.
4. **No console windows** on Windows: the app runs with `pythonw`; every subprocess uses `CREATE_NO_WINDOW`.
5. **Works without any AI key** (local mode) and **without a GPU** (CPU fallbacks: `libx264`, a small Whisper model).
   Clear messages when something is missing ("Ollama is not running — install it from ollama.com or add an API key
   in Settings").
6. **Honest README**: BroClips was built by Omar with heavy AI assistance (Claude Code). Say so.
7. **English UI**, plain words, no jargon. Every button has a tooltip. The in-app Help page explains everything a new
   user needs.
8. Never block the UI: long work runs as a job with progress; the page polls.

## 3. Tech and layout

- Python ≥ 3.10 (developed on 3.12, Windows 11). ffmpeg on PATH (the full build: libass, fribidi, harfbuzz).
- Local web app: stdlib `ThreadingHTTPServer` on `127.0.0.1:8770` + plain HTML/CSS/JS pages (no build step, no npm),
  opened in an Edge/Chrome `--app` window (fallback: default browser). Dark theme. One instance at a time; a newer
  version asks an idle older one to quit (`/api/version`, `/api/quit`, same as LoL Clips `panel.py`).
- Dependencies: `numpy`, `Pillow`, `psutil` (core); `onnxruntime` (thumbnail cut-outs); `faster-whisper` (optional,
  local speech-to-text). HTTP to AI providers with the standard library (`urllib`) — no vendor SDKs.

```
BroClips/
  README.md  LICENSE (MIT)  CLAUDE.md  requirements.txt  requirements-local.txt  install.bat  scripts/install.ps1
  broclips/
    __main__.py      python -m broclips  -> start server (+ open the window unless --no-window)
    config.py        paths, DEFAULTS, settings/secrets load+save, data folder
    util.py          logging, run() (no window), json io (atomic, retrying), part paths, finish() (rename retry)
    media.py         ffprobe, audio extraction (16 kHz mono wav), frames (jpg/rgb), loudness, encoder detection
    providers/       AI backends (§4): base.py, ollama.py, gemini.py, openai_compat.py, anthropic.py,
                     stt_local.py (faster-whisper), stt_openai.py, stt_gemini.py, fake.py (tests), __init__.py
    transcript.py    words -> lines/sentences; language; text helpers
    moments.py       normal search (§6.3)
    marathon.py      Marathon mode (§6.4)
    camera.py        follow-the-action crop path (port of LoL camera.py, no game masks)
    shorts.py        vertical render + captions (§7.1)
    longvideo.py     jump-cut long video + chapters (§7.2)
    texts.py         titles / descriptions / hashtags (§6.6)
    thumbs.py        thumbnail renderer + designs + auto/random/AI (§8)
    jobs.py          the work line: one job at a time, progress, cancel, resume, pause-while-app-runs
    pipeline.py      the per-video steps (§6), resumable
    server.py        HTTP API + static files
    assets.py        downloads fonts / emoji / fribidi (Windows) / cut-out model, rate-limited, hash-checked
    web/             index.html (projects + videos + results), studio.html, settings.html, help.html, app.css,
                     icon (broclips.ico + png)
  tests/             pytest: providers (mocked HTTP), captions, designs, settings, jump cuts, pipeline with fake AI
  docs/              SPEC.md, CHANGELOG.md, screenshots/
```

Data folder (default `%USERPROFILE%\BroClips`, changeable in Settings):
```
settings.json   secrets.json   logs/   cache/ (frames, cut-outs)   assets/ (fonts, emoji, bin/fribidi-0.dll)
projects/<pid>/project.json
projects/<pid>/videos/<vid>/video.json        {"src": "D:\\rec\\lesson1.mp4", "name", "dur", "added", "status"}
                           /audio16k.wav  transcript.json  lines.json  moments.json  marathon_cache.json
                           /shorts/short_01.mp4 (+ .json: segments, title, captions edits)
                           /long.mp4  long.json (kept ranges -> chapters)  texts.json
                           /thumbs/thumb_N.jpg + thumb_N.json (designs)  /progress.json  /stages.json
```
Models: the cut-out ONNX model goes to `<data>/models/` unless Settings point elsewhere.

## 4. AI providers

Two roles, each chosen in Settings: **Thinking** (LLM) and **Listening** (speech-to-text). One provider per role,
plus optional per-role model names. "Test" button per role: a tiny real call with a clear OK / error message.

### 4.1 LLM interface (`providers/base.py`)
```python
class LLM:
    name: str                      # "ollama" | "gemini" | "openai" | "anthropic" | "fake"
    vision: bool                   # can it look at images?
    def chat_json(self, system: str, user: str, schema: dict, images: list[bytes] = None,
                  temperature: float = 0.2, max_tokens: int = 1500) -> dict | None: ...
    def unload(self): ...          # free local memory (no-op for cloud)
    def test(self) -> str: ...      # raises ProviderError with a plain-English message
class ProviderError(Exception): ...
```
- Every provider must return a parsed dict that matches `schema` or `None` after retries (2 retries, backoff). Keep
  schemas simple (objects, arrays with `maxItems`, enums, strings, integers, booleans) so every provider can use them.
- **Ollama** (local): `POST {url}/api/chat`, `format: <schema>`, `stream: false`, `think: false`, `keep_alive: "10m"`,
  options `num_ctx` (16384), `temperature`, `num_predict`. Images: base64 in `images`. Settings: url (default
  `http://127.0.0.1:11434`), model (default `gemma3:12b`; list installed models via `/api/tags` in Settings), optional
  "start a private server" (exe path + models folder + port) like LoL `brain.ensure_server`. `unload()` = every loaded
  model `keep_alive: 0` (`/api/ps`).
- **Gemini**: REST, key in header `x-goog-api-key`. Use the facts verified in the author's private LoL Clips
  app (its `cloud.py` and ground-truth notes: model names, endpoints, limits). Structured output: response schema (OpenAPI subset). Default model: the flash model named there.
- **OpenAI-compatible**: `POST {base_url}/chat/completions` (default base `https://api.openai.com/v1`), `Authorization:
  Bearer`. Structured output: `response_format: {"type":"json_schema","json_schema":{"name":"answer","schema":…,
  "strict":false}}`; if the server rejects it (400), retry with `{"type":"json_object"}` + the schema in the prompt;
  last resort: plain text + extract the first JSON object. Images: `image_url` data URLs. Settings: base URL presets
  (OpenAI, OpenRouter, Groq, LM Studio `http://127.0.0.1:1234/v1`, custom), model name.
- **Anthropic**: Messages API. **Load the `claude-api` skill before writing this provider** (current model ids,
  headers, structured output). Structured JSON via a forced tool (`tools=[{name:"answer", input_schema: schema}]`,
  `tool_choice: {type:"tool", name:"answer"}`) unless the skill documents something better. Images: base64 blocks.
- **fake**: deterministic answers for tests (returns schema-shaped minimal data).

### 4.2 Speech-to-text interface
```python
class STT:
    def transcribe(self, wav_path: str, language: str | None, prompt: str = "", on_progress=None) -> dict:
        # {"language": "ar", "words": [{"w": "word", "s": 1.23, "e": 1.56, "p": 0.98}, ...]}
```
- **local** (`faster-whisper`): model choice (`large-v3` default with CUDA, `small` without), `word_timestamps=True`,
  `vad_filter=True`, the user's optional prompt (vocabulary/style) as `initial_prompt`, hallucination filter (port
  `asr._looks_hallucinated`), floats cast to `float` (numpy floats break JSON). Windows CUDA DLL setup like LoL
  `util.cuda_dll_setup`. Free the model after the job.
- **openai** (OpenAI / Groq / any compatible): `POST {base}/audio/transcriptions`, multipart, `response_format=
  verbose_json`, `timestamp_granularities[]=word`; files over the size limit are split into ≤ 10-minute Opus/MP3
  pieces with time offsets. Default model `whisper-1` (OpenAI) / `whisper-large-v3` (Groq).
- **gemini**: as in LoL `cloud.transcribe` (verified there); when only segment times come back, spread word times
  inside each segment by character length.
- Cloud listening sends the audio to that provider — the UI says so before the first use.

### 4.3 Privacy setting
"Allow sending pictures (video frames) to cloud AI" — off by default. Without it, cloud LLMs only get text.

## 5. Settings (`settings.json`, defaults in `config.DEFAULTS`)
data folder, output folder (default inside data), thinking provider + model + url/base, listening provider + model,
default language (auto), GPU encoder (auto-detect NVENC → else libx264), "pause while these programs run" (list of
exe names, default empty; e.g. a game's `.exe`), always-blur boxes (list of `[x, y, w, h]` in 1920×1080
coordinates, applied to every output — e.g. a watermark that is in every recording), cut-out model path.
`secrets.json`: `{"gemini": "...", "openai": "...", "anthropic": "...", "openrouter": ...}` (masked in the UI).

## 6. Pipeline (per video, `pipeline.py`) — resumable, each stage marked done in `stages.json`

Progress (`progress.json`): step 1..6, label, detail, fraction — the page shows a bar and the step list.

1. **Prepare**: ffprobe (duration, audio streams, fps), extract `audio16k.wav` (mix all audio streams).
2. **Listen**: STT → `transcript.json`; `lines.json` = sentences/lines (port LoL `asr.make_lines`: pause 0.7 s,
   ≤ 12 s). Loudness spikes (port `audio.loudness_spikes`) as a hint for exciting moments.
3. **Find moments** (§6.3 or Marathon §6.4) → `moments.json`: list of
   `{"s", "e", "title", "hook", "why", "type", "score", "segments": [[a, b, speed]]}` (times in the source).
4. **Shorts** (§7.1) → `shorts/short_NN.mp4`.
5. **Long video** (§7.2) → `long.mp4` (+ chapters).
6. **Texts + thumbnails** (§6.6, §8) → `texts.json`, `thumbs/`.

### 6.3 Normal moment search
Windows of ~5 min over the lines (stride ~3.5 min), each asked: "the best self-contained moments for <platforms>"
with the project prompt, the language and these rules: a moment must make sense alone (setup + payoff), 15–90 s
(teaching: 30–120 s), start on a clean sentence start, end after the payoff; return line ids (not times) + title +
hook (≤ 8 words, in the project language) + why + a 1–10 score. Code then: maps ids → times, grows over continuous
speech, merges overlaps (≤ 6 s apart = one moment), drops < 10 s, ranks, keeps the top N (setting, default 8).
User "wishes" box per video (like LoL hints): lines with a time (`5:30 the joke about…`) are searched first; lines
without a time are added to every question as instructions.

### 6.4 Marathon mode (switch per project)
Port the generic parts of LoL Clips' `marathon.py`: (1) **find wide** — in 2-minute windows
the AI only LISTS candidate moments (no judging); (2) **pool** — merge the same moment, grow each to a complete
candidate, ≤ 40; (3) **tournament** — pairs compared in BOTH orders ("which makes the better short for this
channel?"), several random rounds then the best 16 all meet; ranking = wins, never a 1–10 score; (4) winners (≥ half
of their final questions, min 3) become the moments. Every answer cached in `marathon_cache.json` so a paused job
continues. Show live notes ("🥊 A vs B → A") in a "What the AI is doing" panel. Warn in the UI that with a cloud key
this costs many requests (show an estimate).

### 6.6 Texts
One question with the moments + a transcript sample + the project prompt → per platform: 3 video titles, 1
description (YouTube long: + chapters from `long.json`), hashtags; per short: title + caption + hashtags. Saved to
`texts.json`; the UI has copy buttons.

## 7. Rendering

### 7.1 Shorts (port and generalise LoL Clips' `render.py: short`, `_caption_*`, `camera.py`)
- 1080×1920, 60 fps if the source has it (else source fps), NVENC `-cq 21` or libx264 `-crf 20`, AAC 192k, loudnorm
  −14 LUFS, 0.12 s fade in / 0.35 s fade out, `+faststart`. Every subprocess without a window.
- Layouts (project setting, per video override):
  - **follow** — a 9:16 window of the frame follows the action (camera.py: where the picture changes; smooth path).
    Default for gameplay and talking heads.
  - **zoom** — the middle 810×1080 of a following 1080-wide window shown ×1.33 on a blurred copy of the frame (LoL
    "layout C"). Good for gameplay.
  - **fit** — the whole frame, full width, on a blurred, darkened copy; title above, captions below. Default for
    screen recordings / tutorials (nothing is cut off).
- Captions: phrases of 2–6 words, broken at sentence ends, never across a cut; the word being spoken highlighted
  (yellow); big bold font with a thick outline; Arabic RTL correct (libass); kept out of the phone apps' covered
  areas (bottom ~25 %, top ~10 %). Fonts: Latin = Montserrat ExtraBold, Arabic = Lalezar (both OFL, downloaded by
  `assets.py`). Title (hook) on top for the first seconds.
- Always-blur boxes (§5) applied first (ffmpeg `delogo`/`boxblur` crop-overlay), so no crop can show them.
- Simple edits in the UI: start −/+ (1 s, 0.2 s), end −/+, change the title, fix caption words (edit the text of a
  phrase), delete the short, keep/skip mark. Re-render only that short.

### 7.2 Long video (port the parts idea of `render.full_video`)
- Keep every stretch with speech; cut silences longer than a setting (default 1.2 s, keep 0.25 s padding each side;
  "jump cuts: off" keeps everything). Join with 0.08 s audio fades so cuts never click. Encode parts + concat copy.
- `long.json`: the kept ranges (source ↔ output times) → YouTube chapters (texts step names them).

## 8. Thumbnails (port LoL Clips' `thumbs.py` + `studio.html`)

Keep from LoL: ONE renderer for automatic thumbnails AND the studio, designs saved as `thumb_N.json` next to the JPG;
layers (picture → light rays → subject → circle/arrow → text → 3D emoji); `_aim` / `HUD_ZONES` idea becomes
"avoid areas" (§5 blur boxes + nothing else by default); action point by frame difference; Pillow + raqm (+
`fribidi-0.dll` on Windows, from `assets.py`) for Arabic; fonts (Latin: Montserrat, Anton, Bebas Neue; Arabic:
Lalezar, Alexandria, Cairo, Changa, Marhey, Jomhuria); 3D emoji (Microsoft Fluent, MIT); BiRefNet cut-out (CPU,
cached); 🎲 random mix; ✨ AI ideas from a prompt + count (count also read from the prompt); "About this video" box;
🗂️ gallery with multi-delete; Save / Save as new / Undo; YouTube computer + phone previews.

Drop: League champion detection, Riot art, HUD portrait matching, privacy gate. Replace champion art with a
**subject**: (a) cut out the person/thing from the current video frame ("✂️ Cut out from this picture"), (b) an
uploaded image (your photo, logo; PNG with transparency used as is, others cut out), (c) none.
Styles: **subject** (subject big on one side + words), **zoom** (circle + arrow on the action), **reaction** (giant
emoji face), **label** (words in a box + arrow), **headline** (huge words over a darkened blurred picture — good for
tutorials), **split** (two subjects + "VS"). The AI's words are in the project language. Clear requests in the
prompt are enforced in code (LoL lesson: a 12B model ignores a requested style).

## 9. UI (all English, dark, big clear buttons, tooltips)

- **Header**: logo + "BroClips", work line status ("⏳ Lesson 3 — Listening… 40 %"), ⚙ Settings, ❓ Help.
- **Left**: projects (+ New project), each with its video count.
- **Project page**: name, the prompt ("What are these videos? Who watches them? Which platforms? Which language?
  Tone?"), video type (gameplay / talking to camera / screen recording / podcast / other → default layout), platforms
  (YouTube long, YouTube Shorts, TikTok, Instagram Reels), language (auto, Arabic, English, …), switches (Marathon
  mode, jump cuts, max shorts), "➕ Add videos" (Windows file picker via tkinter in a subprocess, like LoL
  `pick_folder`; or paste paths), the video list (status badge, ✨ Make, 🔁 Make again, remove from project — never
  deletes the file).
- **Video page**: progress + "What the AI is doing" log while working; results: shorts grid (player, title, why,
  ✅ keep / ❌ skip / ✏️ edit / 🗑️), long video player + chapters, thumbnails card (🎨 Thumbnail studio, ✏️ Change,
  ➕ New), texts with 📋 copy, 📂 Open folder, a wishes box + 🔁 Find new moments.
- **Settings**: Thinking AI (provider dropdown, model, URL/base, key field, 🧪 Test), Listening AI (same), privacy
  switch, folders, encoder, pause-while list, blur boxes, cut-out model (⬇️ Download, 928 MB, or "light" 214 MB).
- **Help**: what BroClips does, first steps (install Ollama + a model, or paste a key), each page explained, FAQ
  (no GPU? which model? where are my files? does anything leave my PC?).

## 10. Installer (Windows) and launch
`install.bat` → `scripts/install.ps1`: find Python ≥ 3.10 (else explain + link); check ffmpeg (else offer `winget
install --id Gyan.FFmpeg -e`); create `.venv`; `pip install -r requirements.txt`; ask "Local speech-to-text
(faster-whisper + NVIDIA libraries, ~1.5 GB)? [Y/n]"; `python -m broclips.assets --essential` (fonts, emoji,
fribidi); ask for the cut-out model; create the Desktop shortcut **BroClips** (`.venv\Scripts\pythonw.exe -m
broclips`, working dir = repo, icon `broclips/web/broclips.ico`). Also `run.bat` for a console run with logs.
Advanced: `install.ps1 -Python <path>` reuses an existing environment (one that already has every package — no
downloads).

## 11. Quality bar and tests
- Unit tests (pytest, no network, no GPU): providers with mocked HTTP (request shape + response parsing + fallback),
  caption phrasing (Arabic + English), jump-cut planning, design normalising, settings/secret masking, count parsing.
- A sample video generator for tests and the README (`tests/make_sample.py`): a 2–3 minute "lesson" (slides drawn by
  ffmpeg + Windows text-to-speech narration via `System.Speech`) — no downloads, no privacy issues.
- End-to-end acceptance on the developer's PC with local AI: (a) the sample lesson (English, fit layout), (b) a
  real League of Legends recording with Egyptian-Arabic commentary (read-only!).
- README: what it is, who it is for, screenshots (real app, real results), install, first run, providers (local vs
  keys, what is sent where), FAQ, licence, "Built with AI" note, credits (fonts OFL, Fluent Emoji MIT, BiRefNet MIT,
  fribidi LGPL, ffmpeg).

## 12. Development PC
Built and tested on Windows 11 with an RTX 5060 Ti 16 GB, 16 GB RAM, Python 3.12 and an ffmpeg full build
(NVENC). Machine-specific notes for the developer live in `CLAUDE.local.md` (git-ignored, not in the repo).
