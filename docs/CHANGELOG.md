# Changelog

## 2026-10-07 — public clean-up before publishing the redesign
- The raw per-milestone build notes (`docs/notes-*.md`) and machine-specific details (local paths, the installed
  copy's settings, the developer's PC) were moved out of the public repo; the developer keeps them locally
  (`CLAUDE.local.md`, git-ignored). The technical story stays in this file and in `docs/SPEC.md`; SPEC §12 is now a
  short "Development PC" note.
- The GitHub history was replaced by one clean commit (the earlier 15 commits carried those notes), so commit hashes
  quoted anywhere before this date no longer exist.

## 2026-10-07 — UI redesign: one design system, new screenshots, a hero banner
A visual redesign of every page, built by an agent and reviewed screenshot by screenshot. **No behaviour, API call,
route or POST rule changed**: every element id and data attribute the scripts read is still there (the new tiles drive
the same hidden `<select>`s), every POST still goes through `api()` as `application/json`, and everything stays
offline (system fonts, no CDN, no JS library). Suite **184 passed** (the static-files test now also fetches
`/icons.js` and `/studio.html`); every page script parses with `node --check`.
- **Design system** (`broclips/web/app.css`): tokens in `:root` — background `#0b0d12`, panels, hairlines, three text
  tiers (≥ 4.5:1 contrast), the brand gradient pink `#ff4d8d` → violet `#8b5cf6` → cyan `#22d3ee`, semantic
  ok / warn / bad / info, radii, layered shadows, Segoe UI Variable + Cascadia Code, 170 ms motion — and components:
  top bar + status pill, segmented nav, buttons, fields, switches, stepper, cards, glow cards, notes, badges, chips,
  tiles, video cards, work card, terminal log, stat tiles, phone frames, platform tabs, save bar, help contents + FAQ
  accordions, toasts, confetti. Breakpoints 1180 / 1060 / 860 / 760 px; `prefers-reduced-motion` turns animations off.
- **Icons**: Lucide line icons as inline SVG in `broclips/web/icons.js` (ISC licence, credited in the README). Helpers
  in `app.js`: `ic(name)` / `<i data-i="name">`, `toast`, `confirmClick`, `confetti`, `imgIn` / `imgErr`, arrow keys
  inside radio groups, `markNav`, the status pill (`workText()` from `/api/state`).
- **Pages**: a sticky glass top bar with a gradient wordmark and a live status pill ("Making “…” · Listening 41 %",
  amber while it waits for a program, red when the app is not running). Projects: a first-visit card with an inline-SVG
  illustration and a 3-step strip; project cards with gradient avatars and a live dot; video cards with a preview frame
  (`/api/thumbs/tile`), status pill and progress bar. Project page: the prompt card with example chips, video-type
  tiles, platform chips, switch cards, a Shorts stepper. Video page while working: a big gradient %, a caption per
  step, a shimmering bar, the 6-step vertical stepper and a terminal-style live AI log; numbers update in place so the
  animations never restart. When done: stat tiles, a one-time confetti burst (remembered per video), Shorts in phone
  frames with Keep / Skip toggles, the long video with clickable chapters, the thumbnail gallery with "Open Thumbnail
  studio", platform tabs with Copy, layout tiles with small diagrams. Settings: provider tiles with two-letter marks
  (deliberately not company logos), coloured Test badges, a privacy switch card, a sticky save bar. Help: contents that
  follow the scroll, step cards, FAQ accordions. Studio: the same design system and a glowing stage; its logic is
  unchanged apart from HTML strings.
- **App icon** (`broclips/web/broclips.png` / `.ico`, `scripts/make_icon.py`) redrawn in the brand gradient with the
  same play triangle. An existing Desktop shortcut can keep showing the old icon until it is recreated.
- **Screenshots** (`docs/screenshots/`, 2.8 MB): new `hero.png` (1600×700 README banner: an HTML page rendered with
  headless Chrome — logo, tagline, a tilted results page and two real Short frames in phone frames; its HTML source is
  not in the repo), `start.png`, `working.png`; replaced `project.png`, `results.jpg`, `settings.png`, `studio.jpg`,
  `studio-arabic.jpg`; `shorts.jpg` and `thumbnails.jpg` (output pictures, not UI) kept. Made from real runs with local
  AI in a scratch data folder: the sample lesson (2 Shorts; a Marathon-mode project on it: 3 Shorts) and a 13:39
  League of Legends recording with Egyptian-Arabic commentary (6 Shorts, an 8:01 long video, 8 chapters, 3 thumbnails
  in 7 min 40 s). Headless Chrome's `--screenshot` never finishes on a page with many videos, so the pictures were
  taken over the DevTools protocol. Privacy: neutral paths, no keys, the League Shorts paused at second 3 where no
  player name is readable.
- Not done: a light theme (the app is dark by design); the README says "224 MB" for the light cut-out model where the
  app says "214 MB" — the same file in MB vs MiB.

## 2026-10-07 — M4: README with real screenshots, published, installed
- **Screenshots** (`docs/screenshots/`, 1.2 MB; most replaced by the UI redesign above): taken with headless Chrome at
  1400 px on a demo data folder made from M2's real live results (lesson, Marathon run, a League game). Privacy pass:
  the demo lesson moved to a neutral path so no Windows user name shows; Settings cropped above the data-folder box;
  one League thumbnail left out (an augment-menu background with the chat box, which can show other players' names —
  also the weakest design). The League results PAGE never finished in headless Chrome (six shorts + a long video keep
  the network busy past `--virtual-time-budget`; `timeout` + `--timeout` stop it) → the League results were shown
  through output pictures instead.
- **README**: what it does, real Shorts / thumbnails, who it is for, a mermaid "how it works", the four AI options
  (cost, privacy, what is sent), install (`install.bat` steps + advanced options), how to use it, FAQ, **known
  limitations**, **"Built with AI — honestly"**, credits + licences. Secret scan before publishing: only the fake test
  keys in `tests/`; no e-mail or real key in any tracked file.
- **Published**: https://github.com/omar1001/BroClips (public, MIT), topics youtube-shorts, tiktok, video-editing,
  whisper, ollama, thumbnails, ai, python, windows, arabic. Checked on github.com: all 12 images load, mermaid renders.
- **Installed on the developer's PC** by reusing an existing Python environment and asset folder
  (`scripts\install.ps1 -Python <python.exe> -NoPip -AssetsFrom <folder> -LocalSTT no -Cutout no`): no big downloads
  (62 files copied, 7 small Latin font files fetched), the data folder on another disk (`data_location.txt`,
  git-ignored), Desktop shortcut. Settings: thinking = a private Ollama server with `gemma4:12b-it-qat`, listening =
  local large-v3, the cut-out model from the other install, pause while the game runs, blur boxes over the recording
  overlays. 🧪 Tests on the installed copy: "Ready: Ollama 0.34.2, model gemma4:12b-it-qat (can see pictures)" and
  "Ready: GPU, large-v3".

## 2026-10-07 — M2: the video pipeline + results page
Built by an agent in parallel with M3 (stopped once by the usage limit, resumed). Suite **184 passed** (~52 s; 41 new:
caption phrasing EN/AR, hallucination filter, moments mapping/merging, tournament, jump cuts, filtergraphs + real tiny
renders, the whole `make` job with a scripted fake AI, pause/resume, edits, Find new moments keeping a ✅ short).
- `transcript.py` (lines, sentence-aware caption phrases, min word length, Whisper-hallucination filter), `moments.py`
  (windowed search with the project prompt, wishes with/without a time, grow over continuous talk only, join only the
  AI's own touching picks in lessons; rule-based fallback when the AI gives nothing), `marathon.py` (list wide → pool →
  tournament both orders; no cut contest — in LoL Clips shortened cuts lost the context that made a moment funny),
  `camera.py`, `shorts.py` (follow / zoom / fit, word-highlight captions: static Montserrat ExtraBold / Lalezar by
  script, title 4.5 s or the whole short in fit, blur boxes first), `longvideo.py` (jump cuts + chapters), `texts.py`
  (per platform, cleaned of stray Cyrillic), `pipeline.py` (`make`, `short_edit`, `find_again`; resumable stages;
  thumbnails via `thumbs.auto(first_n=next_n)` so studio work is never overwritten), `results_api.py` + the video page
  (shorts with ✅/❌/✏️/🗑️, long video + chapters, texts with copy, Thumbnails card, wishes + 🔁 Find new moments, the
  live AI log).
- Live (local gemma4 12B on a private Ollama server + Whisper large-v3 GPU + NVENC): sample lesson 77 s (2 shorts, long
  1:17 of 1:28, 4 chapters, texts, 3 thumbnails); Marathon on the lesson 106 s, 16 questions, the judge agreed with
  itself in both orders 6/6; a 13:39 League of Legends recording with Egyptian-Arabic commentary: 7.3 min → 6 shorts
  with Egyptian titles incl. «أبو لازقة» at 10:26 (also LoL Clips' own top moment), long video 8:09 (88 kept parts),
  marks blurred, Arabic captions joined RTL; source untouched.
- Fixed live: two lesson tips merged into one 52-s short (growth over a topic pause + scene join) in both searches;
  Cyrillic "з" inside Arabic chapter names; captions/titles enlarged for phones.
- Gaps: cloud providers not run through the pipeline live; no listening-prompt setting (hallucination risk, M1); no
  speed-up control on the page (the renderer supports it); very short videos get no YouTube chapters (YouTube needs 3+).

## 2026-10-07 — M3: thumbnails + Thumbnail studio
Generic port of LoL Clips' studio (v16/v17) by an agent, in parallel with M2; both were stopped once by the Claude
usage limit (HTTP 429) and resumed from their own context after the reset, with checkpoint commits. Suite 183 passed
(20 new).
- `thumbs.py`: one renderer for automatic thumbnails + the studio; styles subject / zoom / reaction / label /
  **headline** (new: huge words over a darkened blurred picture, for tutorials) / **split** (two subjects + VS); the
  zoom steers around `settings["blur_boxes"]` (also blurred in every frame); any frame shape; fonts by the words' script
  (Latin never used for Arabic), raqm + fribidi; **subjects** instead of champion art: BiRefNet cut-out of the current
  frame (`cutout` job) or an uploaded image (base64 in JSON; PNG with alpha as is), cached, cut sides faded; 🎲 random
  mix, ✨ AI ideas (`thumbai` job; `_wants` rules; count from the field or the words; frames only to a local AI or when
  allowed), "About this video", gallery multi-delete; saving refused while that video's Make runs or waits.
- `thumbs_api.py` + `web/studio.html`: the full studio; `GET /api/thumbs/list` + `/studio.html?project=&video=&n=`
  (M2's video page uses them); `thumbs.auto(pid, vid, first_n=1, wait_cut=True)` OVERWRITES thumb_1..3 — M2 passes
  `first_n=thumbs.next_n(...)` when thumbnails exist.
- Live: sample lesson → 3 automatic (zoom / reaction / headline) in 6.9 s, words matching each picture's moment;
  an uploaded champion splash cut out in 26 s; League frames → Arabic in 4 fonts, joined RTL, watermark blurred, source
  file unchanged; ✨ AI with local gemma4 12B: "3, one zoom one reaction" obeyed, 27 s; studio checked with the mouse.
- Gaps: no fonts for CJK/Hindi; tall phone videos show a 16:9 band; zoom can cut a slide title; camera-video subject
  auto-cut-out not run live; cloud AIs with frames not tested live; caches never cleaned; any module drawing text with
  Pillow must import `broclips.thumbs` first.

## 2026-10-07 — M1: AI providers + app shell
Built by two agents in parallel from `docs/SPEC.md`; reviewed by the coordinator (full suite 123 passed, 15.7 s;
headless-Chrome screenshots of Settings / Help / projects at 1400×900 looked right).
- **Providers** (`broclips/providers/`): Ollama (incl. a private server on its own port), Gemini (Interactions API as
  verified in LoL Clips, repair-on-400 ladder), OpenAI-compatible (json_schema → json_object → plain), Anthropic
  (structured outputs first — the `claude-api` skill says a forced tool_choice gives HTTP 400 on the newest models;
  default `claude-haiku-4-5`), local faster-whisper (model shared at module level so the work line can free it),
  OpenAI/Groq and Gemini speech-to-text; `_http.py` maps errors to plain English and masks keys; pictures go to a
  cloud AI only with the privacy switch on. Live on the developer's PC: gemma4 12B answer 1.3 s (vision ok, unload
  frees VRAM); Whisper large-v3 on a 20 s Arabic clip 1.8 s (warm). Cloud providers tested with mocks only.
- **Media** (`media.py`): probe, 16 kHz mix, frames, loudness spikes, NVENC detection, `blur_filter` (delogo needs
  x,y ≥ 1 and ≤ W-1/H-1 — measured), `split_audio` (cuts in quiet moments for cloud listening).
- **App shell**: work line (one job, resume, dedupe, pause while listed programs run, unload after jobs), projects by
  reference (videos never copied; deletions only of BroClips' own folders, to the Recycle Bin), server on :8770 with a
  route registry + `/files` Range + single instance + **POST guard** (Host, Origin, JSON, Sec-Fetch-Site — the same
  lesson as LoL Clips), pages (projects, settings with 🧪 tests, help), `assets.py` (`--from <folder>` reuses another
  install's assets, static Montserrat ExtraBold for captions, zstandard for fribidi without Node), installer
  (`install.bat` → `scripts/install.ps1`, `-Python` to reuse an environment, Desktop shortcut), icon.
- Coordinator: `tests/make_sample.py` (89-s English "Python for loops" lesson: ffmpeg slides + Windows TTS, a moving
  highlight and long pauses — test + README material with no private footage), README draft, `.gitattributes`,
  `data_location.txt` ignored, `server.OPTIONAL_MODULES` += `results_api`, `thumbs_api`.

**Flags from M1:** cloud providers untested against real services (the first real 🧪 Test may need a small fix — the
log names the refused field); a vocabulary prompt made Whisper invent "اشتركوا في القناة" on a test clip (M2 adds a
filter; Help should warn); Ollama `unload()` also unloads another app's model on a shared server; file pickers and the
Edge window were not opened live (they would pop up on the developer's screen); a Recycle-Bin self-test left a folder
named "BroClips recycle-bin test (safe to delete)" in the Recycle Bin.

## 2026-10-07 — project started
Omar asked for a standalone app with the same ideas as his private League of Legends pipeline ("LoL Clips"), for ANY
video he records (other games, teaching school subjects, programming topics) for YouTube / TikTok / Shorts, usable on
its own (README + in-app Help, no AI assistant needed), with local AI or API keys, Marathon mode included, published
publicly on his GitHub with screenshots and an honest "built with AI" note.

**His answers:** "I record, it edits" (no script writing, no AI-made videos) · a separate new app (LoL Clips stays) ·
AI: local + Gemini + OpenAI-compatible + Anthropic · tier `normal` · name with his "Bro" prefix → **BroClips** · MIT.

**Done:** repo scaffold, `docs/SPEC.md` (the contract), `CLAUDE.md`, MIT licence, `.gitignore`.

**Plan:** M1 core (settings, providers, work line, server, pages, installer) → M2 pipeline (listen, moments, Marathon
mode, shorts, long video, texts) → M3 thumbnails (port of the LoL studio, generic "subject") → M4 README, screenshots,
GitHub. Each milestone is built from the spec and checked on the developer's PC before the next one starts.
