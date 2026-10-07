# BroClips — your recordings → Shorts, TikToks, a clean long video, titles and thumbnails

**Criticality tier:** `normal` (declared by the owner 2026-10-07): **don't overkill it** — simple and robust beats
complete.

**Owner:** Omar (`omar1001`). **Public** at github.com/omar1001/BroClips, MIT. Built with heavy AI assistance (Claude
Code) — the README says so honestly.

**Machine-specific notes:** if `CLAUDE.local.md` exists next to this file (git-ignored, only on the developer's PC),
read it first — it holds the local paths, the installed copy's settings and where the raw build notes are.

**What it is:** a local-first Windows desktop app (Python stdlib server + HTML pages, desktop shortcut). The user
records any video where someone talks (a game, a school lesson, a programming tutorial, talking head, podcast),
describes the project in a prompt, and BroClips makes Shorts/TikToks with word-by-word captions, a jump-cut long
video, titles/descriptions/hashtags and thumbnails (studio + 🎲 random + ✨ AI). Marathon mode = many small AI rounds
(list → tournament). AI: local (Ollama + faster-whisper) or keys for Gemini / OpenAI-compatible / Anthropic.
It is the general version of the owner's private "LoL Clips" app (not public).

**Owner's decisions (2026-10-07):** "I record, it edits" only (no scripts, no AI-made videos); separate app from LoL
Clips; all four AI options; Marathon mode included; name with the owner's "Bro" prefix; public + screenshots + honest
AI note; it must install, run and be explained **without** an AI assistant (README + Help page).

## Hard rules
- Never modify, move or delete the user's source videos — read-only.
- One heavy job at a time; unload local models after AI work (built for a 16 GB RAM PC).
- API keys only in `<data>/secrets.json`: never logged, never committed, masked in the UI, sent only to their provider.
- No console windows: `pythonw` + `CREATE_NO_WINDOW` for every subprocess. Heavy work waits while the programs listed
  in Settings run (e.g. a game); no background polling loops of console programs.
- Works with no key (local) and no GPU (CPU fallbacks), with plain-English errors.
- English UI, plain words, tooltips everywhere; Arabic (RTL) content must render correctly.
- Public pictures and docs: no local paths with a user name, no keys, no other people's names (e.g. game chat).

## Folder map
| Path | What |
|---|---|
| `docs/SPEC.md` | **The contract**: features, providers, pipeline, rendering, thumbnails, UI, installer, tests (read first) |
| `docs/CHANGELOG.md` | Dated record of what changed and why (full entries) |
| `docs/screenshots/` | README pictures (real app, real results); `hero.png` = the README banner |
| `broclips/` | The app (module map in SPEC §3); pages + design system in `broclips/web/` (`app.css`, `icons.js`) |
| `tests/` | pytest + `make_sample.py` (generates a sample lesson video) |
| `scripts/`, `install.bat` | Windows installer + shortcut; `scripts/make_icon.py` draws the app icon |

## Run
`install.bat` once (or `scripts\install.ps1 -Python <existing python>`), then the **BroClips** desktop shortcut, or
`python -m broclips` (`--no-window`, `--port`). Tests: `python -m pytest -q` (184 passed on 2026-10-07).

## Read on demand
| Question | Open |
|---|---|
| What exactly must feature X do? | `docs/SPEC.md` (section per feature) |
| Why was it built this way / what was decided? | `docs/CHANGELOG.md` |
| Which CSS tokens / components exist? | top of `broclips/web/app.css` (tokens in `:root`, components below) |
| Local paths, the installed copy, raw build notes | `CLAUDE.local.md` (developer's PC only) |

## Open items
- v1 is built, published and installed on the owner's PC — **waiting for his first real use / verdict.**
- Untested live: the cloud AI providers (Gemini, OpenAI-compatible, Anthropic) against real services; macOS/Linux.
- Known gaps (README "Known limitations" + CHANGELOG): no CJK/Hindi fonts; automatic thumbnails can pick a menu
  frame (no menu filter yet); no listening-prompt setting (hallucination risk); no speed-up control on the page; the
  README says "224 MB" for the light cut-out model where the app says "214 MB" (same file, MB vs MiB).

## Change log (index — full entries in `docs/CHANGELOG.md`)
| Date | Headline | Read before touching |
|---|---|---|
| 2026-10-07 | Public clean-up: raw build notes and machine-specific details moved out of the repo (`CLAUDE.local.md` locally); GitHub history replaced by one clean commit | CHANGELOG "public clean-up" |
| 2026-10-07 | UI redesign: one design system (`app.css`), Lucide icons (`icons.js`), status pill, stepper, Shorts in phone frames, provider tiles, confetti, new screenshots + hero banner; no behaviour changed | CHANGELOG "UI redesign" before touching `broclips/web/*` |
| 2026-10-07 | M4: README with real screenshots (privacy-checked), published public on GitHub, installed on the owner's PC | CHANGELOG "M4" |
| 2026-10-07 | M2: pipeline (listen, moments, Marathon mode, shorts with word captions, jump-cut long video, texts, results page) · M3: thumbnails + studio (subjects, 6 styles, 🎲, ✨ AI) | CHANGELOG "M2" / "M3" |
| 2026-10-07 | M1: AI providers (Ollama, Gemini, OpenAI-compatible, Anthropic, local/cloud Whisper) + app shell (work line, projects, server with POST guard, pages, assets, installer) | CHANGELOG "M1" |
| 2026-10-07 | Project started: spec, tier, decisions | `docs/SPEC.md` |
