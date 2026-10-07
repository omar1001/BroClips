"""Make a sample "lesson" video for tests, demos and the README screenshots — no downloads, no private footage.

    python tests/make_sample.py [out.mp4]

A ~2-minute English mini-lesson ("Python for loops"): slides drawn by ffmpeg, narrated by Windows' built-in
text-to-speech (System.Speech, voice Zira/David), a highlight bar that moves line by line (so the follow-the-action
crop has something to follow) and a few long pauses (so jump cuts have something to cut). Windows only (TTS)."""
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
MONO = "C\\:/Windows/Fonts/consolab.ttf"

SLIDES = [  # (title, lines, narration, pause after in seconds)
    ("Python for loops in 2 minutes", ["The one loop you will use every day"],
     "Hey! In the next two minutes you will finally understand for loops in Python. "
     "They are the loop you will use every single day, so let's go.", 0.6),
    ("What a loop does", ["Repeat the same code", "once for every item", "in a list, a string or a range"],
     "A loop repeats the same code. Once for every item. The items can come from a list, a string, or a range of "
     "numbers. That is the whole idea.", 2.6),
    ("Your first loop", [">>> for i in range(3):", "...     print(i)", "0", "1", "2"],
     "Here is your first loop. For i in range three, print i. Python prints zero, one, two. "
     "Notice it starts at zero and stops before three.", 0.6),
    ("The mistake everybody makes", [">>> for i in range(1, 10):", "...     print(i)", "# stops at 9, not 10!"],
     "Now the mistake everybody makes. range one to ten does not print ten. It stops at nine. "
     "The end number is never included. Remember that and you will save hours of debugging.", 3.0),
    ("Looping over a list", [">>> games = ['LoL', 'Valorant', 'Minecraft']", ">>> for g in games:", "...     print(g)"],
     "You can loop over a list directly. For g in games, print g. No index, no counting. "
     "This is the most pythonic way to do it.", 0.6),
    ("Pro tip: enumerate", [">>> for n, g in enumerate(games, 1):", "...     print(n, g)", "1 LoL", "2 Valorant"],
     "Pro tip. If you need the number too, use enumerate. It gives you the number and the item together, "
     "and you can even start counting from one.", 2.2),
    ("Recap", ["for item in things:", "range(a, b) stops before b", "enumerate gives you the number"],
     "Quick recap. For item in things. Range stops before the end. And enumerate gives you the number. "
     "Follow for more two minute Python lessons!", 0.8),
]


def sh(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=NO_WINDOW)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return p


def tts(text, out_wav):
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "foreach ($v in @('Microsoft Zira Desktop','Microsoft David Desktop')) { try { $s.SelectVoice($v); break } "
          "catch {} }; $s.Rate = 0; $s.SetOutputToWaveFile($env:BC_WAV); $s.Speak($env:BC_TEXT); $s.Dispose()")
    import os
    env = dict(os.environ, BC_WAV=str(out_wav), BC_TEXT=text)
    p = subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, capture_output=True, text=True,
                       creationflags=NO_WINDOW)
    if p.returncode or not Path(out_wav).exists():
        raise RuntimeError("Windows text-to-speech failed: " + (p.stderr or "")[-500:])
    with wave.open(str(out_wav)) as w:
        return w.getnframes() / float(w.getframerate())


def esc_path(p):
    return str(p).replace("\\", "/").replace(":", "\\:")


def slide(k, title, lines, narration, pause, tmp):
    wav = tmp / f"s{k}.wav"
    dur = tts(narration, wav) + pause
    (tmp / f"t{k}.txt").write_text(title, encoding="utf-8")
    for j, ln in enumerate(lines):
        (tmp / f"l{k}_{j}.txt").write_text(ln, encoding="utf-8")
    n = max(1, len(lines))
    vf = [f"[0:v]drawbox=x=0:y=0:w=1920:h=10:color=0xffcc00@1:t=fill,"
          f"drawtext=fontfile='{FONT}':textfile='{esc_path(tmp / f't{k}.txt')}':fontsize=86:fontcolor=white:x=120:y=120,"
          f"drawtext=fontfile='{FONT}':text='{k + 1}/{len(SLIDES)}':fontsize=34:fontcolor=0x8a93a6:x=1740:y=1000[b]",
          f"color=c=0xffcc00@0.20:s=1700x78:r=30[hl]",
          f"[b][hl]overlay=x=110:y='292+92*min({n - 1},floor(t*{n}/{max(0.1, dur - pause):.3f}))':eval=frame[h]"]
    last = "h"
    for j, ln in enumerate(lines):
        font = MONO if ln.startswith((">>>", "...", "#")) or ln[:1].isdigit() else FONT
        vf.append(f"[{last}]drawtext=fontfile='{font}':textfile='{esc_path(tmp / f'l{k}_{j}.txt')}':fontsize=56:"
                  f"fontcolor={'0x9ef0a0' if font == MONO else 'white'}:x=140:y={300 + 92 * j}[v{j}]")
        last = f"v{j}"
    vf.append(f"[{last}]format=yuv420p[v];[1:a]apad=whole_dur={dur:.3f},aresample=48000[a]")
    out = tmp / f"p{k}.mp4"
    sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x161a24:s=1920x1080:r=30:d={dur:.3f}",
        "-i", wav, "-filter_complex", ";".join(vf), "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-ac", "2", out])
    return out, dur


def main(out_path):
    out_path = Path(out_path)
    tmp = Path(tempfile.mkdtemp(prefix="broclips_sample_"))
    try:
        parts, total = [], 0.0
        for k, (title, lines, narration, pause) in enumerate(SLIDES):
            p, d = slide(k, title, lines, narration, pause, tmp)
            parts.append(p)
            total += d
            print(f"slide {k + 1}/{len(SLIDES)}: {d:.1f} s")
        (tmp / "list.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sh(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", tmp / "list.txt", "-c", "copy",
            "-movflags", "+faststart", out_path])
        print(f"done: {out_path} ({total:.0f} s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "broclips_sample_lesson.mp4")
