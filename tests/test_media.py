"""media.py on tiny clips made with ffmpeg's built-in test sources (no downloads, no real recordings)."""
import hashlib
import io
import shutil
import subprocess
import wave

import numpy as np
import pytest

from broclips import media

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True, creationflags=NO_WINDOW)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """3 s, 320x240 at 30 fps, TWO audio tracks (like game sound + microphone)."""
    out = tmp_path_factory.mktemp("media") / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
           "-f", "lavfi", "-i", "sine=f=880:d=3", "-map", "0:v", "-map", "1:a", "-map", "2:a",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out)
    return out


def write_wav(path, x, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(x, np.int16).tobytes())
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe(clip, tmp_path):
    info = media.probe(clip)
    assert info["dur"] == pytest.approx(3.0, abs=0.1) and info["fps"] == pytest.approx(30.0)
    assert (info["width"], info["height"], info["n_audio"]) == (320, 240, 2)
    assert info["has_video"] is True and info["codec"] == "h264"
    wav = write_wav(tmp_path / "a.wav", np.zeros(16000))
    a = media.probe(wav)
    assert a["has_video"] is False and a["n_audio"] == 1 and a["dur"] == pytest.approx(1.0, abs=0.01)
    bad = tmp_path / "notes.mp4"
    bad.write_text("not a video")
    with pytest.raises(RuntimeError, match="Cannot read notes.mp4"):
        media.probe(bad)


def test_extract_audio16k_mixes_all_tracks_and_leaves_the_source_alone(clip, tmp_path):
    before, mtime = sha(clip), clip.stat().st_mtime
    out = media.extract_audio16k(clip, tmp_path / "work" / "audio16k.wav")
    with wave.open(str(out), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() / 16000 == pytest.approx(3.0, abs=0.1)
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    spec = np.abs(np.fft.rfft(x[16000:32000]))                     # one second: both tones must be in the mix
    assert spec[440] > 50 * np.median(spec) and spec[880] > 50 * np.median(spec)
    assert sha(clip) == before and clip.stat().st_mtime == mtime  # read-only (SPEC §2.1)
    assert not list(out.parent.glob("*.part*"))
    silent = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x64:r=10:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", silent)
    with pytest.raises(RuntimeError, match="has no sound"):
        media.extract_audio16k(silent, tmp_path / "x.wav")


def test_frames(clip):
    from PIL import Image
    jpg = media.frame_jpg(clip, 1.0)
    assert jpg[:2] == b"\xff\xd8" and Image.open(io.BytesIO(jpg)).size == (320, 240)
    assert Image.open(io.BytesIO(media.frame_jpg(clip, 1.0, width=160))).size == (160, 120)
    assert Image.open(io.BytesIO(media.frame_jpg(clip, 1.0, vf="crop=100:80:0:0"))).size == (100, 80)
    assert media.frame_jpg(clip, 60.0) is None                      # after the end: no frame
    rgb = media.frame_rgb(clip, 0.5, 32, 24)
    assert rgb.shape == (24, 32, 3) and rgb.dtype == np.uint8 and rgb.flags.writeable and rgb.max() > 0
    assert media.frame_rgb(clip, 60.0, 32, 24) is None


def test_loudness_spikes(tmp_path):
    wav = tmp_path / "loud.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=20:sample_rate=16000", "-af",
           "volume='if(between(t,10,12),1,0.01)':eval=frame", "-ac", "1", "-c:a", "pcm_s16le", wav)
    per_s, spikes = media.loudness_spikes(wav)
    assert len(per_s) == 20 and all(type(v) is float for v in per_s)
    assert spikes and set(spikes) <= {10, 11, 12} and (10 in spikes or 11 in spikes)


def test_encoder_is_detected_once(monkeypatch):
    calls = []
    real = media._run

    def counting(cmd, **kw):
        calls.append(cmd)
        return real(cmd, **kw)
    monkeypatch.setattr(media, "_run", counting)
    monkeypatch.setattr(media, "_ENCODER", {})
    first = media.encoder("auto")
    assert first in ("h264_nvenc", "libx264") and media.encoder("auto") == first and len(calls) == 1
    assert media.encoder("x264") == "libx264" and media.encoder("nvenc") == "h264_nvenc" and len(calls) == 1


def test_blur_filter_boxes():
    assert media.blur_filter([[1650, 950, 190, 50]]) == "delogo=x=1650:y=950:w=190:h=50"
    assert media.blur_filter([[1800, 1000, 120, 80]]) == "delogo=x=1800:y=1000:w=118:h=78"    # kept inside
    assert media.blur_filter([[0, 0, 100, 50]]) == "delogo=x=2:y=2:w=98:h=48"
    assert media.blur_filter([[1650, 950, 190, 50]], 1280, 720) == "delogo=x=1100:y=633:w=127:h=34"
    assert media.blur_filter([[1, 2], ["a", 1, 1, 1], [5000, 10, 10, 10]]) == ""
    assert media.blur_filter([]) == "" and media.blur_filter(None) == ""
    two = media.blur_filter([[10, 10, 50, 50], [100, 100, 50, 50]])
    assert two.count("delogo=") == 2 and "," in two


@pytest.mark.parametrize("w,h,boxes", [(1920, 1080, [[1800, 1000, 120, 80], [0, 0, 300, 40]]),
                                       (1280, 720, [[1650, 950, 270, 130]])])
def test_blur_filter_is_accepted_by_ffmpeg(w, h, boxes):
    vf = media.blur_filter(boxes, w, h)
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:d=0.2:r=10", "-vf", vf,
                        "-f", "null", "-"], capture_output=True, text=True, creationflags=NO_WINDOW)
    assert p.returncode == 0, p.stderr


def test_split_audio_cuts_in_quiet_moments(tmp_path):
    sr = 16000
    t = np.arange(25 * sr) / sr
    x = 8000 * np.sin(2 * np.pi * 300 * t)
    for a, b in ((8.0, 8.6), (17.0, 17.6)):                       # two short pauses
        x[int(a * sr):int(b * sr)] = 0
    wav = write_wav(tmp_path / "talk.wav", x)
    pieces = media.split_audio(wav, tmp_path / "pieces", piece_s=10, search_s=4)
    assert len(pieces) == 3
    cuts = [off for _, off, _ in pieces[1:]]
    assert 8.0 <= cuts[0] <= 8.6 and 17.0 <= cuts[1] <= 17.6
    assert sum(d for _, _, d in pieces) == pytest.approx(25.0, abs=0.01)
    assert all(d <= 10 for _, _, d in pieces)
    for path, _, dur in pieces:
        assert path.endswith(".mp3") and media.probe(path)["dur"] == pytest.approx(dur, abs=0.15)
    wavs = media.split_audio(wav, tmp_path / "w", piece_s=10, search_s=4, fmt="wav")
    assert [round(o, 2) for _, o, _ in wavs] == [round(o, 2) for _, o, _ in pieces]
    one = media.split_audio(wav, tmp_path / "one", piece_s=60)
    assert len(one) == 1 and one[0][1] == 0.0
