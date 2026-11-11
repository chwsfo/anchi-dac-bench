"""Synthesizes studio-grade acoustic test stems, torture vectors, and safe level-matched WAV clips."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

try:
    import soundfile as sf  # Optional: only used when loading local .flac user tracks directly
except ImportError:
    sf = None  # type: ignore[assignment]

from .bridge import NaitDSPBenchBridge

REPO_ROOT = Path(__file__).resolve().parents[2]

# NEVER BLAST: Every generated audible WAV clip is strictly capped <= -6.02 dBFS (0.50 linear)
MAX_SAFE_WAV_PEAK = 0.495  # ~ -6.11 dBFS


@dataclass(frozen=True)
class ReferenceExcerpt:
    """Curated active musical window inside a local EBU SQAM or Open Goldberg master."""

    key: str
    label: str
    path: Path
    start_s: float = 0.0
    duration_s: float = 9.0
    native_fs: int = 44100


def discover_local_reference_library(repo_root: Path | None = None) -> dict[str, ReferenceExcerpt]:
    """Discovers local EBU Tech 3253 SQAM and Open Goldberg Variations FLACs staged in reports/loopback/."""
    root = repo_root or REPO_ROOT
    sqam_dir = root / "reports" / "loopback" / "sqam"
    gold_dir = root / "reports" / "loopback" / "goldberg"

    candidates: list[ReferenceExcerpt] = [
        # EBU Tech 3253 SQAM (44.1 kHz / 16-bit studio test masters)
        ReferenceExcerpt(
            "sqam_violin_ravel",
            "EBU SQAM #59 — Ravel Solo Violin (44.1 kHz / 16-bit)",
            sqam_dir / "59.flac",
            start_s=1.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_glockenspiel",
            "EBU SQAM #35 — Glockenspiel Melodious Phrase (44.1 kHz / 16-bit)",
            sqam_dir / "35.flac",
            start_s=34.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_triangle",
            "EBU SQAM #32 — Triangle Strikes & Roll (44.1 kHz / 16-bit)",
            sqam_dir / "32.flac",
            start_s=1.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_cymbal",
            "EBU SQAM #31 — Cymbal Shimmer & Transients (44.1 kHz / 16-bit)",
            sqam_dir / "31.flac",
            start_s=1.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_harpsichord",
            "EBU SQAM #40 — Harpsichord Melodious Phrase (44.1 kHz / 16-bit)",
            sqam_dir / "40.flac",
            start_s=22.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_piano",
            "EBU SQAM #39 — Grand Piano Chord & Silent-Booth Decay (44.1 kHz / 16-bit)",
            sqam_dir / "39.flac",
            start_s=110.1,
            duration_s=9.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_castanets",
            "EBU SQAM #27 — Castanets Flamenco Rhythm (44.1 kHz / 16-bit)",
            sqam_dir / "27.flac",
            start_s=2.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_snare",
            "EBU SQAM #28 — Side Drum / Snare Transients (44.1 kHz / 16-bit)",
            sqam_dir / "28.flac",
            start_s=1.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_guitar",
            "EBU SQAM #58 — Sarasate Acoustic Guitar (44.1 kHz / 16-bit)",
            sqam_dir / "58.flac",
            start_s=1.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_choir_orff",
            "EBU SQAM #64 — Orff Choir in Concert Hall (44.1 kHz / 16-bit)",
            sqam_dir / "64.flac",
            start_s=0.5,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_orchestra_strauss",
            "EBU SQAM #65 — R. Strauss Orchestra (44.1 kHz / 16-bit)",
            sqam_dir / "65.flac",
            start_s=2.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        ReferenceExcerpt(
            "sqam_abba_pop",
            "EBU SQAM #69 — ABBA Stereo Pop Chorus (44.1 kHz / 16-bit)",
            sqam_dir / "69.flac",
            start_s=8.0,
            duration_s=8.5,
            native_fs=44100,
        ),
        # Kimiko Ishizaka — Open Goldberg Variations (96.0 kHz / 24-bit, Teldex Studio Berlin)
        ReferenceExcerpt(
            "goldberg_aria",
            "Open Goldberg Variations — 01 Aria, Kimiko Ishizaka (96 kHz / 24-bit)",
            gold_dir / "Kimiko Ishizaka - J.S. Bach- -Open- Goldberg Variations, BWV 988 (Piano) - 01 Aria.flac",
            start_s=10.4,
            duration_s=9.0,
            native_fs=96000,
        ),
        ReferenceExcerpt(
            "goldberg_var13",
            "Open Goldberg Variations — 14 Variatio 13 Cantabile (96 kHz / 24-bit)",
            gold_dir / "Kimiko Ishizaka - J.S. Bach- -Open- Goldberg Variations, BWV 988 (Piano) - 14 Variatio 13 a 2 Clav..flac",
            start_s=1.0,
            duration_s=9.0,
            native_fs=96000,
        ),
        ReferenceExcerpt(
            "goldberg_var29",
            "Open Goldberg Variations — 29 Variatio 29 Bravura Chords (96 kHz / 24-bit)",
            gold_dir / "KimikoIshizaka-J.s.Bach--open-GoldbergVariationsBwv988piano-29Variatio29A1Ovvero2Clav..flac",
            start_s=0.5,
            duration_s=8.5,
            native_fs=96000,
        ),
        ReferenceExcerpt(
            "goldberg_final_decay",
            "Open Goldberg Variations — 31 Aria Da Capo Final Chord Decay (96 kHz / 24-bit)",
            gold_dir / "KimikoIshizaka-J.s.Bach--open-GoldbergVariationsBwv988piano-31AriaDaCapoEFine.flac",
            start_s=156.5,
            duration_s=7.5,
            native_fs=96000,
        ),
    ]
    return {item.key: item for item in candidates if item.path.exists()}


def _fade_edges(stereo: np.ndarray, fs: float, fade_ms: float = 20.0) -> np.ndarray:
    out = np.array(stereo, dtype=np.float64, copy=True)
    n_fade = min(int(round(fs * fade_ms / 1000.0)), out.shape[0] // 4)
    if n_fade > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade, dtype=np.float64) / float(n_fade)))
        out[:n_fade, 0] *= ramp
        out[:n_fade, 1] *= ramp
        out[-n_fade:, 0] *= ramp[::-1]
        out[-n_fade:, 1] *= ramp[::-1]
    return out


def _normalize_rms_and_safe_ceiling(
    stereo: np.ndarray,
    target_rms_dbfs: float = -16.0,
    ceiling_linear: float = MAX_SAFE_WAV_PEAK,
) -> np.ndarray:
    x = np.asarray(stereo, dtype=np.float64)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    rms = float(np.sqrt(np.mean(x * x)))
    if rms > 1e-12:
        desired_rms = 10.0 ** (target_rms_dbfs / 20.0)
        x = x * (desired_rms / rms)
    peak = float(np.max(np.abs(x)))
    if peak > ceiling_linear:
        x = x * (ceiling_linear / peak)
    return x


def _write_pcm24_wav(path: Path, stereo: np.ndarray, fs: int) -> None:
    """Writes stereo float64 [-1.0, +1.0) to 24-bit signed little-endian PCM WAV using stdlib wave."""
    pcm32 = np.clip(np.rint(stereo * 8388608.0), -8388608.0, 8388607.0).astype("<i4")
    pcm24_bytes = pcm32.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(stereo.shape[1])
        wf.setsampwidth(3)
        wf.setframerate(int(fs))
        wf.writeframes(pcm24_bytes)


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    """Reads 16/24/32-bit integer WAV via stdlib wave into stereo/multi-channel float64."""
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((b.shape[0], 4), dtype=np.uint8)
        padded[:, 1:4] = b
        i32 = padded.view("<i4").reshape(-1, nch)
        data = i32.astype(np.float64) / 2147483648.0
    elif sw == 2:
        i16 = np.frombuffer(raw, dtype="<i2").reshape(-1, nch)
        data = i16.astype(np.float64) / 32768.0
    elif sw == 4:
        i32 = np.frombuffer(raw, dtype="<i4").reshape(-1, nch)
        data = i32.astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    return data, int(sr)


def _write_safe_wav(path: Path, stereo: np.ndarray, fs: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(stereo, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > MAX_SAFE_WAV_PEAK:
        arr = arr * (MAX_SAFE_WAV_PEAK / peak)
    if sf is not None:
        sf.write(str(path), arr, int(fs), subtype="PCM_24")
    else:
        _write_pcm24_wav(path, arr, int(fs))
    return path


def load_optional_user_track(
    track_path: str | Path | ReferenceExcerpt,
    target_fs: int = 44100,
    max_seconds: float = 8.5,
    start_seconds: float | None = None,
) -> np.ndarray:
    """Loads an active window from a local .flac/.wav/.aiff/.m4a/.mp3 track (seeking before resampling)."""
    if isinstance(track_path, ReferenceExcerpt):
        src = track_path.path.expanduser().resolve()
        if start_seconds is None:
            start_seconds = track_path.start_s
        max_seconds = track_path.duration_s
    else:
        src = Path(track_path).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Track not found: {src}")

    if sf is not None:
        try:
            info = sf.info(str(src))
            sr = int(info.samplerate)
            total_frames = int(info.frames)
            want_frames = int(round(max_seconds * sr))
            if start_seconds is not None:
                start_frame = max(0, min(int(round(start_seconds * sr)), max(0, total_frames - 1)))
            elif total_frames > want_frames:
                start_frame = min(sr, total_frames - want_frames)
            else:
                start_frame = 0
            read_frames = min(want_frames, max(1, total_frames - start_frame))
            data, sr = sf.read(str(src), start=start_frame, frames=read_frames, always_2d=True, dtype="float64")
        except Exception:
            data = None
    else:
        data = None

    if data is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        try:
            subprocess.run(
                [
                    "/usr/bin/afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    f"LEI24@{target_fs}",
                    str(src),
                    str(tmp_wav),
                ],
                check=True,
                capture_output=True,
            )
            data, sr = _read_pcm_wav(tmp_wav)
        finally:
            tmp_wav.unlink(missing_ok=True)

        max_frames_raw = int(round(max_seconds * sr))
        if data.shape[0] > max_frames_raw:
            offset_raw = int(round((start_seconds or 1.0) * sr))
            offset_raw = max(0, min(offset_raw, data.shape[0] - max_frames_raw))
            data = data[offset_raw : offset_raw + max_frames_raw]

    if data.shape[1] == 1:
        data = np.column_stack([data[:, 0], data[:, 0]])
    elif data.shape[1] > 2:
        data = data[:, :2]

    if int(sr) != int(target_fs):
        g = int(np.gcd(int(sr), int(target_fs)))
        up = int(target_fs) // g
        down = int(sr) // g
        data = sp_signal.resample_poly(data, up, down, axis=0)

    max_frames = int(round(max_seconds * target_fs))
    if data.shape[0] > max_frames:
        data = data[:max_frames]

    return _fade_edges(data, float(target_fs), fade_ms=25.0)


def _synth_violin_and_brushes(fs: float = 44100.0, duration_s: float = 4.5) -> np.ndarray:
    """Synthesizes bowed acoustic violin notes + jazz wire brushes & high ride overtones (up to 20 kHz)."""
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(42)

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # Bowed violin melody notes (D5 -> F#5 -> A5 -> E5) with vibrato & upper harmonics to 19.8 kHz
    melody = [(0.0, 1.15, 587.33), (1.10, 2.25, 739.99), (2.20, 3.35, 880.00), (3.30, 4.45, 659.25)]
    for t0, t1, f0 in melody:
        mask = (t >= t0) & (t < t1)
        lt = t[mask] - t0
        dur = t1 - t0
        env = np.sin(np.pi * np.clip(lt / dur, 0.0, 1.0)) ** 0.65
        vib = 1.0 + 0.0032 * np.sin(2.0 * np.pi * 5.6 * lt)
        inst_phase = 2.0 * np.pi * f0 * np.cumsum(vib) / fs
        note = np.zeros_like(lt)
        max_h = int(min(28, np.floor(19900.0 / f0)))
        for h in range(1, max_h + 1):
            amp = (1.0 / (h ** 1.15)) * (1.25 if h in (2, 3, 5) else 1.0)
            note += amp * np.sin(h * inst_phase + 0.2 * h)
        left[mask] += 0.24 * env * note
        right[mask] += 0.19 * env * note

    # Jazz wire brushes + ride cymbal inharmonic modes (11 kHz - 19.6 kHz)
    cymbal_modes = [6340.0, 9420.0, 12870.0, 15430.0, 17890.0, 19420.0]
    beat_times = np.arange(0.15, duration_s - 0.2, 0.55)
    for bt in beat_times:
        mask = (t >= bt) & (t < min(duration_s, bt + 0.48))
        lt = t[mask] - bt
        brush_noise = rng.standard_normal(lt.size)
        b_hp, a_hp = sp_signal.butter(3, 5500.0 / (0.5 * fs), btype="high")
        brush_air = sp_signal.lfilter(b_hp, a_hp, brush_noise) * np.exp(-lt / 0.11)
        shimmer = np.zeros_like(lt)
        for m_idx, cf in enumerate(cymbal_modes):
            shimmer += (0.45 / (m_idx + 1.0)) * np.sin(2.0 * np.pi * cf * lt) * np.exp(-lt / (0.22 - 0.02 * m_idx))
        left[mask] += 0.12 * brush_air + 0.14 * shimmer
        right[mask] += 0.15 * brush_air + 0.18 * shimmer

    return _fade_edges(np.column_stack([left, right]), fs, fade_ms=30.0)


def _synth_piano_decay(fs: float = 44100.0, duration_s: float = 5.5) -> np.ndarray:
    """Synthesizes a warm Steinway-style chord strike + 5.5s natural inharmonic pedal decay."""
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    chord = [
        (185.00, 0.00, -0.25),  # F#3
        (277.18, 0.03, -0.05),  # C#4
        (369.99, 0.06, +0.10),  # F#4
        (440.00, 0.09, +0.25),  # A4
        (659.25, 0.12, +0.35),  # E5
    ]
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    inharmonicity_b = 0.00028

    for f0, onset_s, pan in chord:
        mask = t >= onset_s
        lt = t[mask] - onset_s
        attack = 1.0 - np.exp(-lt / 0.0035)
        note = np.zeros_like(lt)
        for h in range(1, 16):
            fh = f0 * h * np.sqrt(1.0 + inharmonicity_b * (h ** 2))
            if fh >= 0.45 * fs:
                break
            tau = 1.65 / (1.0 + 0.24 * (h - 1))
            amp = (1.0 / (h ** 1.32)) * attack * (0.72 * np.exp(-lt / tau) + 0.28 * np.exp(-lt / (tau * 2.1)))
            note += amp * np.sin(2.0 * np.pi * fh * lt + 0.35 * h)
        g_l = np.sqrt(0.5 * (1.0 - pan))
        g_r = np.sqrt(0.5 * (1.0 + pan))
        left[mask] += 0.28 * g_l * note
        right[mask] += 0.28 * g_r * note

    return _fade_edges(np.column_stack([left, right]), fs, fade_ms=25.0)


def _synth_percussive_prat_groove(fs: float = 44100.0, duration_s: float = 4.2) -> np.ndarray:
    """Synthesizes snappy snare rimshots, plucked upright bass, and crisp hi-hat transients."""
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(108)
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # Upright bass plucks
    bass_notes = [(0.05, 82.41), (0.55, 110.0), (1.05, 123.47), (1.55, 98.0), (2.05, 82.41), (2.55, 110.0), (3.05, 146.83), (3.55, 110.0)]
    for bt, f0 in bass_notes:
        mask = (t >= bt) & (t < min(duration_s, bt + 0.48))
        lt = t[mask] - bt
        pluck = np.zeros_like(lt)
        for h in range(1, 10):
            pluck += (1.0 / (h ** 1.25)) * np.sin(2.0 * np.pi * (f0 * h) * lt) * np.exp(-lt / (0.30 / h))
        click = np.exp(-lt / 0.0025) * np.sin(2.0 * np.pi * 1850.0 * lt) * 0.22
        sig = (1.0 - np.exp(-lt / 0.0015)) * (pluck + click)
        left[mask] += 0.32 * sig
        right[mask] += 0.32 * sig

    # Rimshots & crisp hi-hats with sharp wavefronts
    for step_idx, ht in enumerate(np.arange(0.05, duration_s - 0.15, 0.25)):
        mask = (t >= ht) & (t < min(duration_s, ht + 0.18))
        lt = t[mask] - ht
        hat = sp_signal.lfilter(*sp_signal.butter(3, 7200.0 / (0.5 * fs), btype="high"), rng.standard_normal(lt.size))
        hat *= np.exp(-lt / 0.028) * 0.22
        left[mask] += 0.8 * hat
        right[mask] += 1.1 * hat

        if step_idx % 4 == 2:
            rim = (
                0.55 * np.sin(2.0 * np.pi * 465.0 * lt) * np.exp(-lt / 0.018)
                + 0.45 * np.sin(2.0 * np.pi * 1920.0 * lt) * np.exp(-lt / 0.006)
                + 0.40 * rng.standard_normal(lt.size) * np.exp(-lt / 0.009)
            )
            left[mask] += 0.42 * rim
            right[mask] += 0.38 * rim

    return _fade_edges(np.column_stack([left, right]), fs, fade_ms=20.0)


def _synth_spacious_soundstage_ensemble(fs: float = 44100.0, duration_s: float = 4.8) -> np.ndarray:
    """Synthesizes a 3D chamber ensemble with inter-aural time delays (ITD) & early hall reflections."""
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # Three spatially separated acoustic voices: Left (-0.42 ms ITD), Center (0 ms), Right (+0.38 ms ITD)
    voices = [
        (0.10, [293.66, 440.00, 587.33, 880.00], -19, 0.78, 0.42),   # Left harp/guitar arpeggio
        (0.35, [369.99, 554.37, 739.99, 1108.73], 0, 0.65, 0.65),    # Dead-center oboe/flute harmonics
        (0.60, [440.00, 659.25, 987.77, 1318.51], +17, 0.42, 0.80),  # Right glockenspiel/strings
    ]
    for base_t, notes, itd_samples, gl, gr in voices:
        for idx, f0 in enumerate(notes):
            nt = base_t + idx * 0.85
            mask = (t >= nt) & (t < min(duration_s - 0.05, nt + 1.45))
            lt = t[mask] - nt
            src = np.zeros_like(lt)
            for h in range(1, 18):
                if f0 * h >= 20200.0:
                    break
                src += (1.0 / (h ** 1.18)) * np.sin(2.0 * np.pi * (f0 * h) * lt) * np.exp(-lt / (0.85 / np.sqrt(h)))
            src *= 1.0 - np.exp(-lt / 0.002)

            idx_arr = np.where(mask)[0]
            for s_offset, refl_gain in [(0, 1.0), (680, 0.24), (1390, 0.14), (2410, 0.08)]:
                pos_l = np.clip(idx_arr + s_offset + max(0, -itd_samples), 0, n - 1)
                pos_r = np.clip(idx_arr + s_offset + max(0, +itd_samples), 0, n - 1)
                left[pos_l] += 0.22 * gl * refl_gain * src
                right[pos_r] += 0.22 * gr * refl_gain * src

    return _fade_edges(np.column_stack([left, right]), fs, fade_ms=25.0)


def _synth_hot_modern_master_with_isps(fs: float = 44100.0, duration_s: float = 4.2) -> np.ndarray:
    """Synthesizes a loud modern master legally peaking at 0.00 dBFS sample-wise with > +3.2 dBTP ISPs."""
    base = _synth_percussive_prat_groove(fs=fs, duration_s=duration_s)
    n = base.shape[0]
    t = np.arange(n, dtype=np.float64) / fs

    # Add dense brass/horn chords + fs/4 (11.025 kHz @ 45 deg phase) transient bursts that reconstruct > +3 dBTP
    chords = (
        0.35 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 277.18 * t + 0.4)
        + 0.28 * np.sin(2.0 * np.pi * 329.63 * t + 0.9)
        + 0.22 * np.sin(2.0 * np.pi * 659.25 * t + 1.3)
    )
    isp_carrier = np.cos(2.0 * np.pi * (0.25 * fs) * t + np.pi / 4.0)
    burst_env = (np.sin(2.0 * np.pi * 2.0 * t) > 0.35).astype(np.float64)

    hot_l = 2.8 * base[:, 0] + 0.65 * chords + 0.72 * burst_env * isp_carrier
    hot_r = 2.8 * base[:, 1] + 0.65 * chords + 0.72 * burst_env * isp_carrier

    # Brickwall master at 44.1 kHz sample grid to +/-0.9999 (0.00 dBFS legal PCM, yet +3.3 dBTP between samples!)
    hot = np.column_stack([np.clip(hot_l, -0.9999, 0.9999), np.clip(hot_r, -0.9999, 0.9999)])
    return _fade_edges(hot, fs, fade_ms=20.0)


def _resolve_track_choice(
    choice: str | Path | ReferenceExcerpt | None,
    default_key: str,
    library: dict[str, ReferenceExcerpt],
) -> ReferenceExcerpt | Path | str | None:
    """Resolves a track specification ('__synth__', library key, ReferenceExcerpt, file path, or None -> local default)."""
    if choice == "__synth__":
        return None
    if isinstance(choice, ReferenceExcerpt):
        return choice
    if isinstance(choice, str) and choice in library:
        return library[choice]
    if choice is not None:
        return choice
    return library.get(default_key)


def _native_rate_for_spec(spec: ReferenceExcerpt | Path | str | None, fallback_fs: int = 44100) -> int:
    if isinstance(spec, ReferenceExcerpt):
        return int(spec.native_fs)
    if spec is not None and sf is not None:
        try:
            return int(sf.info(str(Path(spec).expanduser().resolve())).samplerate)
        except Exception:
            return fallback_fs
    return fallback_fs


def generate_day1_audition_clips(
    bridge: NaitDSPBenchBridge,
    out_dir: Path,
    custom_track: str | Path | ReferenceExcerpt | None = None,
) -> dict[str, Path]:
    """Generates Day 1 null residual (+80 dB), leaky ZOH vs Anchi 150 dB, twin-tone IMD, and Brickwall vs Craven Silk."""
    fs = 44100
    lib = discover_local_reference_library()

    violin_spec = _resolve_track_choice(custom_track, "sqam_violin_ravel", lib)
    top_spec = _resolve_track_choice(custom_track, "sqam_glockenspiel", lib)

    raw_music = (
        load_optional_user_track(violin_spec, target_fs=fs, max_seconds=8.5)
        if violin_spec is not None
        else _synth_violin_and_brushes(fs=float(fs))
    )
    music = _normalize_rms_and_safe_ceiling(raw_music, target_rms_dbfs=-16.0)

    raw_top = (
        load_optional_user_track(top_spec, target_fs=fs, max_seconds=8.5)
        if top_spec is not None
        else raw_music
    )
    top_music = _normalize_rms_and_safe_ceiling(raw_top, target_rms_dbfs=-16.0)

    # 1. Passthrough Null Residual boosted +80 dB (10,000x)
    pt_out = bridge.passthrough_f64(music)
    residual_80db = np.clip((pt_out - music) * (10.0 ** (80.0 / 20.0)), -MAX_SAFE_WAV_PEAK, MAX_SAFE_WAV_PEAK)
    p_null = _write_safe_wav(out_dir / "day1_01_passthrough_null_residual_plus80dB.wav", residual_80db, fs)

    # 2. Leaky Zero-Order-Hold NOS + Slew/Analog Stage IMD vs Anchi 150 dB Gaussian Upsampler
    # Repeat each 44.1k sample 4x (unfiltered ZOH at 176.4 kHz -> strong -10 dB ultrasonic mirrors!)
    zoh_176k = np.repeat(music, 4, axis=0)
    anchi_176k, _ = bridge.process_core_f64(music, 44100.0, 176400.0, filter_mode=0, pad_db=0.0)
    min_len = min(zoh_176k.shape[0], anchi_176k.shape[0])
    zoh_176k = zoh_176k[:min_len]
    anchi_176k = anchi_176k[:min_len]

    # Mild downstream analog output stage nonlinearity where ultrasonic images intermodulate into audible band
    def analog_stage_4x_to_1x(sig_176k: np.ndarray) -> np.ndarray:
        # High-frequency slew stress accentuates ultrasonic mirror intermodulation
        diff = np.vstack([np.zeros((1, 2)), np.diff(sig_176k, axis=0)]) * 3.2
        stressed = sig_176k + 0.22 * (diff * diff) - 0.12 * (diff ** 3)
        decimated = sp_signal.resample_poly(stressed, 1, 4, axis=0)
        return _normalize_rms_and_safe_ceiling(decimated, target_rms_dbfs=-16.0)

    p_jazz_leaky = _write_safe_wav(
        out_dir / "day1_02_jazz_violin_leaky_zoh_aliasing.wav", analog_stage_4x_to_1x(zoh_176k), fs
    )
    p_jazz_anchi = _write_safe_wav(
        out_dir / "day1_02_jazz_violin_anchi_150dB_clean.wav", analog_stage_4x_to_1x(anchi_176k), fs
    )

    # 3. 19 kHz + 20 kHz Twin-Tone IMD Audition (gentle -20 dBFS RMS)
    t = np.arange(int(3.0 * fs), dtype=np.float64) / float(fs)
    twin = 0.18 * (np.sin(2.0 * np.pi * 19000.0 * t) + np.sin(2.0 * np.pi * 20000.0 * t))
    twin_st = _fade_edges(np.column_stack([twin, twin]), float(fs), fade_ms=40.0)
    twin_zoh = np.repeat(twin_st, 4, axis=0)
    twin_anchi, _ = bridge.process_core_f64(twin_st, 44100.0, 176400.0, filter_mode=0, pad_db=0.0)
    ml = min(twin_zoh.shape[0], twin_anchi.shape[0])
    p_twin_leaky = _write_safe_wav(
        out_dir / "day1_03_twintone_19k_20k_leaky_zoh.wav",
        analog_stage_4x_to_1x(twin_zoh[:ml]) * 0.5,
        fs,
    )
    p_twin_anchi = _write_safe_wav(
        out_dir / "day1_03_twintone_19k_20k_anchi_clean.wav",
        analog_stage_4x_to_1x(twin_anchi[:ml]) * 0.5,
        fs,
    )

    # 4. Textbook Sharp Brickwall (+-0.25 dB equiripple to 21.8 kHz) vs Anchi Craven Gaussian Silk (-0.60 dB @ 20 kHz)
    # Rendered at 88.2 kHz WAV so the Enyo 15th / Anchi chain reproduces the full 18-22 kHz top-octave contour!
    fs_hi = 88200
    up_brick = sp_signal.resample_poly(top_music, 2, 1, axis=0, window=("kaiser", 14.0))
    # Inject realistic +-0.25 dB Parks-McClellan passband equiripple & brickwall ring on A
    f_rip = np.fft.rfftfreq(up_brick.shape[0], d=1.0 / fs_hi)
    rip_gain = np.where(f_rip <= 21800.0, 10.0 ** ((0.25 * np.cos(2.0 * np.pi * f_rip / 2600.0)) / 20.0), 0.0)
    brick_st = np.column_stack(
        [
            np.fft.irfft(np.fft.rfft(up_brick[:, 0]) * rip_gain, n=up_brick.shape[0]),
            np.fft.irfft(np.fft.rfft(up_brick[:, 1]) * rip_gain, n=up_brick.shape[0]),
        ]
    )
    anchi_88k, _ = bridge.process_core_f64(top_music, 44100.0, float(fs_hi), filter_mode=0, pad_db=0.0)
    p_brickwall = _write_safe_wav(
        out_dir / "day1_04_top_octave_equiripple_brickwall.wav",
        _normalize_rms_and_safe_ceiling(brick_st, target_rms_dbfs=-16.0),
        fs_hi,
    )
    p_craven_silk = _write_safe_wav(
        out_dir / "day1_04_top_octave_anchi_craven_silk.wav",
        _normalize_rms_and_safe_ceiling(anchi_88k, target_rms_dbfs=-16.0),
        fs_hi,
    )

    return {
        "null_residual": p_null,
        "jazz_leaky": p_jazz_leaky,
        "jazz_anchi": p_jazz_anchi,
        "twintone_leaky": p_twin_leaky,
        "twintone_anchi": p_twin_anchi,
        "top_brickwall": p_brickwall,
        "top_craven_silk": p_craven_silk,
    }


def generate_day2_audition_clips(
    out_dir: Path,
    custom_track: str | Path | ReferenceExcerpt | None = None,
) -> dict[str, Path]:
    """Generates Day 2 solo piano (default: EBU #39 silent-booth Steinway chord decay) H2 vs H3 vs truncation clips."""
    lib = discover_local_reference_library()
    piano_spec = _resolve_track_choice(custom_track, "sqam_piano", lib)

    native_sr = _native_rate_for_spec(piano_spec, fallback_fs=44100)
    fs = 96000 if native_sr >= 88200 else 44100

    raw_piano = (
        load_optional_user_track(piano_spec, target_fs=fs, max_seconds=9.0)
        if piano_spec is not None
        else _synth_piano_decay(fs=float(fs))
    )
    piano = _normalize_rms_and_safe_ceiling(raw_piano, target_rms_dbfs=-15.0)
    clips: dict[str, Path] = {}

    clips["piano_clean"] = _write_safe_wav(out_dir / "day2_01_piano_anchi_64bit_clean.wav", piano, fs)

    # Pure H2 (y = x + k2 * x^2, DC-blocked) & Pure H3/Odd (y = x + k3 * x^3 + k7 * x^7)
    b_dc, a_dc = sp_signal.butter(1, 8.0 / (0.5 * fs), btype="high")
    peak_ref = float(np.max(np.abs(piano)))

    for db_level in (-50, -70, -90):
        ratio = 10.0 ** (db_level / 20.0)
        # For a peak A sine, k2 * A^2 / 2 = ratio * A => k2 = 2 * ratio / A
        k2 = (2.0 * ratio) / max(peak_ref, 1e-6)
        h2_sig = sp_signal.lfilter(b_dc, a_dc, piano + k2 * (piano ** 2), axis=0)
        h2_sig = _normalize_rms_and_safe_ceiling(h2_sig, target_rms_dbfs=-15.0)
        clips[f"piano_h2_{abs(db_level)}"] = _write_safe_wav(
            out_dir / f"day2_02_piano_H2_warmth_{db_level}dB.wav", h2_sig, fs
        )

        # Odd-order (H3 + H5/H7 metallic edge)
        k3 = (4.0 * ratio) / max(peak_ref ** 2, 1e-6)
        h3_sig = piano - 0.7 * k3 * (piano ** 3) + 0.3 * (k3 / max(peak_ref ** 4, 1e-6)) * (piano ** 7)
        h3_sig = _normalize_rms_and_safe_ceiling(h3_sig, target_rms_dbfs=-15.0)
        clips[f"piano_h3_{abs(db_level)}"] = _write_safe_wav(
            out_dir / f"day2_03_piano_H3_odd_edge_{db_level}dB.wav", h3_sig, fs
        )

    # Undithered low-bit truncation demonstration (11-bit stair-step so granulation on decay is unmistakable)
    q_levels = 1024.0
    trunc_sig = np.trunc(piano * q_levels) / q_levels
    trunc_sig = _normalize_rms_and_safe_ceiling(trunc_sig, target_rms_dbfs=-15.0)
    clips["piano_truncation"] = _write_safe_wav(
        out_dir / "day2_04_piano_undithered_truncation_grit.wav", trunc_sig, fs
    )

    return clips


def generate_day3_audition_clips(
    bridge: NaitDSPBenchBridge,
    out_dir: Path,
    custom_percussive_track: str | Path | ReferenceExcerpt | None = None,
    custom_spacious_track: str | Path | ReferenceExcerpt | None = None,
) -> dict[str, Path]:
    """Generates Day 3 16x slowed Dirac impulses + Linear vs PRaT Min-Phase ABX stems (default: EBU Castanets & Orff Choir)."""
    fs = 44100
    lib = discover_local_reference_library()
    clips: dict[str, Path] = {}

    # 1. 16x Time-Stretched Dirac Impulse ("zwoop-CLICK-poowz" vs "CLICK-poowz")
    imp_in = np.zeros((4096, 2), dtype=np.float64)
    imp_in[512] = [0.5, 0.5]
    ir_lin, _ = bridge.process_core_f64(imp_in, 44100.0, 176400.0, filter_mode=0, quantizer_mode=0)
    ir_min, _ = bridge.process_core_f64(imp_in, 44100.0, 176400.0, filter_mode=1, quantizer_mode=0)

    def build_slowed_16x_clip(ir_176k: np.ndarray) -> np.ndarray:
        # Extract +-256 samples @ 176.4 kHz around impulse, upsample 4x and play at 44.1 kHz
        # (176.4k / 44.1k * 4 = 16x time stretch! 20.4 kHz ringing becomes an unmistakable 1,277 Hz tone!)
        p_idx = int(np.argmax(np.abs(ir_176k[:, 0])))
        seg = ir_176k[max(0, p_idx - 600) : min(ir_176k.shape[0], p_idx + 900)]
        stretched = sp_signal.resample_poly(seg, 4, 1, axis=0)
        peak = float(np.max(np.abs(stretched)))
        if peak > 1e-12:
            stretched = stretched * (0.42 / peak)
        gap = np.zeros((int(0.45 * fs), 2), dtype=np.float64)
        three_hits = np.vstack([gap, stretched, gap, stretched, gap, stretched, gap])
        return _fade_edges(three_hits, float(fs), fade_ms=10.0)

    clips["impulse_16x_linear"] = _write_safe_wav(
        out_dir / "day3_01_impulse_16x_slowed_linear_phase.wav", build_slowed_16x_clip(ir_lin), fs
    )
    clips["impulse_16x_min_phase"] = _write_safe_wav(
        out_dir / "day3_01_impulse_16x_slowed_min_phase_prat.wav", build_slowed_16x_clip(ir_min), fs
    )

    # 2. Percussive & Spacious ABX Tracks processed through Anchi Linear vs PRaT Min-Phase (2x integer synchronous!)
    perc_spec = _resolve_track_choice(custom_percussive_track, "sqam_castanets", lib)
    perc_native = _native_rate_for_spec(perc_spec, fallback_fs=44100)
    perc_in_fs = 48000 if (perc_native % 48000 == 0) else 44100
    perc_out_fs = perc_in_fs * 2

    perc_src = (
        load_optional_user_track(perc_spec, target_fs=perc_in_fs, max_seconds=8.5)
        if perc_spec is not None
        else _synth_percussive_prat_groove(fs=float(perc_in_fs))
    )
    perc_src = _normalize_rms_and_safe_ceiling(perc_src, target_rms_dbfs=-15.0)
    perc_lin, _ = bridge.process_core_f64(
        perc_src, float(perc_in_fs), float(perc_out_fs), filter_mode=0, pad_db=-6.0, quantizer_mode=2
    )
    perc_min, _ = bridge.process_core_f64(
        perc_src, float(perc_in_fs), float(perc_out_fs), filter_mode=1, pad_db=-6.0, quantizer_mode=2
    )

    clips["abx_perc_linear"] = _write_safe_wav(out_dir / "day3_02_abx_percussive_linear.wav", perc_lin, perc_out_fs)
    clips["abx_perc_min"] = _write_safe_wav(out_dir / "day3_02_abx_percussive_min_phase.wav", perc_min, perc_out_fs)

    spac_spec = _resolve_track_choice(custom_spacious_track, "sqam_choir_orff", lib)
    spac_native = _native_rate_for_spec(spac_spec, fallback_fs=44100)
    spac_in_fs = 48000 if (spac_native % 48000 == 0) else 44100
    spac_out_fs = spac_in_fs * 2

    spac_src = (
        load_optional_user_track(spac_spec, target_fs=spac_in_fs, max_seconds=8.5)
        if spac_spec is not None
        else _synth_spacious_soundstage_ensemble(fs=float(spac_in_fs))
    )
    spac_src = _normalize_rms_and_safe_ceiling(spac_src, target_rms_dbfs=-15.0)
    spac_lin, _ = bridge.process_core_f64(
        spac_src, float(spac_in_fs), float(spac_out_fs), filter_mode=0, pad_db=-6.0, quantizer_mode=2
    )
    spac_min, _ = bridge.process_core_f64(
        spac_src, float(spac_in_fs), float(spac_out_fs), filter_mode=1, pad_db=-6.0, quantizer_mode=2
    )

    clips["abx_space_linear"] = _write_safe_wav(out_dir / "day3_03_abx_spacious_linear.wav", spac_lin, spac_out_fs)
    clips["abx_space_min"] = _write_safe_wav(out_dir / "day3_03_abx_spacious_min_phase.wav", spac_min, spac_out_fs)

    return clips


def generate_day4_audition_clips(
    bridge: NaitDSPBenchBridge,
    out_dir: Path,
    custom_hot_track: str | Path | ReferenceExcerpt | None = None,
    custom_tail_track: str | Path | ReferenceExcerpt | None = None,
) -> dict[str, Path]:
    """Generates Day 4 Loud Master Rescue (default: EBU #69 ABBA) + +70 dB Reverb Tail Microscope (default: EBU #39)."""
    fs = 44100
    lib = discover_local_reference_library()
    clips: dict[str, Path] = {}

    # 1. Loud Master Rescue: Unprotected Naive 0 dB Oversampler + Vintage Input Overload vs -4.5 dB Safety Pad
    hot_spec = _resolve_track_choice(custom_hot_track, "sqam_abba_pop", lib)
    if hot_spec is not None:
        raw_hot = load_optional_user_track(hot_spec, target_fs=fs, max_seconds=8.5)
        # Apply modern hot-mastering top-shelf + brickwall sample ceiling (+/-0.9999 = 0.00 dBFS) so real drums/vocals create > +1.4 dBTP ISPs!
        b_air, a_air = sp_signal.butter(2, 5500.0 / (0.5 * fs), btype="high")
        bright = raw_hot + 1.35 * sp_signal.lfilter(b_air, a_air, raw_hot, axis=0)
        pk = float(np.max(np.abs(bright)))
        if pk > 1e-9:
            hot_src = np.clip((bright / pk) * (10.0 ** (9.0 / 20.0)), -0.9999, 0.9999)
            hot_src = _fade_edges(hot_src, float(fs), fade_ms=20.0)
        else:
            hot_src = _synth_hot_modern_master_with_isps(fs=float(fs))
    else:
        hot_src = _synth_hot_modern_master_with_isps(fs=float(fs))

    # Path A: Naive unprotected 0 dB oversampler clip + overloaded vintage preamp input rail
    isp_clipped_176k, _ = bridge.process_core_f64(
        hot_src, 44100.0, 176400.0, filter_mode=0, pad_db=0.0, limiter_mode=1, quantizer_mode=2
    )
    # Simulate hot 2.0 Vrms output slamming a sensitive vintage input stage without pad
    overloaded = np.tanh(isp_clipped_176k * 1.85) / np.tanh(1.85)

    # Path B: Anchi Vintage Safety Pad (-4.5 dB) -> 0% ISP clipping, pure linear analog stage headroom
    isp_rescued_176k, _ = bridge.process_core_f64(
        hot_src, 44100.0, 176400.0, filter_mode=0, pad_db=-4.5, limiter_mode=2, quantizer_mode=2
    )

    # Decimate cleanly to 88.2 kHz and lock BOTH to the exact same RMS (0.000 dB loudness bias) below -6.11 dBFS!
    rescued_88k = _normalize_rms_and_safe_ceiling(
        sp_signal.resample_poly(isp_rescued_176k, 1, 2, axis=0), target_rms_dbfs=-18.0
    )
    matched_rms_db = 20.0 * float(np.log10(max(float(np.sqrt(np.mean(rescued_88k * rescued_88k))), 1e-12)))
    clipped_88k = _normalize_rms_and_safe_ceiling(
        sp_signal.resample_poly(overloaded, 1, 2, axis=0), target_rms_dbfs=matched_rms_db
    )

    clips["loud_master_clipped"] = _write_safe_wav(
        out_dir / "day4_01_loud_master_naive_isp_clipped_matched.wav", clipped_88k, 88200
    )
    clips["loud_master_rescued"] = _write_safe_wav(
        out_dir / "day4_01_loud_master_safety_pad_rescued_matched.wav", rescued_88k, 88200
    )

    # 2. The +70 dB Reverb-Tail Microscope
    # Default to EBU #39 silent-booth Steinway chord decay (zero studio mic hiss so Truncation vs RPDF vs TPDF is unmistakable!)
    rng = np.random.default_rng(2026)
    tail_spec = _resolve_track_choice(custom_tail_track, "sqam_piano", lib)
    if tail_spec is not None and not (custom_tail_track is None and custom_hot_track == "__synth__"):
        tail_native = _native_rate_for_spec(tail_spec, fallback_fs=44100)
        tail_fs = 96000 if tail_native >= 88200 else 44100
        real_decay = load_optional_user_track(tail_spec, target_fs=tail_fs, max_seconds=8.5)
        pk_decay = float(np.max(np.abs(real_decay)))
        # Place initial chord strike at -77.5 dBFS peak so the decay crosses the -96.3 dBFS quantization step cleanly!
        target_strike = 10.0 ** (-77.5 / 20.0)
        quiet_tail = real_decay * (target_strike / max(pk_decay, 1e-12))
    else:
        tail_fs = fs
        dur_s = 4.8
        n = int(round(dur_s * tail_fs))
        t = np.arange(n, dtype=np.float64) / float(tail_fs)
        env = 10.0 ** ((-82.0 - 4.8 * t) / 20.0) * (1.0 - np.exp(-t / 0.015))
        guitar_l = env * (
            0.70 * np.sin(2.0 * np.pi * 329.63 * t)
            + 0.22 * np.sin(2.0 * np.pi * 659.25 * t + 0.5)
            + 0.08 * np.sin(2.0 * np.pi * 988.89 * t + 1.1)
        )
        guitar_r = env * (
            0.70 * np.sin(2.0 * np.pi * 329.63 * t + 0.15)
            + 0.22 * np.sin(2.0 * np.pi * 659.25 * t + 0.65)
            + 0.08 * np.sin(2.0 * np.pi * 988.89 * t + 1.25)
        )
        quiet_tail = np.column_stack([guitar_l, guitar_r])

    scale16 = 32768.0
    lsb16 = 1.0 / scale16
    boost_70db = 10.0 ** (70.0 / 20.0)  # +70.0 dB microscope gain!

    # (a) Truncated / rounded without dither: signal crumbles into buzz then gates off into dead vacuum
    q_trunc = (np.rint(quiet_tail * scale16) * lsb16) * boost_70db
    # (b) 1-bit RPDF (rectangular dither): distortion drops, but noise variance modulates/pumps with signal!
    rpdf_noise = (rng.random(quiet_tail.shape) - 0.5) * lsb16
    q_rpdf = (np.rint((quiet_tail + rpdf_noise) * scale16) * lsb16) * boost_70db
    # (c) Anchi TPDF (sum of two independent uniform [-0.5, +0.5] LSB sources): zero noise modulation, silky decay!
    tpdf_noise = ((rng.random(quiet_tail.shape) - 0.5) + (rng.random(quiet_tail.shape) - 0.5)) * lsb16
    q_tpdf = (np.rint((quiet_tail + tpdf_noise) * scale16) * lsb16) * boost_70db

    clips["tail_truncated"] = _write_safe_wav(
        out_dir / "day4_02_reverb_tail_plus70dB_truncated.wav", _fade_edges(q_trunc, float(tail_fs)), tail_fs
    )
    clips["tail_rpdf"] = _write_safe_wav(
        out_dir / "day4_02_reverb_tail_plus70dB_rpdf_1bit.wav", _fade_edges(q_rpdf, float(tail_fs)), tail_fs
    )
    clips["tail_tpdf"] = _write_safe_wav(
        out_dir / "day4_02_reverb_tail_plus70dB_anchi_tpdf.wav", _fade_edges(q_tpdf, float(tail_fs)), tail_fs
    )

    return clips
