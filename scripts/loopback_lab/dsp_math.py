"""AES17-aligned 64-bit spectral, phase, impulse, THD+N, and dither analyzers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Exact 7-Term Blackman-Harris coefficients (>180 dB sidelobe suppression)
BH7_COEFFS = (
    0.27105140069342,
    -0.43329793923448,
    0.21812299954311,
    -0.06592544638803,
    0.01081174209837,
    -0.00077658482522,
    0.00001388721735,
)


def blackman_harris_7(n: int) -> np.ndarray:
    """Generates a symmetric 7-term Blackman-Harris window (>180 dB sidelobe rejection)."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    phase = (2.0 * np.pi * k) / float(n)
    w = np.zeros(n, dtype=np.float64)
    for m, coeff in enumerate(BH7_COEFFS):
        w += coeff * np.cos(m * phase)
    return w


def trim_warmup(signal: np.ndarray, fs: float, warmup_seconds: float = 0.500) -> np.ndarray:
    """Discards leading warm-up samples (default 500 ms per AES17 lab pre-flight)."""
    skip = int(round(warmup_seconds * fs))
    if signal.shape[0] <= skip + 1024:
        return signal[min(skip, signal.shape[0] // 4) :]
    return signal[skip:]


def dbfs(linear_amp: float | np.ndarray, floor_db: float = -300.0) -> float | np.ndarray:
    """Converts peak linear amplitude (where 1.0 = 0 dBFS sine peak) to dBFS."""
    floor_lin = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(np.abs(linear_amp), floor_lin))


def rms_dbfs(signal: np.ndarray, aes17_sine_ref: bool = False, floor_db: float = -300.0) -> float:
    """Computes RMS in dBFS.

    By standard digital convention:
    - Raw digital RMS (`aes17_sine_ref=False`): 1.0 RMS = 0 dBFS; full-scale sine = -3.0103 dBFS;
      24-bit TPDF silence (`sigma = LSB / sqrt(6)`) = -142.40 dBFS (or -139.39 dB AES17).
    - AES17 sine-referenced (`aes17_sine_ref=True`): full-scale sine = 0.00 dBFS (+3.0103 dB offset).
    """
    x = np.asarray(signal, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x))) if x.size > 0 else 0.0
    scale = np.sqrt(2.0) if aes17_sine_ref else 1.0
    floor_lin = 10.0 ** (floor_db / 20.0)
    return float(20.0 * np.log10(max(rms * scale, floor_lin)))


def averaged_fft_dbfs(
    mono_signal: np.ndarray,
    fs: float,
    n_fft: int = 262144,
    averages: int = 8,
    window: str = "bh7",
    coherent_vector_avg: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes an N=262,144 averaged single-sided spectrum normalized so a 1.0-peak sine = 0.00 dBFS.

    Returns:
        freqs_hz: (n_fft // 2 + 1,)
        mag_dbfs: (n_fft // 2 + 1,) sine-peak-calibrated dBFS per bin
        power_bins: (n_fft // 2 + 1,) linear sine-equivalent squared peak power per bin
    """
    x = np.asarray(mono_signal, dtype=np.float64).ravel()
    if x.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: x.size] = x
        x = padded

    w = blackman_harris_7(n_fft) if window == "bh7" else np.ones(n_fft, dtype=np.float64)
    coherent_gain = float(np.sum(w))

    max_avail = x.size - n_fft
    if averages <= 1 or max_avail <= 0:
        starts = [0]
    else:
        if coherent_vector_avg and x.size >= averages * n_fft:
            # Exact contiguous N-sample blocks preserve bin-centered phase across all blocks!
            starts = [m * n_fft for m in range(averages)]
        else:
            step = max(1, max_avail // (averages - 1))
            starts = [min(i * step, max_avail) for i in range(averages)]

    if coherent_vector_avg:
        accum_complex = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
        for st in starts:
            seg = x[st : st + n_fft] * w
            accum_complex += np.fft.rfft(seg)
        mean_spec = accum_complex / float(len(starts))
        amp = (2.0 * np.abs(mean_spec)) / coherent_gain
        amp[0] *= 0.5
        if n_fft % 2 == 0:
            amp[-1] *= 0.5
        power_bins = amp * amp
    else:
        accum_power = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        for st in starts:
            seg = x[st : st + n_fft] * w
            spec = np.fft.rfft(seg)
            amp = (2.0 * np.abs(spec)) / coherent_gain
            amp[0] *= 0.5
            if n_fft % 2 == 0:
                amp[-1] *= 0.5
            accum_power += amp * amp
        power_bins = accum_power / float(len(starts))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag_db = 10.0 * np.log10(np.maximum(power_bins, 1e-60))
    return freqs, mag_db, power_bins


def measure_tone_level_dbfs(
    mono_signal: np.ndarray,
    fs: float,
    target_hz: float,
    n_fft: int = 262144,
    averages: int = 4,
) -> float:
    """Measures exact sine peak amplitude (dBFS) at target_hz free of FFT scalloping loss."""
    x = np.asarray(mono_signal, dtype=np.float64).ravel()
    if x.size < n_fft:
        n_fft = 1 << int(np.floor(np.log2(max(x.size, 1024))))
    w = blackman_harris_7(n_fft)
    enbw_bins = float(n_fft) * float(np.sum(w * w)) / float(np.sum(w) ** 2)

    _, _, power_bins = averaged_fft_dbfs(x, fs, n_fft=n_fft, averages=averages, window="bh7")
    bin_idx = int(round(target_hz * n_fft / fs))
    lo = max(1, bin_idx - 9)
    hi = min(len(power_bins) - 1, bin_idx + 9)
    tone_power = float(np.sum(power_bins[lo : hi + 1])) / enbw_bins
    return float(10.0 * np.log10(max(tone_power, 1e-60)))


@dataclass
class THDNResult:
    fundamental_dbfs: float
    thdn_db: float
    thdn_percent: float
    sfdr_dbc: float
    worst_spur_dbfs: float
    worst_spur_hz: float


def measure_thdn_and_sfdr(
    mono_signal: np.ndarray,
    fs: float,
    fundamental_hz: float,
    n_fft: int = 262144,
    averages: int = 8,
    band_low_hz: float = 20.0,
    band_high_hz: float = 20000.0,
) -> THDNResult:
    """AES17-aligned THD+N (20 Hz - 20 kHz) and SFDR analyzer using 7-term Blackman-Harris."""
    x = np.asarray(mono_signal, dtype=np.float64).ravel()
    w = blackman_harris_7(n_fft)
    enbw_bins = float(n_fft) * float(np.sum(w * w)) / float(np.sum(w) ** 2)

    freqs, mag_db, power_bins = averaged_fft_dbfs(
        x, fs, n_fft=n_fft, averages=averages, window="bh7"
    )
    df = fs / float(n_fft)
    k_fund = int(round(fundamental_hz / df))
    k_low = max(1, int(np.ceil(band_low_hz / df)))
    k_high = min(len(power_bins) - 2, int(np.floor(min(band_high_hz, 0.495 * fs) / df)))

    # Integrate fundamental across BH7 main lobe (+-10 bins)
    f_lo = max(1, k_fund - 10)
    f_hi = min(len(power_bins) - 2, k_fund + 10)
    fund_power = float(np.sum(power_bins[f_lo : f_hi + 1])) / enbw_bins
    fund_dbfs = float(10.0 * np.log10(max(fund_power, 1e-60)))

    # Mask fundamental (+-14 bins) for 20 Hz - 20 kHz THD+N integration
    mask = np.zeros_like(power_bins, dtype=bool)
    mask[k_low : k_high + 1] = True
    notch_lo = max(0, k_fund - 14)
    notch_hi = min(len(power_bins) - 1, k_fund + 14)
    mask[notch_lo : notch_hi + 1] = False

    nd_power = float(np.sum(power_bins[mask])) / enbw_bins
    ratio = max(nd_power / max(fund_power, 1e-60), 1e-30)
    thdn_db = float(10.0 * np.log10(ratio))
    thdn_pct = float(100.0 * np.sqrt(ratio))

    # SFDR across full Nyquist view (excluding DC < 15 Hz and fundamental main lobe)
    spur_mask = np.ones_like(mag_db, dtype=bool)
    spur_mask[: max(5, int(np.ceil(15.0 / df)))] = False
    spur_mask[notch_lo : notch_hi + 1] = False
    spur_mask[-4:] = False

    if np.any(spur_mask):
        spur_indices = np.where(spur_mask)[0]
        best_rel = int(np.argmax(mag_db[spur_indices]))
        spur_bin = int(spur_indices[best_rel])
        worst_spur_dbfs = float(mag_db[spur_bin])
        worst_spur_hz = float(freqs[spur_bin])
    else:
        worst_spur_dbfs = -300.0
        worst_spur_hz = 0.0

    sfdr_dbc = float(fund_dbfs - worst_spur_dbfs)
    return THDNResult(
        fundamental_dbfs=fund_dbfs,
        thdn_db=thdn_db,
        thdn_percent=thdn_pct,
        sfdr_dbc=sfdr_dbc,
        worst_spur_dbfs=worst_spur_dbfs,
        worst_spur_hz=worst_spur_hz,
    )


def generate_multitone_32(
    fs: float,
    duration_s: float,
    n_tones: int = 32,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    target_rms_dbfs: float = -18.0,
    n_fft: int = 262144,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a Newman/Schroeder-phased 32-tone AP-style log-spaced stimulus."""
    total_samples = int(round(duration_s * fs))
    raw_freqs = np.geomspace(f_min, min(f_max, 0.45 * fs), n_tones)
    df = fs / float(n_fft)
    bins = np.unique(np.maximum(1, np.round(raw_freqs / df).astype(int)))
    tone_freqs = bins.astype(np.float64) * df

    t = np.arange(total_samples, dtype=np.float64) / fs
    sig = np.zeros(total_samples, dtype=np.float64)
    n_actual = len(tone_freqs)
    for idx, f_hz in enumerate(tone_freqs):
        # Newman crest-factor-minimizing phase formula
        phi = (np.pi * (idx ** 2)) / float(n_actual)
        sig += np.sin(2.0 * np.pi * f_hz * t + phi)

    cur_rms = float(np.sqrt(np.mean(sig * sig)))
    desired_rms = 10.0 ** (target_rms_dbfs / 20.0)
    sig *= desired_rms / max(cur_rms, 1e-30)
    return sig, tone_freqs


def measure_multitone_tdn(
    mono_signal: np.ndarray,
    fs: float,
    tone_freqs_hz: np.ndarray,
    n_fft: int = 262144,
    averages: int = 8,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Measures Total Distortion + Noise (TD+N dB) and worst inter-tone grass (dBFS) on a 32-tone signal."""
    w = blackman_harris_7(n_fft)
    enbw_bins = float(n_fft) * float(np.sum(w * w)) / float(np.sum(w) ** 2)
    freqs, mag_db, power_bins = averaged_fft_dbfs(
        mono_signal, fs, n_fft=n_fft, averages=averages, window="bh7"
    )
    df = fs / float(n_fft)
    k_low = max(1, int(np.ceil(20.0 / df)))
    k_high = min(len(power_bins) - 2, int(np.floor(min(20000.0, 0.49 * fs) / df)))

    tone_mask = np.zeros_like(power_bins, dtype=bool)
    guard_mask = np.zeros_like(power_bins, dtype=bool)
    for f_hz in tone_freqs_hz:
        k = int(round(f_hz / df))
        tone_mask[max(1, k - 9) : min(len(power_bins) - 1, k + 10)] = True
        guard_mask[max(1, k - 14) : min(len(power_bins) - 1, k + 15)] = True

    band_mask = np.zeros_like(power_bins, dtype=bool)
    band_mask[k_low : k_high + 1] = True

    signal_power = float(np.sum(power_bins[tone_mask & band_mask])) / enbw_bins
    grass_bins = band_mask & (~guard_mask)
    tdn_power = float(np.sum(power_bins[grass_bins])) / enbw_bins

    tdn_db = float(10.0 * np.log10(max(tdn_power / max(signal_power, 1e-60), 1e-30)))
    grass_floor_dbfs = float(np.percentile(mag_db[grass_bins], 99.5)) if np.any(grass_bins) else -300.0
    return tdn_db, grass_floor_dbfs, freqs, mag_db


def extract_impulse_phase_and_group_delay(
    impulse_mono: np.ndarray,
    fs_out: float,
    n_fft: int = 65536,
    t0_index: int | None = None,
    gain_norm: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Extracts magnitude, unwrapped phase deviation (deg), and group delay (ms) from a Dirac impulse capture."""
    h = np.asarray(impulse_mono, dtype=np.float64).ravel() / max(gain_norm, 1e-30)
    peak_idx = int(np.argmax(np.abs(h)))

    # Reference window to exact causal input sample t0_index when known so group delay is true filter latency
    win_start = max(0, t0_index if t0_index is not None else (peak_idx - 4096))
    win_end = min(h.size, win_start + n_fft)
    h_seg = np.zeros(n_fft, dtype=np.float64)
    h_seg[: win_end - win_start] = h[win_start:win_end]

    spec = np.fft.rfft(h_seg)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_out)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-30))
    unwrapped_phase = np.unwrap(np.angle(spec))

    # Group delay: tau_g(w) = -d(phi)/dw
    omega = 2.0 * np.pi * freqs
    dphi = np.gradient(unwrapped_phase, omega)
    group_delay_ms = -dphi * 1000.0

    # Fit bulk linear delay across 200 Hz - 10 kHz to measure pure phase deviation (degrees)
    fit_mask = (freqs >= 200.0) & (freqs <= 10000.0)
    coeffs = np.polyfit(omega[fit_mask], unwrapped_phase[fit_mask], 1)
    linear_phase_fit = np.polyval(coeffs, omega)
    phase_dev_deg = np.rad2deg(unwrapped_phase - linear_phase_fit)
    # Remove any integer 180-deg offset at DC
    dc_offset_deg = float(np.median(phase_dev_deg[fit_mask]))
    phase_dev_deg -= dc_offset_deg

    return {
        "freqs_hz": freqs,
        "mag_db": mag_db,
        "phase_unwrapped_rad": unwrapped_phase,
        "phase_dev_deg": phase_dev_deg,
        "group_delay_ms": group_delay_ms,
        "peak_index": peak_idx,
    }
