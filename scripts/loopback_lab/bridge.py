"""C-ABI ctypes bridge for libNaitDSPBench.dylib and compatible custom DSP engines."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DYLIB_CANDIDATES = [
    REPO_ROOT / "lib" / "libNaitDSPBench.dylib",
    REPO_ROOT / ".build" / "release" / "libNaitDSPBench.dylib",
    REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "libNaitDSPBench.dylib",
]


class BenchCoreConfig(ctypes.Structure):
    _fields_ = [
        ("in_rate", ctypes.c_double),
        ("out_rate", ctypes.c_double),
        ("filter_mode", ctypes.c_int),
        ("force_1x_upsample", ctypes.c_int),
        ("enable_biquad_41hz", ctypes.c_int),
        ("biquad_gain_db", ctypes.c_double),
        ("triode_drive", ctypes.c_double),
        ("pad_db", ctypes.c_double),
        ("limiter_mode", ctypes.c_int),
        ("quantizer_mode", ctypes.c_int),
        ("prng_seed", ctypes.c_uint64),
    ]


class BenchRTConfig(ctypes.Structure):
    _fields_ = [
        ("in_rate", ctypes.c_double),
        ("out_rate", ctypes.c_double),
        ("callback_frames", ctypes.c_size_t),
        ("initial_filter_mode", ctypes.c_int),
        ("external_dac_mode", ctypes.c_int),
        ("vintage_amp_mode", ctypes.c_int),
        ("pad_db", ctypes.c_double),
        ("listening_db", ctypes.c_double),
        ("master_bypass", ctypes.c_int),
        ("spectral_tilt", ctypes.c_int),
        ("tilt_db", ctypes.c_double),
        ("bbc_dip", ctypes.c_int),
        ("triode_warmth", ctypes.c_double),
        ("transition_frame", ctypes.c_size_t),
        ("transition_kind", ctypes.c_int),
    ]


class BenchRTStats(ctypes.Structure):
    _fields_ = [
        ("out_frames_written", ctypes.c_size_t),
        ("ring_underruns", ctypes.c_size_t),
        ("peak_left_dbfs", ctypes.c_double),
        ("peak_right_dbfs", ctypes.c_double),
    ]


class NaitDSPBenchBridge:
    """Loads libNaitDSPBench.dylib (or a custom --dylib) and exposes NumPy wrappers."""

    def __init__(self, dylib_path: str | Path | None = None) -> None:
        resolved = self._locate_dylib(dylib_path)
        if resolved is None:
            raise FileNotFoundError(
                "DSP benchmark library not found. Expected lib/libNaitDSPBench.dylib "
                "or pass --dylib /path/to/custom.dylib implementing Sources/NaitDSPBench/include/NaitDSPBench.h."
            )

        self.dylib_path = resolved
        self._lib = ctypes.CDLL(str(resolved))
        self._bind_signatures()

    @staticmethod
    def _locate_dylib(explicit_path: str | Path | None = None) -> Path | None:
        override = explicit_path or os.environ.get("ANCHI_BENCH_DYLIB")
        if override:
            p = Path(override).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Specified --dylib does not exist: {p}")
            return p

        for candidate in DYLIB_CANDIDATES:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _bind_signatures(self) -> None:
        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_float_p = ctypes.POINTER(ctypes.c_float)

        self._lib.bench_passthrough_f64.argtypes = [
            c_double_p,
            ctypes.c_size_t,
            c_double_p,
        ]
        self._lib.bench_passthrough_f64.restype = None

        self._lib.bench_process_core_f64.argtypes = [
            ctypes.POINTER(BenchCoreConfig),
            c_double_p,
            ctypes.c_size_t,
            c_double_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.bench_process_core_f64.restype = ctypes.c_size_t

        self._lib.bench_stream_realtime_dsp.argtypes = [
            ctypes.POINTER(BenchRTConfig),
            c_float_p,
            ctypes.c_size_t,
            c_float_p,
            ctypes.c_size_t,
            ctypes.POINTER(BenchRTStats),
        ]
        self._lib.bench_stream_realtime_dsp.restype = ctypes.c_int

        if hasattr(self._lib, "bench_process_production_offline"):
            self._lib.bench_process_production_offline.argtypes = [
                ctypes.POINTER(BenchRTConfig),
                c_float_p,
                ctypes.c_size_t,
                c_float_p,
                ctypes.c_size_t,
            ]
            self._lib.bench_process_production_offline.restype = ctypes.c_size_t

    @staticmethod
    def _ensure_stereo_f64(signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            arr = np.column_stack([arr, arr])
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Expected mono (N,) or stereo (N, 2) array, got {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float64)

    @staticmethod
    def _ensure_stereo_f32(signal: np.ndarray) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.column_stack([arr, arr])
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Expected mono (N,) or stereo (N, 2) array, got {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def passthrough_f64(self, signal: np.ndarray) -> np.ndarray:
        src = self._ensure_stereo_f64(signal)
        frames = src.shape[0]
        dst = np.zeros_like(src)
        self._lib.bench_passthrough_f64(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(frames),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return dst

    def process_core_f64(
        self,
        signal: np.ndarray,
        in_rate: float,
        out_rate: float,
        *,
        filter_mode: int = 0,
        force_1x_upsample: bool = False,
        enable_biquad_41hz: bool = False,
        biquad_gain_db: float = 0.0,
        triode_drive: float = 0.0,
        pad_db: float = 0.0,
        limiter_mode: int = 0,
        quantizer_mode: int = 0,
        prng_seed: int = 0x853C49E6748FEA9B,
    ) -> tuple[np.ndarray, int]:
        src = self._ensure_stereo_f64(signal)
        in_frames = src.shape[0]
        max_out = int(np.ceil(in_frames * (out_rate / in_rate))) + 8192
        dst = np.zeros((max_out, 2), dtype=np.float64)
        lim_engaged = ctypes.c_size_t(0)

        cfg = BenchCoreConfig(
            in_rate=float(in_rate),
            out_rate=float(out_rate),
            filter_mode=int(filter_mode),
            force_1x_upsample=1 if force_1x_upsample else 0,
            enable_biquad_41hz=1 if enable_biquad_41hz else 0,
            biquad_gain_db=float(biquad_gain_db),
            triode_drive=float(triode_drive),
            pad_db=float(pad_db),
            limiter_mode=int(limiter_mode),
            quantizer_mode=int(quantizer_mode),
            prng_seed=int(prng_seed),
        )

        written = self._lib.bench_process_core_f64(
            ctypes.byref(cfg),
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(in_frames),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(max_out),
            ctypes.byref(lim_engaged),
        )
        return dst[:written], int(lim_engaged.value)

    def stream_realtime_dsp(
        self,
        signal: np.ndarray,
        in_rate: float,
        out_rate: float,
        *,
        callback_frames: int = 512,
        initial_filter_mode: int = 0,
        external_dac_mode: bool = True,
        vintage_amp_mode: bool = False,
        pad_db: float = -6.0,
        listening_db: float = 0.0,
        master_bypass: bool = True,
        spectral_tilt: bool = False,
        tilt_db: float = 0.0,
        bbc_dip: bool = False,
        triode_warmth: float = 0.0,
        transition_frame: int = 0,
        transition_kind: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        src = self._ensure_stereo_f32(signal)
        in_frames = src.shape[0]
        max_out = int(np.ceil(in_frames * (out_rate / in_rate))) + 8192
        dst = np.zeros((max_out, 2), dtype=np.float32)
        stats = BenchRTStats()

        cfg = BenchRTConfig(
            in_rate=float(in_rate),
            out_rate=float(out_rate),
            callback_frames=int(callback_frames),
            initial_filter_mode=int(initial_filter_mode),
            external_dac_mode=1 if external_dac_mode else 0,
            vintage_amp_mode=1 if vintage_amp_mode else 0,
            pad_db=float(pad_db),
            listening_db=float(listening_db),
            master_bypass=1 if master_bypass else 0,
            spectral_tilt=1 if spectral_tilt else 0,
            tilt_db=float(tilt_db),
            bbc_dip=1 if bbc_dip else 0,
            triode_warmth=float(triode_warmth),
            transition_frame=int(transition_frame),
            transition_kind=int(transition_kind),
        )

        rc = self._lib.bench_stream_realtime_dsp(
            ctypes.byref(cfg),
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(in_frames),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(max_out),
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"bench_stream_realtime_dsp failed with code {rc}")

        written = int(stats.out_frames_written)
        info = {
            "out_frames_written": written,
            "ring_underruns": int(stats.ring_underruns),
            "peak_left_dbfs": float(stats.peak_left_dbfs),
            "peak_right_dbfs": float(stats.peak_right_dbfs),
        }
        return dst[:written].astype(np.float64), info


def list_coreaudio_devices() -> list[dict[str, Any]]:
    """Lists CoreAudio devices via sounddevice (if installed) without modifying hardware volume."""
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        return [dict(d, index=i) for i, d in enumerate(devs)]
    except Exception:
        return []
