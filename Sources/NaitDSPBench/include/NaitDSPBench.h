#ifndef NAIT_DSP_BENCH_H
#define NAIT_DSP_BENCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define NAIT_BENCH_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define NAIT_BENCH_API __attribute__((visibility("default")))
#else
  #define NAIT_BENCH_API
#endif

typedef struct {
    double in_rate;
    double out_rate;
    int filter_mode;        // 0 = Gaussian Smooth Linear, 1 = PRaT Minimum-Phase
    int force_1x_upsample;  // 0 = pure 1:1 passthrough when in_rate==out_rate & drive==0; 1 = force 4x->1x stage
    int enable_biquad_41hz; // 1 = engage 64-bit Biquad64 41 Hz low-shelf pole stress test
    double biquad_gain_db;  // Gain in dB for 41 Hz pole stress test (e.g. 0.0 or +1.0 dB)
    double triode_drive;    // [0.0 .. 0.12]
    double pad_db;          // Attenuation pad in dB (0.0 for raw bench, <= -3.0 / -6.0 for padded tests)
    int limiter_mode;       // 0 = Off (unclamped Float64)
                            // 1 = Naive 0.0 dBFS hard clip [-1.0, +1.0]
                            // 2 = Hard Safety Limiter (<= -0.05 dBFS ceiling, zero wrap, <30 ms release)
                            // 3 = Production NAIT 1 soft-knee + -6.0 dBFS ceiling
                            // 4 = Production External DAC soft-knee + -2.0 dBFS ceiling
    int quantizer_mode;     // 0 = Float64 (Dither OFF)
                            // 1 = Float32 (IEEE-754 float transport)
                            // 2 = 24-bit Int TPDF (XorShift64Star flat TPDF on 24-bit signed PCM grid)
                            // 3 = 24-bit Int LNS  (2nd-order ultrasonic noise-shaped 24-bit TPDF)
                            // 4 = 24-bit Int Truncated (undithered 24-bit quantization for comparison)
    uint64_t prng_seed;     // PRNG seed (0 -> default deterministic seed)
} BenchCoreConfig;

typedef struct {
    double in_rate;
    double out_rate;
    size_t callback_frames; // Input chunk frames per simulated AUHAL callback (e.g. 512 or 64)
    int initial_filter_mode;// 0 = Gaussian Linear, 1 = PRaT Min-Phase
    int external_dac_mode;  // 0 = Vintage/NAIT (-6 dBFS ceiling), 1 = External DAC (-2 dBFS ceiling)
    int vintage_amp_mode;   // 1 = Lock -6 dBFS ceiling
    double pad_db;          // [-36.0 .. -3.0]
    double listening_db;    // [-50.0 .. 0.0]
    int master_bypass;      // 1 = Pure mode (Stage 1B-2 bypassed, Stage 4/5 active)
    int spectral_tilt;      // 1 = Spectral tilt enabled
    double tilt_db;         // [-2.0 .. +2.0]
    int bbc_dip;            // 1 = BBC Gundry dip enabled
    double triode_warmth;   // [0.0 .. 1.0]
    size_t transition_frame;// Input frame index at which to trigger a live transition (0 = none)
    int transition_kind;    // 0 = None, 1 = Toggle Filter Mode (Gaussian <-> PRaT),
                            // 2 = Toggle Voicing EQ (Tilt + BBC Dip), 3 = Toggle Master Bypass (Active <-> Pure)
} BenchRTConfig;

typedef struct {
    size_t out_frames_written;
    size_t ring_underruns;
    double peak_left_dbfs;
    double peak_right_dbfs;
} BenchRTStats;

// Bit-exact Float64 passthrough reference
NAIT_BENCH_API void bench_passthrough_f64(const double* in_stereo, size_t frames, double* out_stereo);

// 64-bit precision bench pipeline exercising AnchiFilterBank, r8brain, Biquad64, Triode, Limiter & TPDF/LNS
NAIT_BENCH_API size_t bench_process_core_f64(
    const BenchCoreConfig* cfg,
    const double* in_stereo,
    size_t in_frames,
    double* out_stereo,
    size_t max_out_frames,
    size_t* out_limiter_engaged_frames
);

// Production DSPProcessor real-time dual-callback SPSC RingBuffer + renderOutput harness (512 & 64-frame stress)
NAIT_BENCH_API int bench_stream_realtime_dsp(
    const BenchRTConfig* cfg,
    const float* in_stereo_f32,
    size_t in_frames,
    float* out_stereo_f32,
    size_t max_out_frames,
    BenchRTStats* out_stats
);

// Production DSPProcessor offline wrapper
NAIT_BENCH_API size_t bench_process_production_offline(
    const BenchRTConfig* cfg,
    const float* in_stereo_f32,
    size_t in_frames,
    float* out_stereo_f32,
    size_t max_out_frames
);

#ifdef __cplusplus
}
#endif

#endif // NAIT_DSP_BENCH_H
