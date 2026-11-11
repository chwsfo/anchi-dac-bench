"""Interactive 4-step listening lab, local macOS CoreAudio WAV player, ABX runner, and notes logger."""

from __future__ import annotations

from datetime import datetime
import random
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from .bridge import NaitDSPBenchBridge
from .days import MetricCheck
from .synthesizer import (
    ReferenceExcerpt,
    discover_local_reference_library,
    generate_day1_audition_clips,
    generate_day2_audition_clips,
    generate_day3_audition_clips,
    generate_day4_audition_clips,
)

DAY_METADATA: dict[int, dict[str, Any]] = {
    1: {
        "title": "Day 1 — Frequency Response, Passband Ripple & Ultrasonic Image Rejection",
        "briefing": (
            "When 44.1 kHz audio is upsampled 4x to 176.4 kHz, zero-stuffing creates ultrasonic spectral\n"
            "mirrors above 22.05 kHz (e.g., a 19.1 kHz cymbal harmonic mirrors to 25.0 kHz and 63.2 kHz).\n"
            "Your ears can't hear 25 kHz directly, but analog DAC/amp stages can: ultrasonic images hit\n"
            "slew limits and intermodulate back DOWN into the midrange as glare. Anchi DAC's 64-bit\n"
            "Craven Gaussian-Apodizing filter erases mirrors (>150 dB) with zero equiripple passband waves\n"
            "and a gentle -0.60 dB @ 20 kHz rolloff that strips baked-in Red Book studio ADC pre-echo."
        ),
        "plot_look_for": [
            "Top panel (±0.05 dB zoom): contrast Anchi's dead-flat line against the gold 1990s 8x CD equiripple waves (±0.028 dB) and the red Pure NOS curve plunging off the chart by 2.6 kHz (-3.22 dB @ 20 kHz!).",
            "Bottom panel (0–96 kHz): Pure NOS leaks a massive -3.3 dBFS mirror at 25 kHz; 1990s/embedded filters sit at -62 dBFS; Anchi crushes mirrors below -184 dBFS (120+ dB cleaner!).",
        ],
        "questions": [
            "On acoustic cymbals and close-miked vocals/violin, did you hear upper-midrange 'hash' vanish when switching from leaky ZOH to Anchi's 150 dB Gaussian filter?",
            "Do you prefer dead-flat extension to 20 kHz (48k native behavior) or Anchi's gentle -0.60 dB Craven silk roll-off at 20 kHz on Red Book 44.1k? What should Anchi DAC's default voicing story be?",
        ],
    },
    2: {
        "title": "Day 2 — THD+N, Harmonic Structure & 64-Bit Linearity",
        "briefing": (
            "Single THD+N numbers lie—harmonic order determines musicality! -60 dB of pure 2nd harmonic\n"
            "(H2, one octave up, like a 12AU7 triode) sounds lush and full-bodied, whereas -90 dB of odd\n"
            "harmonics (H3/H7) or undithered digital truncation sounds metallic and fatiguing.\n"
            "Because Anchi DAC's 64-bit double-precision C++17 core holds linearity below -140 dB THD+N\n"
            "across all 10 sample-rate pairs and 32-tone dense mixes, zero digital grit is ever added—\n"
            "letting your Naim NAIT 1's natural analog character (or intentional 4x oversampled 12AU7 stage) shine."
        ),
        "plot_look_for": [
            "Top panel (997 Hz @ -1 dBFS): compare the red Budget Embedded Motherboard Codec (~-79 dB THD+N, H2/H3 spikes) & gold 1990s 16-bit Multibit CD Player (~-93 dB) against Anchi's flat blue 64-bit->24-bit TPDF floor (-140.9 dB, 22+ dB cleaner than even elite -118 dB R-2R hardware!).",
            "Bottom panel (32-Tone Multitone, 44.1k→192k): cheap OS linear-interp SRC + embedded codec fills the valleys with red -85 dBFS IMD 'grass', whereas Anchi holds dead-black -194 dBFS valleys.",
        ],
        "questions": [
            "At what dB level (-50 dB, -70 dB, or -90 dB) could you reliably distinguish warm H2 body vs. edgy H3 vs. truncation grit on the piano chord decay?",
            "How does hearing pure H2 vs. truncation grit shape how you explain '64-bit transparency + intentional 4x tube voicing' in Anchi DAC's product copy?",
        ],
    },
    3: {
        "title": "Day 3 — Impulse Response, Phase Linearity & The 'PRaT' Debate",
        "briefing": (
            "By Kramers-Kronig / Fourier physics, no steep filter can have both zero phase shift AND\n"
            "zero pre-ringing. Gaussian Linear Phase preserves exact studio inter-aural phase cues (throwing\n"
            "a deep, holographic 3D soundstage) with symmetric Gabor-minimal pre/post envelope.\n"
            "PRaT Cepstral Minimum-Phase shifts 100% of filter energy AFTER t=0 (cutting wavefront arrival\n"
            "from 2.90 ms to 0.13 ms with zero pre-echo!) for immediate, front-foot rhythmic snap."
        ),
        "plot_look_for": [
            "Top-Left & Top-Right (±2 ms Impulse): gold 90s CD / Chip-DAC Sharp Brickwall rings past ±2 ms and red 1st-Gen CD Elliptic rings heavily after t0, whereas Anchi's Gaussian settles in ±0.35 ms (Linear) or zero pre-echo (PRaT).",
            "Bottom-Right (Group Delay): 1st-Gen CD 11th-order Elliptic filter surges +0.91 ms at 20 kHz (nearly 4x Anchi PRaT's gentle +0.25 ms rise), while Anchi Linear holds a dead-flat 2.66 ms.",
        ],
        "questions": [
            "In the 16x slowed impulse & blind 3x ABX shootouts, did your ear gravitate toward Linear Phase's spatial layering or PRaT Minimum Phase's transient immediacy—or did it depend on the track?",
            "How will you describe these two filter modes inside the Anchi DAC menu bar UI in 5 words or fewer?",
        ],
    },
    4: {
        "title": "Day 4 — Vintage Hardware Safety Pad, ISP Limiter & 24-Bit TPDF Dither",
        "briefing": (
            "Two silent killers ruin computer audio on classic hi-fi:\n"
            "1. Inter-Sample Peaks (ISPs) & Vintage Input Overload: 0.0 dBFS modern streams reconstruct\n"
            "   to +3.0 to +5.8 dBTP between samples, clipping DAC oversamplers and slamming 75mV–500mV\n"
            "   vintage inputs (like the Naim NAIT 1!). Anchi's Vintage Safety Pad restores pure linear\n"
            "   crest factor, backed by a zero-overshoot safety limiter.\n"
            "2. 24-Bit TPDF Dither: 2-LSB triangular noise eliminates quantization buzz and noise pumping,\n"
            "   resolving reverb tails 11.5 dB quieter than the 24th bit (-150 dBFS)!"
        ),
        "plot_look_for": [
            "Top panel (+3.01 dBTP ISP): red 90s CD / typical unpadded Chip-DAC oversampler clips hard at ±1.0, vs. green uncompressed Anchi Safety Pad (-4.5 dB).",
            "Bottom-Left & Bottom-Right (24-Bit TPDF vs. Undithered DAC): red undithered truncation shatters -136.5 dBFS into an odd-harmonic comb (and gates off to dead zero at -150 dBFS!), whereas Anchi resolves -150.00 dBFS cleanly with zero H3/H5 spurs.",
        ],
        "questions": [
            "Listening to the level-matched Loud Master Rescue and your own NAIT 1 / Enyo 15th chain, what pad/preset feels effortless on hot modern masters?",
            "After hearing the +70 dB reverb-tail microscope (Truncated crumble vs. RPDF pumping vs. TPDF silk), how does this clarify Anchi DAC's 'secret weapon' story for vintage hi-fi owners?",
        ],
    },
}


def print_checks_table(title: str, checks: list[MetricCheck]) -> None:
    print(f"\n{'=' * 96}")
    print(f"  {title}")
    print(f"{'=' * 96}")
    for c in checks:
        badge = "PASS" if c.passed else "FAIL"
        icon = "✅" if c.passed else "❌"
        print(f"  {icon} [{badge}] {c.name}")
        print(f"       Condition : {c.condition}")
        print(f"       Measured  : {c.measured}   (Target: {c.threshold})")
        print(f"       Meaning   : {c.diagnostic_note}")
    print(f"{'=' * 96}\n")


class AuditionPlayer:
    """Plays level-calibrated WAV clips locally via macOS afplay (capped <= -6.11 dBFS)."""

    def __init__(self, target: str = "local") -> None:
        self.target = target

    def play_wav(self, wav_path: Path, label: str = "") -> None:
        disp = label or wav_path.name
        print(f"    🔊 Playing [{disp}] on local Mac default output ... (Ctrl+C to stop early)")
        try:
            subprocess.run(["/usr/bin/afplay", str(wav_path)], check=False)
        except KeyboardInterrupt:
            print("    ⏹  Stopped playback.")


def _prompt_line(prompt_text: str) -> str:
    try:
        return input(prompt_text).strip()
    except EOFError:
        return ""


def _clean_dragged_path(raw: str) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    try:
        tokens = shlex.split(cleaned)
        if tokens:
            cleaned = tokens[0]
    except ValueError:
        cleaned = cleaned.strip("'\"")
    p = Path(cleaned).expanduser()
    return str(p) if p.exists() else None


def _interactive_ab_pair(
    player: AuditionPlayer,
    title: str,
    path_a: Path,
    label_a: str,
    path_b: Path,
    label_b: str,
    on_switch_track: Callable[[], tuple[Path, Path]] | None = None,
) -> None:
    print(f"\n  🎧 Sighted A/B Audition: {title}")
    print(f"     [a] Play A: {label_a}")
    print(f"     [b] Play B: {label_b}")
    if on_switch_track is not None:
        print("     [t] Switch reference track")
    print("     [Enter / c] Continue to next step")
    prompt_lbl = "     Choice [a / b / t / Enter]: " if on_switch_track is not None else "     Choice [a / b / Enter]: "
    while True:
        choice = _prompt_line(prompt_lbl).lower()
        if choice in ("", "c", "n", "q"):
            break
        if choice == "t" and on_switch_track is not None:
            path_a, path_b = on_switch_track()
            print("     ✅ Switched track! Ready to play [a] or [b].")
        elif choice == "a":
            player.play_wav(path_a, f"A — {label_a}")
        elif choice == "b":
            player.play_wav(path_b, f"B — {label_b}")


def _interactive_abx_shootout(
    player: AuditionPlayer,
    title: str,
    path_a: Path,
    name_a: str,
    path_b: Path,
    name_b: str,
    trials: int = 3,
    on_switch_track: Callable[[], tuple[Path, Path]] | None = None,
) -> str:
    print(f"\n  🙈 Blind {trials}x ABX Shootout: {title}")
    print(f"     Reference A = {name_a}")
    print(f"     Reference B = {name_b}")
    t_hint = " | [t] Switch track" if on_switch_track is not None else ""
    print(
        f"     Commands: [a] Play A | [b] Play B | [x] Play Mystery X | [1] Guess X=A | [2] Guess X=B{t_hint} | [s] Skip shootout"
    )

    rng = random.SystemRandom()
    correct = 0
    completed = 0

    for t_idx in range(1, trials + 1):
        actual = rng.choice(["A", "B"])
        path_x = path_a if actual == "A" else path_b
        print(f"\n     --- Trial {t_idx} of {trials} (Mystery X locked in) ---")
        prompt_str = (
            f"     Trial {t_idx} [a/b/x, 1(=A)/2(=B), t(track), s(skip)]: "
            if on_switch_track is not None
            else f"     Trial {t_idx} [a/b/x or guess 1(=A) / 2(=B) / s(skip)]: "
        )
        while True:
            cmd = _prompt_line(prompt_str).lower()
            if cmd == "s":
                return f"{correct}/{completed} (skipped after {completed} trials)"
            if cmd == "t" and on_switch_track is not None:
                path_a, path_b = on_switch_track()
                path_x = path_a if actual == "A" else path_b
                print(f"     ✅ Switched track (Trial {t_idx} Mystery X updated seamlessly)!")
            elif cmd == "a":
                player.play_wav(path_a, f"Reference A ({name_a})")
            elif cmd == "b":
                player.play_wav(path_b, f"Reference B ({name_b})")
            elif cmd == "x":
                player.play_wav(path_x, f"Mystery X (Trial {t_idx})")
            elif cmd in ("1", "2"):
                guess = "A" if cmd == "1" else "B"
                completed += 1
                if guess == actual:
                    correct += 1
                    print(f"     ✅ Spot on! Mystery X was {actual} ({name_a if actual == 'A' else name_b}).")
                else:
                    print(f"     🔎 Close! Mystery X was {actual} ({name_a if actual == 'A' else name_b}).")
                break
            elif cmd == "":
                continue

    summary = f"{correct}/{completed} correct on {title}"
    print(f"     🏆 ABX Score: {summary}")
    return summary


def open_plot_on_macos(plot_path: Path) -> None:
    if plot_path.exists():
        subprocess.run(["/usr/bin/open", str(plot_path)], check=False)


def ensure_taste_journal_initialized(journal_path: Path) -> None:
    if journal_path.exists():
        return
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Anchi DAC Bench — Interactive Listening & ABX Notes\n\n"
        "> Captured during the 4-Day Interactive Loopback Measurement & Listening Lab\n"
        "> (`python3 scripts/audio_loopback_test.py --day <1-4> --interactive`).\n\n"
        "---\n"
    )
    journal_path.write_text(header, encoding="utf-8")


def append_journal_entry(
    journal_path: Path,
    day: int,
    audition_target: str,
    checks: list[MetricCheck],
    abx_notes: list[str],
    qa_pairs: list[tuple[str, str]],
) -> None:
    ensure_taste_journal_initialized(journal_path)
    meta = DAY_METADATA[day]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"\n## {meta['title']} ({stamp})\n",
        f"- **Audition Chain:** `{audition_target}`\n",
        f"- **Objective Verdict:** `{sum(1 for c in checks if c.passed)}/{len(checks)} AES17 checks PASSED`\n",
    ]
    if abx_notes:
        for note in abx_notes:
            lines.append(f"- **ABX Result:** {note}\n")

    lines.append("\n### Objective Bench Snapshot\n")
    lines.append("| Metric | Condition | Measured | Threshold | Status |\n")
    lines.append("| :--- | :--- | :--- | :--- | :---: |\n")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"| **{c.name}** | {c.condition} | `{c.measured}` | `{c.threshold}` | **{status}** |\n")

    lines.append("\n### Founder Listening Impressions & Product POV\n")
    for q, ans in qa_pairs:
        recorded = ans if ans else "_(Skipped / recorded offline)_"
        lines.append(f"- **Q:** *{q}*\n  - **POV:** {recorded}\n")

    lines.append("\n---\n")
    with journal_path.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def _prompt_reference_preset(
    heading: str,
    default_desc: str,
    preset_keys: list[str],
    library: dict[str, ReferenceExcerpt],
    synth_desc: str,
) -> str | ReferenceExcerpt | None:
    """1-keystroke selector over local EBU SQAM / Open Goldberg masters, synth stem, or custom path."""
    available = [library[k] for k in preset_keys if k in library]
    if not available:
        raw = _prompt_line(f"  {heading}: Drag/paste a .flac/.wav track or [Enter] for {synth_desc}: ")
        return _clean_dragged_path(raw)

    print(f"  🎵 {heading}:")
    print(f"     [Enter] Curated Default -> {default_desc}")
    key_map: dict[str, ReferenceExcerpt] = {}
    for idx, item in enumerate(available, start=1):
        key_map[str(idx)] = item
        print(f"     [{idx}]     {item.label}")
    print(f"     [s]     {synth_desc} (synthesized test stem)")
    raw = _prompt_line("     Select [Enter / 1..N / s] or drag/paste any .flac/.wav path: ").strip()
    if not raw:
        return None
    low = raw.lower()
    if low == "s":
        return "__synth__"
    if low in key_map:
        return key_map[low]
    return _clean_dragged_path(raw)


def run_interactive_day(
    day: int,
    bridge: NaitDSPBenchBridge,
    checks: list[MetricCheck],
    details: dict[str, Any],
    clips_dir: Path,
    journal_path: Path,
    audition_target: str = "local",
    custom_track: str | None = None,
    open_plots: bool = True,
    skip_to_step3: bool = False,
) -> None:
    meta = DAY_METADATA[day]
    player = AuditionPlayer(target=audition_target)
    lib = discover_local_reference_library()
    repo_root = Path(__file__).resolve().parents[2]

    if not skip_to_step3:
        # STEP 1: BRIEFING
        print(f"\n{'#' * 96}")
        print(f"  STEP 1 OF 4 — BRIEFING: {meta['title']}")
        print(f"{'#' * 96}")
        for line in meta["briefing"].splitlines():
            print(f"  {line}")
        _prompt_line("\n  Press [Enter] to view Step 2 (Bench Measurements & Annotated Plot)... ")

        # STEP 2: MEASURE & VISUALIZE
        print(f"\n{'#' * 96}")
        print("  STEP 2 OF 4 — MEASURE & VISUALIZE")
        print(f"{'#' * 96}")
        print_checks_table(meta["title"], checks)
        raw_plot = Path(details["plot"])
        plot_path = raw_plot if raw_plot.is_absolute() else (repo_root / raw_plot)
        print(f"  📈 Annotated Graph Saved: {details['plot']}")
        print("  🔍 Where to look on the graph:")
        for bullet in meta["plot_look_for"]:
            print(f"     • {bullet}")
        if open_plots:
            open_plot_on_macos(plot_path)

        _prompt_line("\n  Press [Enter] to generate & launch Step 3 (Audition & Exaggerate Ear Lab)... ")

    # STEP 3: AUDITION & EXAGGERATE
    print(f"\n{'#' * 96}")
    print("  STEP 3 OF 4 — AUDITION & EXAGGERATE (SAFE LEVEL-CALIBRATED <= -6.1 dBFS)")
    print(f"{'#' * 96}")
    abx_notes: list[str] = []

    if day == 1:
        d1_track: str | ReferenceExcerpt | None = custom_track
        if not d1_track:
            d1_track = _prompt_reference_preset(
                heading="Day 1 Reference Masters (Local EBU SQAM + Open Goldberg)",
                default_desc="EBU #59 Ravel Violin (NOS Aliasing) + EBU #35 Glockenspiel (Top-Octave 88.2 kHz)",
                preset_keys=["sqam_violin_ravel", "sqam_glockenspiel", "sqam_triangle", "sqam_cymbal", "goldberg_aria"],
                library=lib,
                synth_desc="Built-in Bowed Violin & Jazz Wire Brushes",
            )
        clips = generate_day1_audition_clips(bridge, clips_dir, d1_track)
        print(f"  Generated {len(clips)} ear-training WAVs in {clips_dir}")
        ans = _prompt_line("  1) Play Passthrough Null Residual boosted +80 dB (hear pure bit-exact silence)? [Y/n]: ").lower()
        if ans in ("", "y", "yes"):
            player.play_wav(clips["null_residual"], "Passthrough Null Residual (+80 dB)")
        _interactive_ab_pair(
            player,
            "Acoustic Air & Transients — Leaky NOS Ultrasonic Aliasing vs. Anchi 150 dB Gaussian",
            clips["jazz_leaky"],
            "Leaky ZOH Upsampler (ultrasonic mirrors intermodulating into upper-midrange hash)",
            clips["jazz_anchi"],
            "Anchi 64-Bit Craven Gaussian Upsampler (>150 dB mirror rejection)",
        )
        _interactive_ab_pair(
            player,
            "19 kHz + 20 kHz Twin-Tone IMD Probe (keep volume moderate!)",
            clips["twintone_leaky"],
            "Leaky ZOH (audible 1 kHz / 4.1 kHz analog intermodulation whistle)",
            clips["twintone_anchi"],
            "Anchi 150 dB Gaussian (ultrasonic mirrors erased -> clean silence)",
        )
        _interactive_ab_pair(
            player,
            "Top-Octave Voicing (88.2 kHz) — Equiripple Brickwall vs. Anchi Craven Gaussian Silk (-0.60 dB @ 20 kHz)",
            clips["top_brickwall"],
            "Parks-McClellan Equiripple Brickwall (±0.25 dB passband waves + sharp 21.8 kHz corner)",
            clips["top_craven_silk"],
            "Anchi Pedestal-Zeroed Craven Gaussian (0.000 dB flat to 19k, -0.60 dB @ 20k silk)",
        )

    elif day == 2:
        d2_track: str | ReferenceExcerpt | None = custom_track
        if not d2_track:
            d2_track = _prompt_reference_preset(
                heading="Day 2 Harmonic Voicing Master (Silent-Booth EBU SQAM + 24/96 Open Goldberg)",
                default_desc="EBU SQAM #39 — Grand Piano Chord & Silent-Booth Decay (zero mic hiss!)",
                preset_keys=["sqam_piano", "sqam_harpsichord", "sqam_guitar", "goldberg_aria", "goldberg_var13"],
                library=lib,
                synth_desc="Built-in Anechoic Steinway Chord + 5.5s Decay",
            )
        while True:
            clips = generate_day2_audition_clips(clips_dir, d2_track)
            print(f"  Generated {len(clips)} harmonic & truncation ear-training WAVs in {clips_dir}")
            print("  🎹 Interactive Harmonic & Truncation Menu:")
            menu = {
                "1": ("Clean 64-Bit Reference", clips["piano_clean"]),
                "2": ("Pure H2 Warmth @ -50 dB (lush 12AU7 octave body)", clips["piano_h2_50"]),
                "3": ("Pure H2 Warmth @ -70 dB (subtle tube bloom)", clips["piano_h2_70"]),
                "4": ("Pure H2 Warmth @ -90 dB (threshold of audibility)", clips["piano_h2_90"]),
                "5": ("Odd H3/H7 @ -50 dB (edgy / metallic)", clips["piano_h3_50"]),
                "6": ("Odd H3/H7 @ -70 dB (subtle glassy etch)", clips["piano_h3_70"]),
                "7": ("Odd H3/H7 @ -90 dB", clips["piano_h3_90"]),
                "8": ("Undithered Bit-Truncation Grit (buzzy decay crumble)", clips["piano_truncation"]),
            }
            for k, (lbl, _) in menu.items():
                print(f"     [{k}] {lbl}")
            print("     [t] Switch reference track (try EBU Harpsichord, Guitar, Goldberg, or Synth)")
            print("     [Enter / c] Continue to Journal")
            switch_requested = False
            while True:
                pick = _prompt_line("     Pick clip [1-8 / t / Enter]: ").lower()
                if pick in ("", "c", "q"):
                    break
                if pick == "t":
                    d2_track = _prompt_reference_preset(
                        heading="Switch Day 2 Reference Master",
                        default_desc="EBU SQAM #39 — Grand Piano Chord & Silent-Booth Decay",
                        preset_keys=["sqam_piano", "sqam_harpsichord", "sqam_guitar", "goldberg_aria", "goldberg_var13"],
                        library=lib,
                        synth_desc="Built-in Anechoic Steinway Chord + 5.5s Decay",
                    )
                    switch_requested = True
                    break
                if pick in menu:
                    player.play_wav(menu[pick][1], menu[pick][0])
            if not switch_requested:
                break

    elif day == 3:
        perc_custom: str | ReferenceExcerpt | None = custom_track
        space_custom: str | ReferenceExcerpt | None = None
        if not perc_custom:
            perc_custom = _prompt_reference_preset(
                heading="Day 3 Blind ABX Track A — Percussive / PRaT Transient Snap",
                default_desc="EBU SQAM #27 — Castanets Flamenco Rhythm (classic pre-ringing test)",
                preset_keys=["sqam_castanets", "sqam_snare", "sqam_guitar", "sqam_harpsichord", "goldberg_var29"],
                library=lib,
                synth_desc="Built-in Jazz Rimshot & Plucked Upright Bass",
            )
            space_custom = _prompt_reference_preset(
                heading="Day 3 Blind ABX Track B — Spacious 3D Acoustic Soundstage",
                default_desc="EBU SQAM #64 — Orff Choir in Concert Hall (3D hall depth & layering)",
                preset_keys=["sqam_choir_orff", "sqam_orchestra_strauss", "goldberg_aria", "sqam_violin_ravel"],
                library=lib,
                synth_desc="Built-in 3D Chamber Ensemble",
            )

        clips = generate_day3_audition_clips(bridge, clips_dir, perc_custom, space_custom)
        _interactive_ab_pair(
            player,
            "16x Time-Stretched Dirac Impulse Microscope (20.4 kHz ring shifted to 1,277 Hz!)",
            clips["impulse_16x_linear"],
            "Gaussian Linear Phase ('zwoop-CLICK-poowz' symmetric pre/post envelope)",
            clips["impulse_16x_min_phase"],
            "PRaT Cepstral Minimum Phase ('CLICK-poowz' zero-pre-echo wavefront)",
        )

        def _switch_day3_perc() -> tuple[Path, Path]:
            nonlocal perc_custom, clips
            perc_custom = _prompt_reference_preset(
                heading="Switch Day 3 Track A (Percussive / PRaT)",
                default_desc="EBU SQAM #27 — Castanets Flamenco Rhythm",
                preset_keys=["sqam_castanets", "sqam_snare", "sqam_guitar", "sqam_harpsichord", "goldberg_var29"],
                library=lib,
                synth_desc="Built-in Jazz Rimshot & Plucked Upright Bass",
            )
            clips = generate_day3_audition_clips(bridge, clips_dir, perc_custom, space_custom)
            return clips["abx_perc_linear"], clips["abx_perc_min"]

        def _switch_day3_space() -> tuple[Path, Path]:
            nonlocal space_custom, clips
            space_custom = _prompt_reference_preset(
                heading="Switch Day 3 Track B (Spacious 3D Acoustic)",
                default_desc="EBU SQAM #64 — Orff Choir in Concert Hall",
                preset_keys=["sqam_choir_orff", "sqam_orchestra_strauss", "goldberg_aria", "sqam_violin_ravel"],
                library=lib,
                synth_desc="Built-in 3D Chamber Ensemble",
            )
            clips = generate_day3_audition_clips(bridge, clips_dir, perc_custom, space_custom)
            return clips["abx_space_linear"], clips["abx_space_min"]

        abx_notes.append(
            _interactive_abx_shootout(
                player,
                "Track A — Percussive / PRaT Transient Groove",
                clips["abx_perc_linear"],
                "Gaussian Linear Phase",
                clips["abx_perc_min"],
                "PRaT Minimum Phase",
                trials=3,
                on_switch_track=_switch_day3_perc,
            )
        )
        abx_notes.append(
            _interactive_abx_shootout(
                player,
                "Track B — Spacious 3D Acoustic Soundstage",
                clips["abx_space_linear"],
                "Gaussian Linear Phase",
                clips["abx_space_min"],
                "PRaT Minimum Phase",
                trials=3,
                on_switch_track=_switch_day3_space,
            )
        )

    elif day == 4:
        hot_custom: str | ReferenceExcerpt | None = custom_track
        tail_custom: str | ReferenceExcerpt | None = None
        if not hot_custom:
            hot_custom = _prompt_reference_preset(
                heading="Day 4 Exercise 1 — Hot-Mastered Inter-Sample Peak (ISP) Rescue",
                default_desc="EBU SQAM #69 — ABBA Stereo Pop Chorus (hot-mastered to 0.00 dBFS w/ >+1.4 dBTP ISPs)",
                preset_keys=["sqam_abba_pop", "sqam_orchestra_strauss", "goldberg_var29", "sqam_cymbal"],
                library=lib,
                synth_desc="Extreme +3.3 dBTP Synthetic Torture Stem",
            )
        clips = generate_day4_audition_clips(bridge, clips_dir, hot_custom, tail_custom)

        def _switch_day4_hot() -> tuple[Path, Path]:
            nonlocal hot_custom, clips
            hot_custom = _prompt_reference_preset(
                heading="Switch Day 4 Hot-Mastered ISP Track",
                default_desc="EBU SQAM #69 — ABBA Stereo Pop Chorus",
                preset_keys=["sqam_abba_pop", "sqam_orchestra_strauss", "goldberg_var29", "sqam_cymbal"],
                library=lib,
                synth_desc="Extreme +3.3 dBTP Synthetic Torture Stem",
            )
            clips = generate_day4_audition_clips(bridge, clips_dir, hot_custom, tail_custom)
            return clips["loud_master_clipped"], clips["loud_master_rescued"]

        _interactive_ab_pair(
            player,
            "Exercise 1 — The Loud Master Rescue (Strictly RMS Level-Matched to 0.00 dB!)",
            clips["loud_master_clipped"],
            "Unprotected 0 dBFS Oversampler + Overloaded Vintage Preamp Input",
            clips["loud_master_rescued"],
            "Anchi Vintage Safety Pad (-4.5 dB): 0% ISP Clipping, Open Vocal & Drum Dynamics",
            on_switch_track=_switch_day4_hot,
        )
        while True:
            print("\n  🔬 Exercise 2 — The +70 dB Reverb-Tail Microscope:")
            print("     [1] Truncated (no dither -> buzzy harmonic fuzz & hard digital gate-off)")
            print("     [2] 1-Bit RPDF Dither (distortion gone, but noise floor pumps with the sustain)")
            print("     [3] Anchi 2-LSB TPDF Dither (pure decay melts into steady tape-like air)")
            print("     [t] Switch microscope decay source (EBU Piano, Guitar, Goldberg #31, or Synth)")
            print("     [Enter / c] Continue to Journal")
            switch_tail = False
            while True:
                pick = _prompt_line("     Pick microscope clip [1 / 2 / 3 / t / Enter]: ").lower()
                if pick in ("", "c", "q"):
                    break
                if pick == "t":
                    tail_custom = _prompt_reference_preset(
                        heading="Switch +70 dB Reverb-Tail Microscope Source",
                        default_desc="EBU SQAM #39 — Silent-Booth Steinway Chord Decay",
                        preset_keys=["sqam_piano", "sqam_guitar", "sqam_harpsichord", "goldberg_final_decay"],
                        library=lib,
                        synth_desc="-90 dBFS Pure Acoustic Guitar Harmonic Fade",
                    )
                    clips = generate_day4_audition_clips(bridge, clips_dir, hot_custom, tail_custom)
                    switch_tail = True
                    break
                if pick == "1":
                    player.play_wav(clips["tail_truncated"], "Truncated (No Dither, +70 dB)")
                elif pick == "2":
                    player.play_wav(clips["tail_rpdf"], "1-Bit RPDF (Noise Pumping, +70 dB)")
                elif pick == "3":
                    player.play_wav(clips["tail_tpdf"], "Anchi TPDF (Silky Steady Air, +70 dB)")
            if not switch_tail:
                break

    # STEP 4: LOG YOUR LISTENING NOTES
    print(f"\n{'#' * 96}")
    print(f"  STEP 4 OF 4 — LOG LISTENING NOTES -> {journal_path.name}")
    print(f"{'#' * 96}")
    qa_pairs: list[tuple[str, str]] = []
    for idx, q in enumerate(meta["questions"], start=1):
        print(f"\n  Q{idx}: {q}")
        ans = _prompt_line("  Your notes (or [Enter] to skip): ")
        qa_pairs.append((q, ans))

    append_journal_entry(journal_path, day, "Local Mac CoreAudio Output", checks, abx_notes, qa_pairs)
    print(f"\n  ✍️  Saved session metrics & listening notes to {journal_path}\n")
