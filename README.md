# Continuous Psychophysics Memory Tracker

A Corsi-block sequence-recall task extended with **continuous mouse-trajectory analysis**. The classic paradigm scores only whether you clicked the right circles in the right order. This version keeps that discrete score but also records the full cursor path between clicks and analyses it using cross-correlation — the same signal-processing approach used in modern continuous psychophysics. The result is a richer picture of how memory actually guides movement.

---

## Why Two Scores?

### The Problem with Discrete Scoring

Traditional memory tasks reduce behaviour to a single number: how many items were recalled correctly. This is fast and interpretable, but it discards everything about *how* the response was executed. Two people can score identically on a Corsi task while using completely different cognitive strategies underneath.

### What the Continuous Signal Reveals

Continuous psychophysics records the full response trajectory and compares it against an ideal signal over time. This recovers temporal structure — response lag, smoothness, and deviation — that a binary score is blind to. Burge & Bonnen (2025) show that continuous tracking measures are tightly correlated with traditional forced-choice estimates, while also capturing dynamics that forced-choice cannot.

---

## Cognitive Science Motivation

### People Encode Spatial Sequences Differently

Research in spatial cognition identifies three levels of environmental knowledge: **landmark** (discrete objects in isolation), **route** (a sequential path through decision points), and **survey** (a metric, map-like layout of the whole space). These are not stages everyone passes through — they reflect stable individual differences in how the same information gets encoded (Siegel & White, 1975; Thorndyke & Hayes-Roth, 1982). A participant using survey knowledge will pre-compute the path before moving; a landmark encoder treats each target as an isolated object.

### Spatial vs Object Visualizers

Kozhevnikov et al. (2005) showed that people who prefer visual thinking divide into two groups: **object visualizers**, who encode scenes holistically as a single perceptual unit, and **spatial visualizers**, who analyse them part-by-part using spatial relations. These are not ability differences — they are stable strategy preferences that persist even when they are not the most efficient approach. Both groups can recall the correct sequence, but their cursor trajectories between clicks will look structurally different.

### Fine-Grained vs Categorical Spatial Memory

An fMRI meta-analysis (n = 153) found that people differ in whether they maintain **fine-grained** (precise, continuous) or **categorical** (chunked, approximate) spatial representations in working memory, and that these differences are neurally dissociable — engaging different brain networks (Johansson et al., 2023). Fine-grained encoders are expected to produce smoother trajectories with lower peak lag; categorical encoders produce noisier paths with broader cross-correlogram widths. A binary score cannot distinguish them. The continuous scorer can.

### Why This Justifies a Continuous Scorer

If cognition were uniform, the discrete score would be sufficient. But because people differ in whether they encode a sequence as discrete landmarks or as a continuous spatial path, the mouse trajectory between clicks carries diagnostic information that the click outcome does not. This experiment is designed to make both visible simultaneously.

---

## How It Works

### Task Overview

Thirty circles are scattered across the screen. Five light up in sequence — watch, memorise, then click them back in order. Your mouse path throughout the recall phase is recorded at 60 Hz. After each trial, a three-panel results figure is generated comparing your discrete and continuous performance.

### Trial Flow

`Watch sequence light up` → `"Your turn!" pause` → `Click circles in order (path recorded)` → `Results figure` → `SPACE to continue`

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start / advance to next trial |
| `Q` | Quit at any time |
| `ESC` | End recall phase early |

---

## Analysis Pipeline

### Sequence Score (Discrete)

The fraction of clicks whose target matches the correct circle at that ordinal position. This is the standard Corsi scoring method — a number between 0 and 1 based purely on what you clicked, not how you got there.

### Path Score (Continuous)

The cursor path between each pair of consecutive correct circles is compared to the optimal straight-line segment. Mean perpendicular deviation is computed per segment, normalised by segment length, and subtracted from 1. A score of 1.0 means you moved in a perfectly straight line between every target.

### Cross-Correlogram

The ideal trajectory is constructed as a piecewise-linear path through the correct circles, uniformly traversed in time. Both the actual and ideal trajectories are resampled onto a common time grid, and a normalised Pearson cross-correlation is computed per axis (X and Y) over a ±2 s lag window. This follows the formulation in Iriventi (2025, pp. 20–22) and Box 1 of Burge & Bonnen (2025).

### Extracted Metrics

Three metrics are pulled from each cross-correlogram: `ρ_max` (peak correlation — how well the paths match), `τ_peak` (the lag at peak — how delayed the response is relative to the ideal), and `FWHM` (full-width at half-maximum — how temporally precise the tracking is). These three values together characterise the quality, timing, and sharpness of the continuous motor response.

---

## Output: Three-Panel Results Figure

### Panel 1 — Mouse Trajectory

Your actual cursor path is colour-coded by per-point deviation from the optimal route (green = on-track, red = off-track). The correct sequence circles are shown with their ordinal numbers, and the ideal dashed path connects them. This panel makes the spatial encoding strategy immediately visible.

### Panel 2 — Discrete vs Continuous Score

A side-by-side bar chart comparing the sequence score (blue) and path score (orange). Participants who recall correctly but move inefficiently will show a gap between the two bars — this gap is the signal that the discrete score alone would miss.

### Panel 3 — Cross-Correlogram

Normalised correlation curves for the X and Y axes, with `ρ_max`, `τ_peak`, and `FWHM` annotated for each. A sharp, high, near-zero-lag peak indicates fluent spatial encoding. A broad, low, or delayed peak indicates categorical or landmark-based encoding — even if the sequence score is perfect.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/continuous-psychophysics-memory-tracker.git
cd continuous-psychophysics-memory-tracker
pip install -r requirements.txt
python continuous_memory_test.py
```

---

## Configuration

All task parameters are constants at the top of `continuous_memory_test.py`.

| Constant | Default | Meaning |
|----------|---------|---------|
| `NUM_CIRCLES` | 30 | Total circles on screen |
| `SEQ_LEN` | 5 | Sequence length to memorise |
| `SHOW_DUR` | 0.80 s | How long each circle stays lit |
| `GAP_DUR` | 0.40 s | Dark gap between lit circles |
| `WAIT_DUR` | 1.20 s | Pause before recall begins |
| `WIDTH`, `HEIGHT` | 1200 × 800 | Window size in pixels |
| `FPS` | 60 | Trajectory sampling rate |

---

## Project Structure

```
.
├── continuous_memory_test.py   # task, scoring, and visualisation
├── requirements.txt
├── README.md
├── LICENSE
└── docs/
    └── results_example.png     # sample output figure
```

---

## References

- Burge, J. & Bonnen, K. (2025). *Continuous psychophysics: past, present, future.* Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2025.01.005
- Bonnen, K. et al. (2015). *Continuous psychophysics: target-tracking to measure visual sensitivity.* Journal of Vision, 15(3), 14.
- Iriventi, S. K. (2025). *Modulation Frequency as a Cue for Auditory Motion Perception.* M.Sc. Thesis, Universität Ulm.
- Kozhevnikov, M., Kosslyn, S., & Shepard, J. (2005). *Spatial versus object visualizers: A new characterization of visual cognitive style.* Memory & Cognition, 33(4), 710–726.
- Siegel, A. W., & White, S. H. (1975). *The development of spatial representations of large-scale environments.* Advances in Child Development and Behavior, 10, 9–55.
- Johansson, M. et al. (2023). *Individual differences in spatial working memory strategies differentially reflected in the engagement of control and default brain networks.* PMC11364466.

---

## License

MIT — see [LICENSE](LICENSE).
