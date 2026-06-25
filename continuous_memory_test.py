#!/usr/bin/env python3
"""
Continuous Psychophysics Memory Tracker
========================================
Corsi-block-style sequence recall extended with continuous mouse-trajectory
analysis, demonstrating that the continuous signal reveals information that
a discrete right/wrong score cannot.

References:
  Burge & Bonnen (2025). Continuous psychophysics: past, present, future.
    Trends in Cognitive Sciences.
  Iriventi, S. K. (2025). Modulation Frequency as a Cue for Auditory Motion
    Perception. Master Thesis, Universität Ulm.

Controls
--------
  SPACE  — advance (start / next trial)
  Q      — quit at any time
"""

import sys
import random
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pygame
import matplotlib
matplotlib.use("Agg")                   # off-screen backend; avoids display conflicts
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ── Constants ─────────────────────────────────────────────────────────────────
WIDTH, HEIGHT  = 1200, 800
FPS            = 60
NUM_CIRCLES    = 30         # total circles spread across screen
SEQ_LEN        = 5          # how many light up (the sequence to memorise)
CIRCLE_R       = 22         # px
MIN_SEP        = 60         # px, minimum centre-to-centre gap
SHOW_DUR       = 0.80       # s — each circle stays lit during memorisation
GAP_DUR        = 0.40       # s — dark gap between consecutive circle lights
WAIT_DUR       = 1.20       # s — "Your turn!" pause before recall begins

# Palette  (R, G, B)
BG           = ( 28,  28,  38)
C_DIM        = ( 60,  60,  85)
C_LIT        = (255, 165,  50)
C_OK         = ( 90, 200, 110)
C_ERR        = (200,  70,  70)
C_TRAJ       = (110, 110, 155)
TXT          = (220, 220, 230)
ACCENT       = (255, 165,  50)

# Matplotlib palette
MPL_DARK  = "#1c1c26"
MPL_PANEL = "#28283a"


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Circle:
    idx: int
    x: float
    y: float
    state: str = "dim"   # "dim" | "lit" | "ok" | "err"

    def hit(self, px: float, py: float) -> bool:
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= CIRCLE_R ** 2

    def colour(self):
        return {"dim": C_DIM, "lit": C_LIT, "ok": C_OK, "err": C_ERR}[self.state]


@dataclass
class Trial:
    sequence: List[int]                                              # correct order
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, t)
    clicks: List[Tuple[int, float, float, float]] = field(default_factory=list) # (cid, x, y, t)


# ── Scorer ────────────────────────────────────────────────────────────────────
class Scorer:
    """All scoring and signal-processing logic."""

    # ── internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _seg_dist(p, a, b) -> float:
        """Perpendicular distance from 2-D point p to segment a–b."""
        p = np.asarray(p, float)
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ab  = b - a
        sq  = float(ab @ ab)
        if sq < 1e-10:
            return float(np.linalg.norm(p - a))
        t = float(np.clip((p - a) @ ab / sq, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    @staticmethod
    def _pearson_xcorr(x: np.ndarray, y: np.ndarray, dt: float):
        """
        Normalised Pearson cross-correlation following the thesis formula
        (pp. 20–22):  C_xy(τ) = Σ[(x(t)−x̄)(y(t+τ)−ȳ)] / √(Σ(x−x̄)² · Σ(y−ȳ)²)
        """
        x = x - x.mean()
        y = y - y.mean()
        sx, sy = x.std(), y.std()
        if sx < 1e-10 or sy < 1e-10:
            return None, None
        corr = np.correlate(x, y, mode="full") / (sx * sy * len(x))
        lags = np.arange(-(len(x) - 1), len(x)) * dt
        return lags, corr

    @staticmethod
    def _metrics(corr: np.ndarray, lags: np.ndarray) -> Tuple[float, float, float]:
        """Extract ρ_max, τ_peak, FWHM — same three metrics as in thesis results."""
        pk   = int(np.argmax(corr))
        rho  = float(corr[pk])
        tau  = float(lags[pk])
        half = rho * 0.5
        l, r = pk, pk
        while l > 0 and corr[l] >= half:
            l -= 1
        while r < len(corr) - 1 and corr[r] >= half:
            r += 1
        fwhm = float(lags[r] - lags[l])
        return rho, tau, fwhm

    # ── public API ────────────────────────────────────────────────────────────
    def sequence_score(self, trial: Trial) -> float:
        """Fraction of clicks in the correct sequential position (discrete)."""
        if not trial.sequence:
            return 0.0
        correct = sum(
            1 for i, (cid, *_) in enumerate(trial.clicks)
            if i < len(trial.sequence) and cid == trial.sequence[i]
        )
        return correct / len(trial.sequence)

    def path_score(self, trial: Trial, circles: List[Circle]) -> float:
        """
        Continuous score: mean perpendicular deviation of actual cursor path
        from the optimal straight-line segment between consecutive correct
        circles, normalised by segment length and subtracted from 1.
        """
        n_segs = min(len(trial.clicks) - 1, len(trial.sequence) - 1)
        if n_segs < 1 or len(trial.trajectory) < 2:
            return 0.0

        traj = np.array(trial.trajectory)     # (N, 3) — x, y, t
        all_devs = []

        for i in range(n_segs):
            t_a = trial.clicks[i][3]
            t_b = trial.clicks[i + 1][3]
            a   = (circles[trial.sequence[i]].x,     circles[trial.sequence[i]].y)
            b   = (circles[trial.sequence[i + 1]].x, circles[trial.sequence[i + 1]].y)
            d_opt = max(np.linalg.norm(np.array(b) - np.array(a)), 1.0)

            seg = traj[(traj[:, 2] >= t_a) & (traj[:, 2] <= t_b)]
            if len(seg) < 2:
                continue
            devs = [self._seg_dist((r[0], r[1]), a, b) / d_opt for r in seg]
            all_devs.append(float(np.mean(devs)))

        if not all_devs:
            return 0.0
        return float(np.clip(1.0 - np.mean(all_devs), 0.0, 1.0))

    def per_point_deviation(self, trial: Trial, circles: List[Circle]) -> np.ndarray:
        """Per-trajectory-point normalised deviation for trajectory heatmap."""
        if not trial.trajectory:
            return np.array([])
        traj = np.array(trial.trajectory)
        devs = np.zeros(len(traj))
        n_segs = min(len(trial.clicks) - 1, len(trial.sequence) - 1)

        for i in range(n_segs):
            t_a = trial.clicks[i][3]
            t_b = trial.clicks[i + 1][3]
            a   = (circles[trial.sequence[i]].x,     circles[trial.sequence[i]].y)
            b   = (circles[trial.sequence[i + 1]].x, circles[trial.sequence[i + 1]].y)
            d_opt = max(np.linalg.norm(np.array(b) - np.array(a)), 1.0)
            for k, row in enumerate(traj):
                if t_a <= row[2] <= t_b:
                    devs[k] = self._seg_dist((row[0], row[1]), a, b) / d_opt
        return devs

    def xcorr(
        self,
        trial: Trial,
        circles: List[Circle],
        lag_window: float = 2.0,
        n_pts: int = 600,
    ) -> Optional[dict]:
        """
        Build normalised cross-correlogram between actual and ideal trajectory
        (follows thesis methodology, pp. 20–22 and Burge & Bonnen 2025 Box 1).

        Ideal trajectory: piecewise linear through correct-sequence circles,
        uniformly traversed in time — the same construction used for the
        speaker position signal in the thesis.
        """
        if len(trial.trajectory) < 5 or len(trial.clicks) < 2:
            return None

        traj = np.array(trial.trajectory)          # (N, 3)
        t0, t1 = traj[0, 2], traj[-1, 2]
        total = t1 - t0
        if total < 0.2:
            return None

        # Ideal piecewise-linear path: parameterise by arc-length fraction
        pos = np.array(
            [(circles[i].x, circles[i].y) for i in trial.sequence], float
        )
        seg_lens  = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        arc_total = seg_lens.sum()
        if arc_total < 1:
            return None
        cum_frac = np.concatenate([[0.0], np.cumsum(seg_lens) / arc_total])

        t_grid = np.linspace(t0, t1, n_pts)
        s_norm = (t_grid - t0) / total              # 0 → 1

        ideal = np.zeros((n_pts, 2))
        for k, s in enumerate(s_norm):
            si = max(0, int(np.searchsorted(cum_frac, s, side="right")) - 1)
            si = min(si, len(pos) - 2)
            span = cum_frac[si + 1] - cum_frac[si]
            alpha = np.clip((s - cum_frac[si]) / span if span > 1e-10 else 0.0, 0.0, 1.0)
            ideal[k] = pos[si] + alpha * (pos[si + 1] - pos[si])

        actual = np.column_stack([
            np.interp(t_grid, traj[:, 2], traj[:, 0]),
            np.interp(t_grid, traj[:, 2], traj[:, 1]),
        ])

        dt = total / n_pts
        lags, cx = self._pearson_xcorr(actual[:, 0], ideal[:, 0], dt)
        if lags is None:
            return None
        _, cy = self._pearson_xcorr(actual[:, 1], ideal[:, 1], dt)
        if cy is None:
            return None

        mask = np.abs(lags) <= lag_window
        lm   = lags[mask]
        return {
            "lags":      lm,
            "corr_x":    cx[mask],
            "corr_y":    cy[mask],
            "metrics_x": self._metrics(cx[mask], lm),
            "metrics_y": self._metrics(cy[mask], lm),
        }


# ── Results figure ────────────────────────────────────────────────────────────
def _build_results_surface(
    trial: Trial, circles: List[Circle], scorer: Scorer
) -> pygame.Surface:
    """Render a 3-panel matplotlib figure into a pygame Surface (via PNG buffer)."""

    s_seq  = scorer.sequence_score(trial)
    s_path = scorer.path_score(trial, circles)
    xc     = scorer.xcorr(trial, circles)
    devs   = scorer.per_point_deviation(trial, circles)

    fig = plt.figure(figsize=(12, 8), facecolor=MPL_DARK)
    gs  = fig.add_gridspec(
        2, 2, hspace=0.52, wspace=0.33, left=0.07, right=0.97, top=0.92, bottom=0.09
    )
    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_bar   = fig.add_subplot(gs[0, 1])
    ax_xcorr = fig.add_subplot(gs[1, :])

    for ax in (ax_traj, ax_bar, ax_xcorr):
        ax.set_facecolor(MPL_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#44445a")
        ax.tick_params(colors="#aaaacc", labelsize=9)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#ddddee")

    # ── Panel 1: Trajectory overlay ───────────────────────────────────────────
    # pygame y-axis points down; flip for matplotlib's default (y up).
    def flip(x, y):
        return x, -y

    seq_set = set(trial.sequence)

    # Background (non-sequence) circles: tiny dots so they don't dominate
    for c in circles:
        if c.idx not in seq_set:
            mx, my = flip(c.x, c.y)
            ax_traj.add_patch(plt.Circle((mx, my), 7, color="#2e2e48", zorder=1))

    # Sequence circles: full-size, distinct colour
    for c in circles:
        if c.idx in seq_set:
            mx, my = flip(c.x, c.y)
            ax_traj.add_patch(plt.Circle((mx, my), CIRCLE_R, color="#383858", zorder=2))

    # Optimal dashed path through the correct sequence
    for i in range(len(trial.sequence) - 1):
        x0, y0 = flip(circles[trial.sequence[i]].x,     circles[trial.sequence[i]].y)
        x1, y1 = flip(circles[trial.sequence[i + 1]].x, circles[trial.sequence[i + 1]].y)
        ax_traj.plot([x0, x1], [y0, y1], "--", color="#777792",
                     lw=1.5, alpha=0.65, zorder=3)

    # Sequence order numbers on circles
    for rank, idx in enumerate(trial.sequence):
        mx, my = flip(circles[idx].x, circles[idx].y)
        ax_traj.text(mx, my, str(rank + 1), ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold", zorder=6)

    # Actual trajectory coloured by deviation (cool=low, warm=high)
    if len(trial.trajectory) > 1:
        pts = np.array([[x, -y] for x, y, _ in trial.trajectory])
        d   = np.clip(devs, 0.0, 1.0)
        segs   = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_d  = 0.5 * (d[:-1] + d[1:])
        lc = LineCollection(
            segs, cmap="RdYlGn_r",
            norm=plt.Normalize(vmin=0.0, vmax=0.55),
            linewidth=2.5, zorder=3,
        )
        lc.set_array(seg_d)
        ax_traj.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax_traj, fraction=0.035, pad=0.02)
        cb.set_label("Path deviation (norm.)", color="#aaaacc", fontsize=8)
        cb.ax.tick_params(labelcolor="#aaaacc", labelsize=8)

    # User click locations
    for _, cx_, cy_, _ in trial.clicks:
        mx, my = flip(cx_, cy_)
        ax_traj.scatter([mx], [my], s=65, c="white", zorder=6,
                        edgecolors="#888899", linewidths=1.2)

    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.autoscale_view()
    ax_traj.set_title("Mouse Trajectory", fontsize=11, pad=7)
    ax_traj.set_xlabel("X (px)", fontsize=9)
    ax_traj.set_ylabel("Y (px)", fontsize=9)

    # ── Panel 2: Score bar chart ──────────────────────────────────────────────
    bars = ax_bar.bar(
        ["Sequence\n(discrete)", "Path\n(continuous)"],
        [s_seq, s_path],
        color=["#5a8fd0", "#e07050"],
        width=0.46, edgecolor="#555570",
    )
    ax_bar.axhline(1.0, color="#aaaacc", linestyle="--", lw=1.0, alpha=0.5)
    ax_bar.set_ylim(0, 1.28)
    ax_bar.set_ylabel("Score  (0–1)", fontsize=9)
    ax_bar.set_title("Discrete vs Continuous Scoring", fontsize=11, pad=7)
    for bar, val in zip(bars, [s_seq, s_path]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.04,
            f"{val:.2f}",
            ha="center", va="bottom",
            color="white", fontsize=15, fontweight="bold",
        )

    # ── Panel 3: Cross-correlogram ────────────────────────────────────────────
    if xc is not None:
        lags = xc["lags"]
        ax_xcorr.plot(lags, xc["corr_x"], color="#5a8fd0", lw=2.0,
                      label="X (horizontal)")
        ax_xcorr.plot(lags, xc["corr_y"], color="#e09050", lw=2.0,
                      label="Y (vertical)")
        ax_xcorr.axvline(0, color="#666680", lw=1.0, linestyle="--", alpha=0.7)
        ax_xcorr.axhline(0, color="#404055", lw=0.8, linestyle=":")

        rho_x, tau_x, fwhm_x = xc["metrics_x"]
        rho_y, tau_y, fwhm_y = xc["metrics_y"]

        # Vertical markers at τ_peak
        ax_xcorr.axvline(tau_x, color="#5a8fd0", lw=1.0, linestyle=":", alpha=0.75)
        ax_xcorr.axvline(tau_y, color="#e09050", lw=1.0, linestyle=":", alpha=0.75)

        # Fixed-position metric boxes — top-right corner, stacked, no overlap
        _bbox = dict(boxstyle="round,pad=0.4", alpha=0.92)
        ax_xcorr.text(
            0.98, 0.97,
            f"X (horiz)   ρ_max = {rho_x:.3f}   τ_peak = {tau_x:+.3f} s   FWHM = {fwhm_x:.3f} s",
            transform=ax_xcorr.transAxes, va="top", ha="right",
            color="#5a8fd0", fontsize=9, fontweight="bold",
            bbox={**_bbox, "facecolor": MPL_PANEL, "edgecolor": "#5a8fd0"},
        )
        ax_xcorr.text(
            0.98, 0.80,
            f"Y (vert)     ρ_max = {rho_y:.3f}   τ_peak = {tau_y:+.3f} s   FWHM = {fwhm_y:.3f} s",
            transform=ax_xcorr.transAxes, va="top", ha="right",
            color="#e09050", fontsize=9, fontweight="bold",
            bbox={**_bbox, "facecolor": MPL_PANEL, "edgecolor": "#e09050"},
        )

        ax_xcorr.legend(
            loc="upper left",
            facecolor=MPL_PANEL, edgecolor="#44445a",
            labelcolor="#ddddee", fontsize=9,
        )
    else:
        ax_xcorr.text(
            0.5, 0.5,
            "Not enough data for cross-correlogram\n"
            "(complete at least 2 clicks after a trajectory longer than 0.2 s)",
            ha="center", va="center", transform=ax_xcorr.transAxes,
            color="#9999bb", fontsize=11,
        )

    ax_xcorr.set_xlabel("Lag  τ  (seconds)", fontsize=9)
    ax_xcorr.set_ylabel("Normalised correlation", fontsize=9)
    ax_xcorr.set_title(
        "Cross-Correlogram: actual vs ideal trajectory   "
        "[method: Burge & Bonnen 2025; Iriventi 2025]",
        fontsize=10, pad=7,
    )

    fig.suptitle(
        "Continuous Psychophysics Memory Tracker — Trial Results",
        color="#ddddee", fontsize=13, y=0.98,
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=MPL_DARK)
    plt.close(fig)
    buf.seek(0)
    return pygame.image.load(buf)


# ── Main application ──────────────────────────────────────────────────────────
class App:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Continuous Psychophysics Memory Tracker")
        self.clock  = pygame.time.Clock()
        self.scorer = Scorer()

        def _font(size: int, bold: bool = False) -> pygame.font.Font:
            try:
                return pygame.font.SysFont("Arial", size, bold=bold)
            except Exception:
                return pygame.font.Font(None, size)

        self.f_title = _font(44, bold=True)
        self.f_body  = _font(26)
        self.f_small = _font(20)

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _pump(self) -> List:
        """Drain the event queue; handle global quit events."""
        events = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                pygame.quit(); sys.exit()
            events.append(e)
        return events

    def _text(self, msg: str, y: int, font, colour=TXT, cx: int = WIDTH // 2):
        s = font.render(msg, True, colour)
        r = s.get_rect(center=(cx, y))
        self.screen.blit(s, r)

    def _draw_circles(self, circles: List[Circle]):
        for c in circles:
            pygame.draw.circle(self.screen, c.colour(), (int(c.x), int(c.y)), CIRCLE_R)
            pygame.draw.circle(self.screen, (155, 155, 180), (int(c.x), int(c.y)), CIRCLE_R, 1)

    @staticmethod
    def _make_circles() -> List[Circle]:
        result: List[Circle] = []
        margin = CIRCLE_R + 55
        tries  = 0
        while len(result) < NUM_CIRCLES and tries < 30_000:
            x = random.uniform(margin, WIDTH  - margin)
            y = random.uniform(margin + 90, HEIGHT - margin - 40)
            if all((x - c.x) ** 2 + (y - c.y) ** 2 >= MIN_SEP ** 2 for c in result):
                result.append(Circle(len(result), x, y))
            tries += 1
        return result

    # ── Phases ────────────────────────────────────────────────────────────────
    def phase_intro(self):
        while True:
            self.screen.fill(BG)
            self._text("Continuous Psychophysics Memory Tracker",
                       HEIGHT // 2 - 130, self.f_title, ACCENT)
            self._text(
                f"{NUM_CIRCLES} circles appear. Watch {SEQ_LEN} of them light up — then click those in order.",
                HEIGHT // 2 - 58, self.f_body,
            )
            self._text(
                "Your mouse path between clicks is also scored — move smoothly and directly!",
                HEIGHT // 2 - 10, self.f_small,
            )
            self._text("Press  SPACE  to begin", HEIGHT // 2 + 78, self.f_body, ACCENT)
            pygame.display.flip()
            for e in self._pump():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    return
            self.clock.tick(FPS)

    def phase_show_sequence(self, circles: List[Circle], seq: List[int]):
        for step, idx in enumerate(seq):
            label = f"Memorise  ({step + 1} / {len(seq)})"
            circles[idx].state = "lit"
            self._timed_draw(circles, label, SHOW_DUR)
            circles[idx].state = "dim"
            self._timed_draw(circles, label, GAP_DUR)

    def _timed_draw(self, circles: List[Circle], label: str, duration: float):
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration:
            self.screen.fill(BG)
            self._text(label, 38, self.f_small, ACCENT)
            self._draw_circles(circles)
            pygame.display.flip()
            self._pump()
            self.clock.tick(FPS)

    def phase_wait(self):
        t0 = time.monotonic()
        while time.monotonic() - t0 < WAIT_DUR:
            self.screen.fill(BG)
            self._text("Your turn!", HEIGHT // 2, self.f_title, ACCENT)
            pygame.display.flip()
            self._pump()
            self.clock.tick(FPS)

    def phase_recall(self, circles: List[Circle], trial: Trial):
        n_clicks  = 0
        n_needed  = len(trial.sequence)
        recording = True

        while recording:
            mx, my = pygame.mouse.get_pos()
            trial.trajectory.append((float(mx), float(my), time.monotonic()))

            self.screen.fill(BG)
            self._text(
                f"Click circle  {n_clicks + 1}  of  {n_needed}",
                38, self.f_small, ACCENT,
            )

            # Faint trajectory trail (drawn before circles so it sits underneath)
            if len(trial.trajectory) > 1:
                pts = [(int(x), int(y)) for x, y, _ in trial.trajectory]
                pygame.draw.lines(self.screen, C_TRAJ, False, pts, 2)

            self._draw_circles(circles)
            pygame.display.flip()

            for e in self._pump():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    recording = False
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for c in circles:
                        if c.state == "dim" and c.hit(*e.pos):
                            t_click = time.monotonic()
                            trial.clicks.append(
                                (c.idx, float(e.pos[0]), float(e.pos[1]), t_click)
                            )
                            expected = (
                                trial.sequence[n_clicks] if n_clicks < n_needed else -1
                            )
                            c.state = "ok" if c.idx == expected else "err"
                            n_clicks += 1
                            if n_clicks >= n_needed:
                                pygame.time.wait(450)
                                recording = False
                            break

            self.clock.tick(FPS)

    def phase_results(self, circles: List[Circle], trial: Trial):
        surf = _build_results_surface(trial, circles, self.scorer)

        # Scale down if larger than the window
        sw, sh = surf.get_size()
        scale  = min(WIDTH / sw, (HEIGHT - 44) / sh, 1.0)
        if scale < 1.0:
            surf = pygame.transform.smoothscale(
                surf, (int(sw * scale), int(sh * scale))
            )

        while True:
            self.screen.fill(BG)
            x_off = (WIDTH  - surf.get_width())  // 2
            y_off = max(0, (HEIGHT - surf.get_height()) // 2 - 10)
            self.screen.blit(surf, (x_off, y_off))
            self._text(
                "SPACE → next trial          Q → quit",
                HEIGHT - 16, self.f_small, ACCENT,
            )
            pygame.display.flip()
            for e in self._pump():
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    return
            self.clock.tick(FPS)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self.phase_intro()
        while True:
            circles  = self._make_circles()
            sequence = random.sample(range(NUM_CIRCLES), SEQ_LEN)
            trial    = Trial(sequence=sequence)

            self.phase_show_sequence(circles, sequence)
            self.phase_wait()
            self.phase_recall(circles, trial)
            self.phase_results(circles, trial)


if __name__ == "__main__":
    App().run()
