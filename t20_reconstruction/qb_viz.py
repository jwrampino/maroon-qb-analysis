"""
QB scatter visualization — print-ready, bordered panel layout.
Each chart sits in a framed box with a shaded header,
scatter plot on the left, and metric summary on the right.

Install:
    pip install pandas numpy matplotlib scipy

Usage:
    python qb_viz.py --outdir qb_viz_outputs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300

# Palette
GOOD_COLOR  = "#b6d9b0"
BAD_COLOR   = "#e8b0b0"
PANEL_HEAD  = "#e8e8e8"
BORDER_COL  = "#aaaaaa"
OTHERS_FACE = "#c0c0c0"
OTHERS_EDGE = "#666666"
CALEB_FACE  = "#111111"
CALEB_EDGE  = "white"

HIGHLIGHT_PLAYER = "Caleb Williams"
DEFAULT_CSV = "qb_top20.csv"

COLUMN_ALIASES = {
    "player":     ["Player Name", "Player", "player", "player_name", "name"],
    "plays":      ["Plays", "plays", "Total Plays", "total_plays"],
    "attempts":   ["Attempts", "Pass Attempts", "Att", "ATT", "dropbacks", "Dropbacks"],
    "pass_yards": ["Pass Yards", "Passing Yards", "pass_yards", "Yards"],
    "ypa":        ["YPA", "Yards/Attempt", "yards_per_attempt"],
    "epa_play":   ["EPA/Play", "EPA per Play", "epa_per_play", "epa_play"],
    "success":    ["Success %", "Success Rate", "success_rate", "success_pct"],
    "sack":       ["Sack %", "Sack Rate", "sack_rate", "sack_pct"],
    "adot":       ["ADoT", "aDOT", "Average Depth of Target", "avg_depth_of_target"],
    "bad_throw":  ["Bad Throw %", "Bad%", "Bad Throws %", "Bad Throw Rate", "bad_throw_pct", "bad_throw_rate"],
    "ttt":        ["Time To Throw", "Time to Throw", "TTT", "time_to_throw"],
}

PANELS = [
    dict(
        x_key="success", y_key="epa_play",
        x_label="Success % (successful plays)",
        y_label="EPA / play (team contribution)",
        x_fmt="{:.0f}", y_fmt="{:.2f}",
        good_quad="top_right", bad_quad="bottom_left",
        header="Efficiency and consistency",
        metrics=[
            dict(label="Consistency",
                 sub="% of plays gaining positive expected value",
                 key="success", good_direction="high"),
            dict(label="Efficiency",
                 sub="Expected points added per snap",
                 key="epa_play", good_direction="high"),
        ],
    ),
    dict(
        x_key="adot", y_key="sack",
        x_label="ADoT — avg. depth of target (yds)",
        y_label="Sack rate (%)",
        x_fmt="{:.1f}", y_fmt="{:.1f}",
        good_quad="bottom_right", bad_quad="top_left",
        header="Pressure avoidance vs. downfield aggression",
        metrics=[
            dict(label="Downfield aggression",
                 sub="Avg. yards downfield per target",
                 key="adot", good_direction="high"),
            dict(label="Pressure avoidance",
                 sub="Sack rate",
                 key="sack", good_direction="low"),
        ],
    ),
    dict(
        x_key="ttt", y_key="bad_throw",
        x_label="Time to throw (seconds)",
        y_label="Bad throw rate (%)",
        x_fmt="{:.2f}", y_fmt="{:.0f}",
        good_quad="bottom_left", bad_quad="top_right",
        header="Decision speed and throw quality",
        metrics=[
            dict(label="Decision speed",
                 sub="Secs from snap to release",
                 key="ttt", good_direction="low"),
            dict(label="Throw accuracy",
                 sub="% of passes off-target",
                 key="bad_throw", good_direction="low"),
        ],
    ),
]


def slugify(value):
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_player(value):
    return re.sub(r"\s+", " ", str(value).replace("*", "").replace("+", "").strip()).lower()


def find_col(df, key, override=None, required=True):
    if override:
        if override in df.columns:
            return override
        raise ValueError(f"Column override for {key!r} not found: {override!r}")
    normalized = {slugify(c): c for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        if alias in df.columns:
            return alias
        if slugify(alias) in normalized:
            return normalized[slugify(alias)]
    if required:
        raise ValueError(f"Could not find column for {key!r}. Tried: {COLUMN_ALIASES[key]}")
    return None


def to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def add_attempts(df, cols):
    out = df.copy()
    if cols.get("attempts"):
        out["_attempts"] = to_number(out[cols["attempts"]])
        return out
    py_col, ypa_col = cols.get("pass_yards"), cols.get("ypa")
    if not py_col or not ypa_col:
        raise ValueError("Need attempts OR (pass_yards + ypa) to rank QBs.")
    out["_attempts"] = to_number(out[py_col]) / to_number(out[ypa_col]).replace(0, np.nan)
    return out


def percentile_rating(value, all_values, good_direction):
    """Return (label, color, perf_pct). perf_pct is 0-100 where 100 = best performance."""
    raw_pct = float(
        pd.Series(all_values).rank(pct=True)[pd.Series(all_values) == value].iloc[0]
    ) * 100
    perf_pct = raw_pct if good_direction == "high" else (100 - raw_pct)
    if perf_pct >= 60:
        color = "#27ae60"
    elif perf_pct >= 40:
        color = "#b8860b"
    else:
        color = "#c0392b"
    label = f"{perf_pct:.0f}th percentile"
    return label, color, perf_pct


def prep_data(df, cols):
    out = add_attempts(df, cols)
    out["_plays"]     = to_number(out[cols["plays"]])
    out["_epa_play"]  = to_number(out[cols["epa_play"]])
    out["_success"]   = to_number(out[cols["success"]])
    out["_sack"]      = to_number(out[cols["sack"]])
    out["_adot"]      = to_number(out[cols["adot"]])
    out["_bad_throw"] = to_number(out[cols["bad_throw"]])
    out["_ttt"]       = to_number(out[cols["ttt"]])
    out["_player"]    = out[cols["player"]]
    return out


def sym_range(vals, frac=0.18):
    center = float(vals.median())
    dev = float(np.nanmax(np.abs(vals - center)))
    if dev == 0:
        dev = abs(center) * 0.1 or 1.0
    dev *= 1 + frac
    return center - dev, center + dev, center


def pick_ticks(vmin, vmax, n=4):
    return list(np.linspace(vmin, vmax, n + 2)[1:-1])


def draw_panel(ax_scatter, ax_info, data, panel, player_col):
    x_col = f"_{panel['x_key']}"
    y_col = f"_{panel['y_key']}"

    sub = data[[player_col, x_col, y_col]].dropna().copy()
    sub["_is_caleb"] = sub[player_col].map(normalize_player).eq(normalize_player(HIGHLIGHT_PLAYER))

    xmin, xmax, xmid = sym_range(sub[x_col])
    ymin, ymax, ymid = sym_range(sub[y_col])

    quad_rects = {
        "top_right":    (xmid, xmax, ymid, ymax),
        "top_left":     (xmin, xmid, ymid, ymax),
        "bottom_right": (xmid, xmax, ymin, ymid),
        "bottom_left":  (xmin, xmid, ymin, ymid),
    }
    for quad, color in [(panel["good_quad"], GOOD_COLOR), (panel["bad_quad"], BAD_COLOR)]:
        x0, x1, y0, y1 = quad_rects[quad]
        ax_scatter.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=0, facecolor=color, zorder=0
        ))

    ax_scatter.axvline(xmid, color="#888", linewidth=0.8, linestyle="--", zorder=1, alpha=0.7)
    ax_scatter.axhline(ymid, color="#888", linewidth=0.8, linestyle="--", zorder=1, alpha=0.7)

    others = sub[~sub["_is_caleb"]]
    caleb  = sub[sub["_is_caleb"]]

    ax_scatter.scatter(
        others[x_col], others[y_col],
        s=80, facecolors=OTHERS_FACE, edgecolors=OTHERS_EDGE,
        linewidths=1.3, zorder=2
    )
    if not caleb.empty:
        ax_scatter.scatter(
            caleb[x_col], caleb[y_col],
            s=130, facecolors=CALEB_FACE, edgecolors=CALEB_EDGE,
            linewidths=2.0, zorder=4
        )

    ax_scatter.set_xlim(xmin, xmax)
    ax_scatter.set_ylim(ymin, ymax)

    xticks = pick_ticks(xmin, xmax, 4)
    yticks = pick_ticks(ymin, ymax, 4)
    x_fmt = panel.get("x_fmt", "{:.1f}")
    y_fmt = panel.get("y_fmt", "{:.1f}")
    ax_scatter.set_xticks(xticks)
    ax_scatter.set_xticklabels([x_fmt.format(v) for v in xticks], fontsize=7.5, color="#555")
    ax_scatter.set_yticks(yticks)
    ax_scatter.set_yticklabels([y_fmt.format(v) for v in yticks], fontsize=7.5, color="#555")
    ax_scatter.tick_params(axis="both", length=3, color="#bbb", pad=3)

    ax_scatter.spines[["top", "right"]].set_visible(False)
    ax_scatter.spines["bottom"].set_color("#bbb")
    ax_scatter.spines["left"].set_color("#bbb")
    ax_scatter.set_xlabel(panel["x_label"], fontsize=8.5, color="#444", labelpad=9)
    ax_scatter.set_ylabel(panel["y_label"], fontsize=8.5, color="#444", labelpad=9)

    # Right info panel
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    # Legend
    legend_y = 0.97
    ax_info.scatter([0.055], [legend_y], s=60, facecolors="#111111", edgecolors="white",
                    linewidths=1.5, zorder=5, transform=ax_info.transAxes, clip_on=False)
    ax_info.text(0.13, legend_y, HIGHLIGHT_PLAYER,
                 fontsize=9, fontweight="bold", color="#111",
                 va="center", transform=ax_info.transAxes)
    ax_info.scatter([0.055], [legend_y - 0.07], s=55, facecolors="#c0c0c0", edgecolors="#666666",
                    linewidths=1.2, zorder=5, transform=ax_info.transAxes, clip_on=False)
    ax_info.text(0.13, legend_y - 0.07, "Other QBs",
                 fontsize=8.5, color="#666", va="center", transform=ax_info.transAxes)

    ax_info.add_patch(mpatches.Rectangle(
        (0.02, legend_y - 0.115), 0.96, 0.004,
        linewidth=0, facecolor="#ddd", zorder=3, transform=ax_info.transAxes
    ))

    n_metrics = len(panel["metrics"])
    slot_h    = 0.72 / n_metrics
    top_start = 0.82

    for idx, m in enumerate(panel["metrics"]):
        slot_top = top_start - idx * slot_h

        all_vals   = sub[f"_{m['key']}"].dropna().values
        caleb_vals = sub.loc[sub["_is_caleb"], f"_{m['key']}"].values

        if len(caleb_vals) == 0 or len(all_vals) == 0:
            rating_str, rating_color, perf_pct = "N/A", "#888", 50.0
        else:
            rating_str, rating_color, perf_pct = percentile_rating(
                caleb_vals[0], all_vals, m["good_direction"]
            )

        x0    = 0.04
        bar_w = 0.92

        # Bold metric title
        ax_info.text(x0, slot_top, m["label"],
                     fontsize=10, fontweight="bold", color="#111",
                     va="top", transform=ax_info.transAxes)

        # Grey subtitle
        ax_info.text(x0, slot_top - 0.095, m["sub"],
                     fontsize=6.8, color="#999", va="top",
                     transform=ax_info.transAxes)

        # Bar track (grey background)
        bar_y = slot_top - 0.195
        bar_h = 0.050
        ax_info.add_patch(mpatches.Rectangle(
            (x0, bar_y), bar_w, bar_h,
            linewidth=0, facecolor="#e4e4e2", zorder=3,
            transform=ax_info.transAxes
        ))

        # Bar fill — actual value position in the min-max range
        val_min   = float(np.nanmin(all_vals))
        val_max   = float(np.nanmax(all_vals))
        val_span  = val_max - val_min if val_max != val_min else 1.0
        caleb_val = caleb_vals[0] if len(caleb_vals) > 0 else val_min
        raw_fill  = (caleb_val - val_min) / val_span
        fill_w    = bar_w * max(float(raw_fill), 0.02)
        ax_info.add_patch(mpatches.Rectangle(
            (x0, bar_y), fill_w, bar_h,
            linewidth=0, facecolor=rating_color, alpha=0.82, zorder=4,
            transform=ax_info.transAxes
        ))

        # Percentile label below bar, coloured to match
        ax_info.text(x0, bar_y - 0.055,
                     f"{rating_str} among top-20 QBs",
                     fontsize=7.5, fontweight="bold", color=rating_color,
                     va="top", transform=ax_info.transAxes)

        # Divider between metric cards
        if idx < n_metrics - 1:
            div_y = slot_top - slot_h + 0.025
            ax_info.add_patch(mpatches.Rectangle(
                (0.02, div_y), 0.96, 0.004,
                linewidth=0, facecolor="#e4e4e2", zorder=3,
                transform=ax_info.transAxes
            ))

    return ax_scatter, ax_info


def build_figure(data, subtitle, cols, player_col):
    FW, FH = 10.0, 16.5
    fig = plt.figure(figsize=(FW, FH), facecolor="white")

    margin_l    = 0.55 / FW
    margin_r    = 1.0  - 0.30 / FW
    title_top   = 1.0  - 0.28 / FH
    sub_top     = 1.0  - 0.60 / FH
    content_top = 1.0  - 0.95 / FH
    footer_bot  = 0.48 / FH

    usable_h  = content_top - (footer_bot + 0.55 / FH)
    n_panels  = len(PANELS)
    panel_gap = 0.22 / FH
    panel_h   = (usable_h - panel_gap * (n_panels - 1)) / n_panels

    header_h    = 0.38 / FH
    inner_pad   = 0.18 / FH
    inner_pad_l = 0.90 / FW

    full_w    = margin_r - margin_l
    scatter_w = full_w * 0.505
    info_gap  = 0.025
    info_l    = margin_l + inner_pad_l + scatter_w + info_gap
    info_w    = margin_r - info_l - 0.008

    fig.text(margin_l, title_top,
             "2025 QB Profile: Caleb Williams in Context",
             fontsize=15, fontweight="bold", color="#111", va="top")
    fig.text(margin_l, sub_top, subtitle,
             fontsize=9.5, color="#666", va="top", style="italic")

    for i, panel in enumerate(PANELS):
        panel_top = content_top - i * (panel_h + panel_gap)
        panel_bot = panel_top - panel_h

        # Outer border box
        box = fig.add_axes([margin_l, panel_bot, margin_r - margin_l, panel_h])
        box.set_facecolor("white")
        for sp in box.spines.values():
            sp.set_edgecolor(BORDER_COL)
            sp.set_linewidth(0.9)
        box.set_xticks([]); box.set_yticks([])
        box.set_zorder(1)

        # Shaded header band
        hdr = fig.add_axes([margin_l, panel_top - header_h, margin_r - margin_l, header_h])
        hdr.set_facecolor(PANEL_HEAD)
        for sp in hdr.spines.values():
            sp.set_visible(False)
        hdr.set_xticks([]); hdr.set_yticks([])
        hdr.set_zorder(3)
        hdr.text(0.013, 0.5, panel["header"],
                 fontsize=10, fontweight="bold", color="#111",
                 va="center", transform=hdr.transAxes)

        # Scatter plot axes
        scatter_bot = panel_bot + 0.52 / FH
        scatter_top = panel_top - header_h - inner_pad
        ax_scatter = fig.add_axes([
            margin_l + inner_pad_l, scatter_bot,
            scatter_w, scatter_top - scatter_bot
        ])
        ax_scatter.set_zorder(4)

        # Info panel axes
        ax_info = fig.add_axes([info_l, scatter_bot, info_w, scatter_top - scatter_bot])
        ax_info.set_zorder(4)

        draw_panel(ax_scatter, ax_info, data, panel, player_col)

    fig.text(
        margin_l, 0.032,
        "Green = preferred quadrant  ·  Red = least preferred  ·  Dashed lines = median among plotted QBs",
        fontsize=7.5, color="#999", va="bottom",
    )

    return fig


def save_all(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_base.with_suffix(".jpg")), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_base.with_suffix(".svg")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default=DEFAULT_CSV)
    parser.add_argument("--outdir",        default="qb_viz_outputs")
    parser.add_argument("--bad-throw-col", default=None)
    parser.add_argument("--attempts-col",  default=None)
    args = parser.parse_args()

    df  = pd.read_csv(args.csv)
    out = Path(args.outdir)

    cols = {
        "player":     find_col(df, "player"),
        "plays":      find_col(df, "plays"),
        "attempts":   find_col(df, "attempts",   override=args.attempts_col, required=False),
        "pass_yards": find_col(df, "pass_yards",  required=False),
        "ypa":        find_col(df, "ypa",         required=False),
        "epa_play":   find_col(df, "epa_play"),
        "success":    find_col(df, "success"),
        "sack":       find_col(df, "sack"),
        "adot":       find_col(df, "adot"),
        "bad_throw":  find_col(df, "bad_throw",   override=args.bad_throw_col),
        "ttt":        find_col(df, "ttt"),
    }

    data = prep_data(df, cols)
    needed = ["_plays", "_attempts", "_epa_play", "_success",
              "_sack", "_adot", "_bad_throw", "_ttt"]
    data = data.dropna(subset=needed)

    player_col   = "_player"
    top_plays    = data.nlargest(20, "_plays").copy()
    top_attempts = data.nlargest(20, "_attempts").copy()

    subtitles = {
        "by_plays":    "Top 20 quarterbacks by total plays",
        "by_attempts": "Top 20 quarterbacks by approximated attempts",
    }
    datasets = {
        "by_plays":    top_plays,
        "by_attempts": top_attempts,
    }

    for key, subset in datasets.items():
        fig = build_figure(subset, subtitles[key], cols, player_col)
        save_all(fig, out / f"qb_chart_{key}")
        print(f"Saved: {out}/qb_chart_{key}.{{png,jpg,svg}}")


if __name__ == "__main__":
    main()
