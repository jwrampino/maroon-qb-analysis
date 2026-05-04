"""
QB scatter visualization.
Each chart sits in a framed box with a shaded header,
scatter plot on the left, and metric summary on the right.

Data:
    Expects qb_combined.csv produced by merge_qb_data.py,
    or pass any CSV with the columns listed in COLUMN_ALIASES.
    Sample is all QBs with >= 200 attempts.

Logos:
    Set USE_LOGOS = True and place PNG files in logos/
    named by the Team abbreviation in the CSV, e.g. CHI.png.
    When False, plain filled circles are used instead.

Labeled QBs:
    LABEL_PLAYERS lists names that get annotated in the scatter
    and appear as named entries in the right-panel legend.

Install:
    pip install pandas numpy matplotlib scipy pillow

Usage:
    python qb_viz_combined.py
    python qb_viz_combined.py --csv qb_combined.csv --outdir qb_viz_outputs
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300

# Config
USE_LOGOS   = False
LOGOS_DIR   = Path("logos")
LOGO_ZOOM   = 0.022   # scale factor for logo images in scatter

HIGHLIGHT_PLAYER = "Caleb Williams"
DEFAULT_CSV      = "qb_combined.csv"
MIN_ATTEMPTS     = 200

# QBs to label by name in scatter (per panel, or global fallback)
LABEL_PLAYERS = {
    "panel_0": ["Sam Darnold", "Patrick Mahomes", "Dak Prescott"],
    "panel_1": [],
    "panel_2": ["J.J. McCarthy", "Shedeur Sanders"],
}

# Palette
GOOD_COLOR  = "#b6d9b0"
BAD_COLOR   = "#e8b0b0"
PANEL_HEAD  = "#e8e8e8"
BORDER_COL  = "#aaaaaa"
OTHERS_FACE = "#c0c0c0"
OTHERS_EDGE = "#666666"
CALEB_FACE  = "#111111"
CALEB_EDGE  = "white"

# Distinct colors for labeled QBs — assigned in order of LABEL_PLAYERS per panel
LABEL_PALETTE = ["#2980b9", "#e67e22", "#8e44ad", "#16a085"]

COLUMN_ALIASES = {
    "player":     ["Player", "Player Name", "player", "player_name", "name"],
    "team":       ["Team", "team"],
    "attempts":   ["Att", "Attempts", "Pass Attempts", "ATT", "dropbacks"],
    "plays":      ["Plays", "plays", "Total Plays", "total_plays"],
    "epa_play":   ["EPA_per_play", "EPA/Play", "EPA per Play", "epa_play"],
    "success":    ["Success_pct", "Success %", "Success Rate", "Succ%"],
    "sack":       ["Sack_pct", "Sack %", "Sack Rate", "Sk%"],
    "adot":       ["ADoT", "aDOT", "Average Depth of Target"],
    "bad_throw":  ["Bad_throw_pct", "Bad%", "Bad Throw %", "Bad Throws %"],
    "ttt":        ["Time_to_throw", "Time To Throw", "Time to Throw", "TTT"],
}

PANELS = [
    dict(
        x_key="success", y_key="epa_play",
        x_label="Success % (successful plays)",
        y_label="EPA / play (team contribution)",
        x_fmt="{:.0f}", y_fmt="{:.2f}",
        good_quad="top_right", bad_quad="bottom_left",
        header="Efficiency and consistency",
        label_key="panel_0",
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
        label_key="panel_1",
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
        label_key="panel_2",
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


def find_col(df, key, required=True):
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


def load_logo(team_abbr):
    """Load a logo PNG for a team abbreviation. Returns None if not found."""
    if not USE_LOGOS:
        return None
    path = LOGOS_DIR / f"{team_abbr}.png"
    if not path.exists():
        return None
    try:
        img = plt.imread(str(path))
        return img
    except Exception:
        return None


def get_logo_cache(data, team_col):
    """Pre-load all logos for teams in dataset. Returns dict team -> image or None."""
    cache = {}
    if not USE_LOGOS or team_col is None:
        return cache
    for team in data[team_col].dropna().unique():
        cache[str(team)] = load_logo(str(team))
    return cache


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
    return f"{perf_pct:.0f}th percentile", color, perf_pct


def sym_range(vals, frac=0.18):
    center = float(vals.median())
    dev = float(np.nanmax(np.abs(vals - center)))
    if dev == 0:
        dev = abs(center) * 0.1 or 1.0
    dev *= 1 + frac
    return center - dev, center + dev, center


def pick_ticks(vmin, vmax, n=4):
    return list(np.linspace(vmin, vmax, n + 2)[1:-1])


def plot_marker(ax, x, y, team, logo_cache, is_caleb=False, label_color=None):
    """Plot a single QB marker — logo or dot. label_color is set for labeled QBs."""
    logo = logo_cache.get(str(team)) if logo_cache else None

    if logo is not None:
        imagebox = OffsetImage(logo, zoom=LOGO_ZOOM)
        ab = AnnotationBbox(imagebox, (x, y), frameon=is_caleb,
                            bboxprops=dict(edgecolor="#111", linewidth=1.5 if is_caleb else 0),
                            zorder=4 if is_caleb else 2)
        ax.add_artist(ab)
    else:
        if is_caleb:
            ax.scatter([x], [y], s=130, facecolors=CALEB_FACE, edgecolors=CALEB_EDGE,
                       linewidths=2.0, zorder=4)
        elif label_color is not None:
            ax.scatter([x], [y], s=90, facecolors=label_color, edgecolors="white",
                       linewidths=1.5, zorder=3)
        else:
            ax.scatter([x], [y], s=80, facecolors=OTHERS_FACE, edgecolors=OTHERS_EDGE,
                       linewidths=1.3, zorder=2)


def draw_panel(ax_scatter, ax_info, data, panel, player_col, team_col, logo_cache):
    x_col = f"_{panel['x_key']}"
    y_col = f"_{panel['y_key']}"

    label_names  = LABEL_PLAYERS.get(panel["label_key"], [])
    label_norm   = [normalize_player(n) for n in label_names]
    # Map each labeled player's normalized name to a distinct color
    label_color_map = {n: LABEL_PALETTE[i % len(LABEL_PALETTE)]
                       for i, n in enumerate(label_norm)}

    sub = data[[player_col, team_col, x_col, y_col]].dropna(subset=[x_col, y_col]).copy()
    sub["_is_caleb"]   = sub[player_col].map(normalize_player).eq(normalize_player(HIGHLIGHT_PLAYER))
    sub["_label_color"] = sub[player_col].map(normalize_player).map(label_color_map)

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

    for _, row in sub.iterrows():
        lc = row["_label_color"] if pd.notna(row["_label_color"]) else None
        plot_marker(ax_scatter, row[x_col], row[y_col], row[team_col],
                    logo_cache, is_caleb=row["_is_caleb"], label_color=lc)

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

    # Legend — fixed top zone (0.97 down to LEGEND_BOT)
    # Up to 4 labeled QBs + Caleb + Other QBs = 6 entries max at 0.055 spacing = 0.33
    LEGEND_BOT  = 0.60   # legend always ends here regardless of entry count
    CARDS_TOP   = 0.56   # metric cards always start here
    CARD_SLOT_H = 0.27   # fixed height per metric card

    legend_y = 0.97
    caleb_team = sub.loc[sub["_is_caleb"], team_col].values
    caleb_logo = logo_cache.get(str(caleb_team[0])) if (USE_LOGOS and len(caleb_team)) else None

    if caleb_logo is not None:
        imagebox = OffsetImage(caleb_logo, zoom=LOGO_ZOOM * 1.8)
        ab = AnnotationBbox(imagebox, (0.055, legend_y), frameon=False,
                            xycoords=ax_info.transAxes, zorder=5)
        ax_info.add_artist(ab)
    else:
        ax_info.scatter([0.055], [legend_y], s=60, facecolors=CALEB_FACE, edgecolors="white",
                        linewidths=1.5, zorder=5, transform=ax_info.transAxes, clip_on=False)
    ax_info.text(0.13, legend_y, HIGHLIGHT_PLAYER,
                 fontsize=9, fontweight="bold", color="#111",
                 va="center", transform=ax_info.transAxes)

    # Labeled QBs in legend with their assigned color
    legend_cursor = legend_y - 0.075
    for name, lc in zip(label_names, [label_color_map[normalize_player(n)] for n in label_names]):
        logo = logo_cache.get(str(
            sub.loc[sub[player_col].map(normalize_player) == normalize_player(name), team_col]
            .values[0])) if USE_LOGOS else None
        if logo is not None:
            imagebox = OffsetImage(logo, zoom=LOGO_ZOOM * 1.6)
            ab = AnnotationBbox(imagebox, (0.055, legend_cursor), frameon=False,
                                xycoords=ax_info.transAxes, zorder=5)
            ax_info.add_artist(ab)
        else:
            ax_info.scatter([0.055], [legend_cursor], s=55, facecolors=lc, edgecolors="white",
                            linewidths=1.3, zorder=5, transform=ax_info.transAxes, clip_on=False)
        ax_info.text(0.13, legend_cursor, name,
                     fontsize=8, color="#444", va="center", transform=ax_info.transAxes)
        legend_cursor -= 0.075

    # Other QBs entry
    ax_info.scatter([0.055], [legend_cursor], s=50, facecolors=OTHERS_FACE, edgecolors=OTHERS_EDGE,
                    linewidths=1.2, zorder=5, transform=ax_info.transAxes, clip_on=False)
    ax_info.text(0.13, legend_cursor, "Other QBs",
                 fontsize=8, color="#666", va="center", transform=ax_info.transAxes)

    # Thin rule between legend and metric cards
    ax_info.add_patch(mpatches.Rectangle(
        (0.02, LEGEND_BOT), 0.96, 0.004,
        linewidth=0, facecolor="#ddd", zorder=3, transform=ax_info.transAxes
    ))

    # Metric cards — fixed zone, consistent across all panels
    n_metrics = len(panel["metrics"])
    for idx, m in enumerate(panel["metrics"]):
        slot_top = CARDS_TOP - idx * CARD_SLOT_H

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

        ax_info.text(x0, slot_top, m["label"],
                     fontsize=10, fontweight="bold", color="#111",
                     va="top", transform=ax_info.transAxes)

        ax_info.text(x0, slot_top - 0.075, m["sub"],
                     fontsize=6.8, color="#999", va="top",
                     transform=ax_info.transAxes)

        bar_y = slot_top - 0.155
        bar_h = 0.048
        ax_info.add_patch(mpatches.Rectangle(
            (x0, bar_y), bar_w, bar_h,
            linewidth=0, facecolor="#e4e4e2", zorder=3,
            transform=ax_info.transAxes
        ))

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

        ax_info.text(x0, bar_y - 0.048,
                     f"{rating_str} among {len(all_vals)} QBs",
                     fontsize=7.5, fontweight="bold", color=rating_color,
                     va="top", transform=ax_info.transAxes)

        if idx < n_metrics - 1:
            div_y = slot_top - CARD_SLOT_H + 0.01
            ax_info.add_patch(mpatches.Rectangle(
                (0.02, div_y), 0.96, 0.004,
                linewidth=0, facecolor="#e4e4e2", zorder=3,
                transform=ax_info.transAxes
            ))

    return ax_scatter, ax_info


def build_figure(data, subtitle, team_col, logo_cache):
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

    player_col = "_player"

    for i, panel in enumerate(PANELS):
        panel_top = content_top - i * (panel_h + panel_gap)
        panel_bot = panel_top - panel_h

        box = fig.add_axes([margin_l, panel_bot, margin_r - margin_l, panel_h])
        box.set_facecolor("white")
        for sp in box.spines.values():
            sp.set_edgecolor(BORDER_COL)
            sp.set_linewidth(0.9)
        box.set_xticks([]); box.set_yticks([])
        box.set_zorder(1)

        hdr = fig.add_axes([margin_l, panel_top - header_h, margin_r - margin_l, header_h])
        hdr.set_facecolor(PANEL_HEAD)
        for sp in hdr.spines.values():
            sp.set_visible(False)
        hdr.set_xticks([]); hdr.set_yticks([])
        hdr.set_zorder(3)
        hdr.text(0.013, 0.5, panel["header"],
                 fontsize=10, fontweight="bold", color="#111",
                 va="center", transform=hdr.transAxes)

        scatter_bot = panel_bot + 0.52 / FH
        scatter_top = panel_top - header_h - inner_pad
        ax_scatter = fig.add_axes([
            margin_l + inner_pad_l, scatter_bot,
            scatter_w, scatter_top - scatter_bot
        ])
        ax_scatter.set_zorder(4)

        ax_info = fig.add_axes([info_l, scatter_bot, info_w, scatter_top - scatter_bot])
        ax_info.set_zorder(4)

        draw_panel(ax_scatter, ax_info, data, panel, player_col, team_col, logo_cache)

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
    parser.add_argument("--csv",    default=DEFAULT_CSV)
    parser.add_argument("--outdir", default="qb_viz_outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.outdir)

    player_col  = find_col(df, "player")
    team_col    = find_col(df, "team", required=False)
    att_col     = find_col(df, "attempts", required=False)
    plays_col   = find_col(df, "plays", required=False)

    df["_player"] = df[player_col].apply(
        lambda n: re.sub(r"\s+", " ", re.sub(r"[*+]", "", re.sub(r"^\d+\.\s*", "", str(n))).strip())
    )
    df["_team"] = df[team_col] if team_col else "UNK"

    for key in ["epa_play", "success", "sack", "adot", "bad_throw", "ttt"]:
        col = find_col(df, key)
        df[f"_{key}"] = pd.to_numeric(
            df[col].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce"
        )

    # Filter to >= MIN_ATTEMPTS
    if att_col:
        df["_att"] = pd.to_numeric(
            df[att_col].astype(str).str.replace("*", "", regex=False).str.strip(),
            errors="coerce"
        )
        df = df[df["_att"] >= MIN_ATTEMPTS].copy()
    elif plays_col:
        df["_plays"] = pd.to_numeric(df[plays_col], errors="coerce")
        df = df.nlargest(20, "_plays").copy()

    df = df.dropna(subset=["_epa_play","_success","_sack","_adot","_bad_throw","_ttt"])
    df = df.reset_index(drop=True)

    logo_cache = get_logo_cache(df, "_team")
    subtitle = f"All QBs with {MIN_ATTEMPTS}+ pass attempts, 2025 season"

    fig = build_figure(df, subtitle, "_team", logo_cache)
    save_all(fig, out / "qb_chart")
    print(f"Saved: {out}/qb_chart.{{png,jpg,svg}}  ({len(df)} QBs)")


if __name__ == "__main__":
    main()
