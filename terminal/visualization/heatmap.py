import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")


def _compute_pivot_zones(
    vol_profile: np.ndarray,
    bin_centers: np.ndarray,
    bin_width: float,
    poc_price: float,
    vol_norm: np.ndarray,
    max_pivots: int = 5,
) -> list:
    """
    Find high-probability pivot zones from local volume profile maxima.
    Returns list of (price, strength_0_to_1, label) sorted by price descending.
    Excludes a band around POC (already annotated separately).
    """
    if vol_profile.max() == 0:
        return []

    # Dynamic parameters
    min_height   = vol_profile.max() * 0.10    # at least 10% of peak volume
    min_dist     = max(2, len(bin_centers) // 14)  # minimum bins between peaks
    min_prom     = vol_profile.max() * 0.04

    peaks, props = find_peaks(
        vol_profile,
        height=min_height,
        distance=min_dist,
        prominence=min_prom,
    )

    poc_exclusion = bin_width * 2.5  # exclude peaks too close to POC

    pivot_zones = []
    for i in peaks:
        p   = float(bin_centers[i])
        if abs(p - poc_price) < poc_exclusion:
            continue
        strength = float(vol_norm[i])
        pivot_zones.append((p, strength))

    # Sort by strength descending, keep top N
    pivot_zones.sort(key=lambda x: -x[1])
    pivot_zones = pivot_zones[:max_pivots]

    # Sort by price descending for consistent visual labeling
    pivot_zones.sort(key=lambda x: -x[0])
    return pivot_zones


def build_order_flow_heatmap(
    price: pd.Series,
    returns: pd.Series,
    n_bins: int = 50,
    window: int = 20,
) -> go.Figure:
    """
    Institutional order flow heatmap with pivot zones:
    - Numeric y-axis (real zoom/pan)
    - Price path line overlaid on heatmap
    - High-contrast colorscale
    - POC / VAH / VAL / NOW annotations
    - Pivot zones: high-probability reaction levels detected via vol profile peaks
    - Volume profile sidebar
    """
    def _strip_tz(s):
        idx = pd.to_datetime(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        s = s.copy()
        s.index = idx
        return s[~s.index.duplicated(keep="last")]

    price   = _strip_tz(price)
    returns = _strip_tz(returns)

    price_clean = price.dropna()
    common = price_clean.index.intersection(returns.dropna().index)
    if len(common) >= 10:
        ret_clean     = returns.reindex(common).dropna()
        price_aligned = price_clean.reindex(ret_clean.index)
    else:
        combined      = pd.concat([price_clean.rename("p"), returns.rename("r")],
                                   axis=1).ffill().dropna()
        price_aligned = combined["p"]
        ret_clean     = combined["r"]

    if len(price_aligned) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color="#00ff41", size=14))
        fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000",
                          font=dict(color="#00ff41"))
        return fig

    vol_proxy  = (ret_clean.abs() * price_aligned).values
    prices_arr = price_aligned.values.astype(float)
    dates      = price_aligned.index

    p_min, p_max = float(prices_arr.min()), float(prices_arr.max())
    pad          = (p_max - p_min) * 0.03
    bins         = np.linspace(p_min - pad, p_max + pad, n_bins + 1)
    bin_centers  = 0.5 * (bins[:-1] + bins[1:])
    bin_width    = bins[1] - bins[0]

    # ── heatmap matrix (price_bins × time_slices) ─────────────────────────────
    n_time   = len(dates)
    n_slices = min(100, n_time)
    step     = max(1, n_time // n_slices)
    time_idx = list(range(0, n_time, step))

    hm = np.zeros((n_bins, len(time_idx)))
    for j, t in enumerate(time_idx):
        start   = max(0, t - window)
        p_slice = prices_arr[start: t + 1]
        v_slice = vol_proxy[start: t + 1]
        for i, (bl, bu) in enumerate(zip(bins[:-1], bins[1:])):
            mask      = (p_slice >= bl) & (p_slice < bu)
            hm[i, j]  = v_slice[mask].sum()

    # ── KDE smoothing ──────────────────────────────────────────────────────────
    smoothed = np.zeros_like(hm)
    for j in range(hm.shape[1]):
        col = hm[:, j]
        if col.sum() > 0:
            try:
                kde = gaussian_kde(bin_centers, weights=col + 1e-10, bw_method=0.15)
                smoothed[:, j] = kde(bin_centers)
            except Exception:
                smoothed[:, j] = col
        else:
            smoothed[:, j] = col

    smoothed = np.log1p(smoothed * 500)
    col_max  = smoothed.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1
    smoothed /= col_max

    # ── volume profile ─────────────────────────────────────────────────────────
    vol_profile = np.zeros(n_bins)
    for i, (bl, bu) in enumerate(zip(bins[:-1], bins[1:])):
        mask            = (prices_arr >= bl) & (prices_arr < bu)
        vol_profile[i]  = vol_proxy[mask].sum()

    if vol_profile.sum() > 0:
        try:
            kde_vp      = gaussian_kde(bin_centers, weights=vol_profile + 1e-10, bw_method=0.15)
            vol_profile = kde_vp(bin_centers)
        except Exception:
            pass

    vol_norm = vol_profile / vol_profile.max() if vol_profile.max() > 0 else vol_profile

    # ── key levels ────────────────────────────────────────────────────────────
    poc_idx       = int(np.argmax(vol_profile))
    poc_price     = float(bin_centers[poc_idx])
    sorted_idx    = np.argsort(vol_profile)[::-1]
    cum_vol       = np.cumsum(vol_profile[sorted_idx])
    va_threshold  = 0.70 * vol_profile.sum()
    va_idx        = sorted_idx[: np.searchsorted(cum_vol, va_threshold) + 1]
    vah_price     = float(bin_centers[va_idx].max())
    val_price     = float(bin_centers[va_idx].min())
    current_price = float(price_clean.iloc[-1])

    # ── PIVOT ZONES — high-probability reaction levels ────────────────────────
    pivot_zones = _compute_pivot_zones(
        vol_profile, bin_centers, bin_width, poc_price, vol_norm, max_pivots=5
    )

    # ── date labels (x-axis) ──────────────────────────────────────────────────
    x_vals = []
    for t in time_idx:
        dt = dates[t]
        try:
            if hasattr(dt, "hour") and dt.hour != 0:
                x_vals.append(dt.strftime("%m/%d %H:%M"))
            else:
                x_vals.append(dt.strftime("%b %d"))
        except Exception:
            x_vals.append(str(dt)[:10])

    # ── colorscale: dark → blue → cyan → green → yellow → red ────────────────
    COLORSCALE = [
        [0.00, "#050510"],
        [0.08, "#0d1b3e"],
        [0.20, "#0a3d7a"],
        [0.35, "#0077b6"],
        [0.50, "#00b4d8"],
        [0.65, "#00cc7a"],
        [0.78, "#90e000"],
        [0.88, "#ffd000"],
        [0.95, "#ff7700"],
        [1.00, "#ff2200"],
    ]

    # ── figure layout: 2 columns — main heatmap | volume profile ─────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.80, 0.20],
        shared_yaxes=True,
        horizontal_spacing=0.01,
        subplot_titles=["", "VOL PROFILE"],
    )

    # ① Main heatmap (numeric y — enables real zoom)
    fig.add_trace(
        go.Heatmap(
            z=smoothed,
            x=x_vals,
            y=bin_centers,
            colorscale=COLORSCALE,
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="DENSITY",
                    font=dict(color="#aaaaaa", family="monospace", size=9),
                    side="right",
                ),
                tickfont=dict(color="#aaaaaa", family="monospace", size=9),
                tickvals=[0, 0.5, 1.0],
                ticktext=["LOW", "MED", "HIGH"],
                len=0.75,
                thickness=10,
                x=1.01,
                bgcolor="#000000",
                bordercolor="#222222",
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Price:</b> $%{y:,.1f}<br>"
                "<b>Density:</b> %{z:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # ② Price path line overlaid
    price_path_x = [x_vals[j] for j in range(len(time_idx))]
    price_path_y = [float(prices_arr[t]) for t in time_idx]
    fig.add_trace(
        go.Scatter(
            x=price_path_x,
            y=price_path_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.85)", width=2),
            name="XAU/USD",
            hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ③ Volume profile bars (right panel)
    bar_colors = [
        f"rgba({int(255*(v**0.5))},{int(200*(1-v))},0,0.85)"
        for v in vol_norm
    ]
    fig.add_trace(
        go.Bar(
            x=vol_norm,
            y=bin_centers,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            showlegend=False,
            hovertemplate="$%{y:,.1f}  vol: %{x:.2f}<extra></extra>",
            name="Vol Profile",
        ),
        row=1, col=2,
    )

    # ── Pivot zone shaded bands ───────────────────────────────────────────────
    half_band = bin_width * 1.2  # band height = ±1.2 bin widths
    PZ_COLOR  = "#cc44ff"        # magenta-purple for pivot zones

    for pz_price, strength in pivot_zones:
        opacity = 0.10 + strength * 0.18  # stronger = more opaque band

        # Shaded band on heatmap
        fig.add_shape(
            type="rect",
            row=1, col=1,
            x0=0, x1=1, xref="x domain",
            y0=pz_price - half_band,
            y1=pz_price + half_band,
            yref="y",
            fillcolor=f"rgba(180,60,255,{opacity:.2f})",
            line=dict(color="rgba(180,60,255,0.0)", width=0),
            layer="below",
        )

        # Dashed pivot line
        fig.add_shape(
            type="line",
            row=1, col=1,
            x0=0, x1=1, xref="x domain",
            y0=pz_price, y1=pz_price, yref="y",
            line=dict(color=PZ_COLOR, width=1, dash="dot"),
            opacity=0.65,
        )

        # Label — alternate sides to avoid overlap
        xanchor_side = "right"
        xpos = 0.97

        fig.add_annotation(
            row=1, col=1,
            x=xpos, y=pz_price,
            xref="x domain", yref="y",
            text=f"<b>PZ</b> ${pz_price:,.0f} <span style='color:#888;'>({strength:.0%})</span>",
            showarrow=False,
            font=dict(color=PZ_COLOR, size=9, family="monospace"),
            xanchor=xanchor_side,
            bgcolor="rgba(0,0,0,0.82)",
            bordercolor=PZ_COLOR,
            borderwidth=1,
            borderpad=2,
        )

        # Mirror pivot line on volume profile panel
        fig.add_shape(
            type="line",
            row=1, col=2,
            x0=0, x1=1, xref="x2 domain",
            y0=pz_price, y1=pz_price, yref="y",
            line=dict(color=PZ_COLOR, width=1, dash="dot"),
            opacity=0.50,
        )

    # ── Fixed key level horizontal lines ─────────────────────────────────────
    LEVELS = [
        (poc_price,     "#ff7700", "solid", 2,   "POC",  "left",  0.02),
        (vah_price,     "#ffd700", "dot",   1.5, "VAH",  "left",  0.02),
        (val_price,     "#ffd700", "dot",   1.5, "VAL",  "left",  0.02),
        (current_price, "#00ff41", "dash",  2.5, "NOW",  "right", 0.97),
    ]

    for lv, color, dash, lw, label, side, xpos in LEVELS:
        fig.add_shape(
            type="line", row=1, col=1,
            x0=0, x1=1, xref="x domain",
            y0=lv, y1=lv, yref="y",
            line=dict(color=color, width=lw, dash=dash),
            opacity=0.9,
        )
        fig.add_annotation(
            row=1, col=1,
            x=xpos, y=lv,
            xref="x domain", yref="y",
            text=f"<b>{label}</b> ${lv:,.0f}",
            showarrow=False,
            font=dict(color=color, size=10, family="monospace"),
            xanchor=side,
            bgcolor="rgba(0,0,0,0.80)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )
        fig.add_shape(
            type="line", row=1, col=2,
            x0=0, x1=1, xref="x2 domain",
            y0=lv, y1=lv, yref="y",
            line=dict(color=color, width=1, dash=dash),
            opacity=0.7,
        )

    # ── layout ────────────────────────────────────────────────────────────────
    n_ticks  = min(10, len(x_vals))
    tick_step = max(1, len(x_vals) // n_ticks)
    x_ticks   = x_vals[::tick_step]

    pz_count_str = f"  │  <span style='color:#cc44ff'>PZ ×{len(pivot_zones)}</span>" if pivot_zones else ""

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#050510",
        font=dict(color="#cccccc", family="monospace"),
        height=620,
        margin=dict(l=80, r=80, t=55, b=65),
        bargap=0,
        showlegend=False,
        title=dict(
            text=(
                "<b style='color:#00ff41'>ORDER FLOW HEATMAP</b>"
                f"<span style='color:#888888;font-size:11px;'>"
                f"  ▶  POC <b style='color:#ff7700'>${poc_price:,.0f}</b>"
                f"  │  VAH <b style='color:#ffd700'>${vah_price:,.0f}</b>"
                f"  │  VAL <b style='color:#ffd700'>${val_price:,.0f}</b>"
                f"  │  NOW <b style='color:#00ff41'>${current_price:,.0f}</b>"
                f"{pz_count_str}"
                "</span>"
            ),
            font=dict(color="#00ff41", family="monospace", size=12),
            x=0,
            xanchor="left",
        ),
        annotations=[
            a for a in fig.layout.annotations
            if not (hasattr(a, "text") and a.text in ("", "VOL PROFILE"))
        ] + [
            dict(
                text="VOL PROFILE",
                x=0.895, y=1.04,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#555555", size=9, family="monospace"),
                xanchor="center",
            )
        ],
    )

    # x-axis — main heatmap
    fig.update_xaxes(
        color="#666666",
        gridcolor="#111122",
        gridwidth=1,
        tickfont=dict(size=9, family="monospace", color="#888888"),
        tickangle=40,
        tickvals=x_ticks,
        ticktext=x_ticks,
        showgrid=True,
        zeroline=False,
        row=1, col=1,
    )
    # x-axis — volume profile
    fig.update_xaxes(
        color="#333333",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1, col=2,
    )
    # y-axis (shared, numeric)
    fig.update_yaxes(
        color="#888888",
        gridcolor="#111122",
        gridwidth=1,
        tickfont=dict(size=9, family="monospace", color="#888888"),
        tickformat="$,.0f",
        title=dict(text="XAU/USD PRICE", font=dict(color="#555555", size=10)),
        showgrid=True,
        zeroline=False,
        range=[p_min - pad * 2, p_max + pad * 2],
        row=1, col=1,
    )

    return fig
