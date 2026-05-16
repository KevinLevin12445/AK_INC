import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


# Only 3 features remain in GC3D — zscore_20/60, VWAP_Dev, OrderImbalance, Carry removed
FEATURE_LABELS = {
    "YieldAnomaly": "Yield Anomaly",
    "StochVol":     "Stoch. Volatility",
    "Coint_ZScore": "Coint. Spread Z",
}


def build_gc3d(
    price: pd.Series,
    features: pd.DataFrame,
    feature_col: str = "YieldAnomaly",
    window: int = 5,
) -> go.Figure:
    """
    GC3D — 3-dimensional alpha surface:
        X = Time index
        Y = Price (XAUUSD)
        Z = Feature value (YieldAnomaly | StochVol | Coint_ZScore)
    """
    # Restrict to valid features only
    valid_cols = [c for c in features.columns if c in FEATURE_LABELS]
    if feature_col not in valid_cols:
        feature_col = valid_cols[0] if valid_cols else (features.columns[0] if len(features.columns) else "")

    if not feature_col or feature_col not in features.columns:
        fig = go.Figure()
        fig.add_annotation(text="Feature not available. Initialize engines first.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#00ff41", size=14))
        return _style_3d(fig, feature_col)

    # Strip timezone only — do NOT normalize to midnight (breaks intraday alignment)
    def _strip_tz(idx):
        idx = pd.to_datetime(idx)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        return idx

    price_s = price.copy()
    price_s.index = _strip_tz(price_s.index)
    price_s = price_s[~price_s.index.duplicated(keep="last")]

    feat_s = features[feature_col].copy()
    feat_s.index = _strip_tz(feat_s.index)
    feat_s = feat_s[~feat_s.index.duplicated(keep="last")]

    # Align on the intersection of both indexes
    common_idx = price_s.index.intersection(feat_s.index)
    if len(common_idx) >= 5:
        df = pd.concat([price_s.reindex(common_idx).rename("price"),
                        feat_s.reindex(common_idx).rename("feature")], axis=1).dropna()
    else:
        df = pd.concat([price_s.rename("price"), feat_s.rename("feature")], axis=1)
        df = df.sort_index().ffill().bfill().dropna()

    if len(df) < 5:
        df = pd.concat([price_s.rename("price"), feat_s.rename("feature")],
                       axis=1).dropna(how="all").ffill().dropna()

    if len(df) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for GC3D", xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color="#00ff41", size=14))
        return _style_3d(fig, feature_col)

    df = df.tail(120)
    t = np.arange(len(df))
    y = df["price"].values
    z = df["feature"].values

    ts_index = df.index
    is_intraday = hasattr(ts_index, 'hour') and ts_index[0].hour != 0
    fmt = "%m-%d %H:%M" if is_intraday else "%Y-%m-%d"
    ts_labels = [pd.Timestamp(ts).strftime(fmt) for ts in ts_index]

    tick_step = max(1, len(t) // 10)
    tick_vals = t[::tick_step].tolist()
    tick_text = [ts_labels[i] for i in range(0, len(t), tick_step)]

    z_norm = (z - z.min()) / ((z.max() - z.min()) + 1e-10)
    last_dt = pd.Timestamp(ts_index[-1]).strftime(fmt)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=t, y=y, z=z,
        mode="lines+markers",
        marker=dict(
            size=3,
            color=z_norm,
            colorscale=[
                [0.0, "#000080"],
                [0.25, "#004400"],
                [0.5,  "#00ff41"],
                [0.75, "#ffd700"],
                [1.0,  "#ff4444"],
            ],
            opacity=0.9,
            colorbar=dict(
                title=dict(text=FEATURE_LABELS.get(feature_col, feature_col),
                           font=dict(color="#00ff41", family="monospace", size=10)),
                tickfont=dict(color="#00ff41", family="monospace", size=9),
                thickness=12,
            ),
        ),
        line=dict(
            color=z,
            colorscale=[
                [0.0, "#000080"],
                [0.5,  "#00ff41"],
                [1.0,  "#ff4444"],
            ],
            width=3,
        ),
        customdata=ts_labels,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Price: $%{y:,.2f}<br>"
            f"{FEATURE_LABELS.get(feature_col, feature_col)}: %{{z:.4f}}<extra></extra>"
        ),
        name="Alpha Surface",
    ))

    last_t = t[-1]
    last_y = y[-1]
    last_z = z[-1]

    fig.add_trace(go.Scatter3d(
        x=[last_t], y=[last_y], z=[last_z],
        mode="markers+text",
        marker=dict(size=8, color="#ffd700", symbol="diamond"),
        text=[f"  ${last_y:,.2f}  [{last_dt}]"],
        textfont=dict(color="#ffd700", size=10, family="monospace"),
        name="Current",
        showlegend=False,
    ))

    return _style_3d(fig, feature_col, tick_vals=tick_vals, tick_text=tick_text)


def _style_3d(
    fig: go.Figure,
    feature_col: str,
    tick_vals: list | None = None,
    tick_text: list | None = None,
) -> go.Figure:
    xaxis_extra = {}
    if tick_vals and tick_text:
        xaxis_extra = dict(tickvals=tick_vals, ticktext=tick_text)

    fig.update_layout(
        title=dict(
            text=f"GC3D — Alpha Surface | Feature: {FEATURE_LABELS.get(feature_col, feature_col)}",
            font=dict(color="#00ff41", family="monospace", size=13),
        ),
        paper_bgcolor="#000000",
        scene=dict(
            bgcolor="#000000",
            xaxis=dict(
                title=dict(text="Date / Time", font=dict(color="#00ff41", family="monospace")),
                color="#00ff41", gridcolor="#0a2a0a", backgroundcolor="#000000",
                tickfont=dict(color="#00ff41", family="monospace", size=8),
                **xaxis_extra,
            ),
            yaxis=dict(
                title=dict(text="Price (XAUUSD)", font=dict(color="#00ff41", family="monospace")),
                color="#00ff41", gridcolor="#0a2a0a", backgroundcolor="#000000",
                tickfont=dict(color="#00ff41", family="monospace", size=8),
            ),
            zaxis=dict(
                title=dict(text=FEATURE_LABELS.get(feature_col, feature_col),
                           font=dict(color="#00ff41", family="monospace")),
                color="#00ff41", gridcolor="#0a2a0a", backgroundcolor="#000000",
                tickfont=dict(color="#00ff41", family="monospace", size=8),
            ),
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
        ),
        font=dict(color="#00ff41", family="monospace"),
        height=580,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(font=dict(color="#00ff41", family="monospace"), bgcolor="#000000"),
    )
    return fig


def build_volatility_surface(features: pd.DataFrame) -> go.Figure:
    """
    Secondary GC3D view: multi-feature surface over time.
    Uses only the 3 valid GC3D features.
    """
    valid = ["YieldAnomaly", "StochVol", "Coint_ZScore"]
    cols  = [c for c in valid if c in features.columns]
    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need ≥2 features (initialize engines first)",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#00ff41"))
        return _style_3d(fig, "surface")

    df = features[cols].dropna().tail(100)
    t  = np.arange(len(df))

    fig = go.Figure(data=[go.Surface(
        z=df.values.T,
        x=t,
        y=cols,
        colorscale=[
            [0.0, "#000000"],
            [0.3, "#003300"],
            [0.6, "#00ff41"],
            [1.0, "#ffd700"],
        ],
        opacity=0.85,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#ffd700", project_z=True),
        ),
    )])

    return _style_3d(fig, "Multi-Feature Surface")
