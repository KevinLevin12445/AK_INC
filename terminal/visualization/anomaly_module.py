import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from engines.yf_cache import yf_download


def _download(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        raw = yf_download(ticker, period=period, interval=interval,
                          auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.dropna(subset=["Close"])
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert(None)
        return raw
    except Exception:
        return pd.DataFrame()


def _detect_anomalies(df: pd.DataFrame, roll_window: int = 20, threshold: float = 2.0) -> pd.DataFrame:
    """Compute returns, rolling mean/std, bands, and anomaly flag."""
    df = df.copy()
    df["ret"]        = df["Close"].pct_change()
    df["roll_mean"]  = df["ret"].rolling(roll_window, min_periods=5).mean()
    df["roll_std"]   = df["ret"].rolling(roll_window, min_periods=5).std()
    df["upper"]      = df["roll_mean"] + threshold * df["roll_std"]
    df["lower"]      = df["roll_mean"] - threshold * df["roll_std"]
    df["anomaly_up"] = df["ret"] > df["upper"]
    df["anomaly_dn"] = df["ret"] < df["lower"]
    df["anomaly"]    = df["anomaly_up"] | df["anomaly_dn"]
    return df.dropna(subset=["ret", "upper", "lower"])


def _add_anomaly_panel(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    label: str,
) -> None:
    """Add returns line + bands + anomaly dots to a subplot row."""

    # Zero reference line
    fig.add_hline(y=0, line=dict(color="#333333", width=1), row=row, col=1)

    # Returns (yellow)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ret"],
        mode="lines",
        name=f"Returns {label}",
        line=dict(color="#ffd700", width=0.9),
        showlegend=True,
    ), row=row, col=1)

    # Upper band (red/orange)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["upper"],
        mode="lines",
        name=f"Upper {label}",
        line=dict(color="#ff4400", width=1.2, dash="solid"),
        showlegend=False,
    ), row=row, col=1)

    # Lower band (green dashed)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["lower"],
        mode="lines",
        name=f"Lower {label}",
        line=dict(color="#00cc44", width=1.2, dash="dash"),
        showlegend=False,
    ), row=row, col=1)

    # Fill between bands
    idx_full = pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]).values
    y_fill   = pd.concat([df["upper"], df["lower"].iloc[::-1]]).values
    fig.add_trace(go.Scatter(
        x=idx_full, y=y_fill,
        fill="toself",
        fillcolor="rgba(255,100,0,0.04)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ), row=row, col=1)

    # Roll mean (orange reference)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["roll_mean"],
        mode="lines",
        name="Mean",
        line=dict(color="#ff8800", width=0.8, dash="dot"),
        showlegend=False,
    ), row=row, col=1)

    # Anomaly dots (UP = magenta, DOWN = cyan)
    anom_up = df[df["anomaly_up"]]
    anom_dn = df[df["anomaly_dn"]]

    if not anom_up.empty:
        fig.add_trace(go.Scatter(
            x=anom_up.index,
            y=anom_up["ret"],
            mode="markers+text",
            marker=dict(color="#ff00ff", size=7, symbol="circle",
                        line=dict(color="#ffffff", width=0.5)),
            text=[ts.strftime("%H:%M") for ts in anom_up.index],
            textfont=dict(color="#ff88ff", size=7, family="monospace"),
            textposition="top center",
            name=f"Anomalía ▲ {label}",
            hovertemplate="<b>%{x}</b><br>ret=%{y:.4f}<extra></extra>",
            showlegend=(row == 1),
        ), row=row, col=1)

    if not anom_dn.empty:
        fig.add_trace(go.Scatter(
            x=anom_dn.index,
            y=anom_dn["ret"],
            mode="markers+text",
            marker=dict(color="#00eeff", size=7, symbol="circle",
                        line=dict(color="#ffffff", width=0.5)),
            text=[ts.strftime("%H:%M") for ts in anom_dn.index],
            textfont=dict(color="#88eeff", size=7, family="monospace"),
            textposition="bottom center",
            name=f"Anomalía ▼ {label}",
            hovertemplate="<b>%{x}</b><br>ret=%{y:.4f}<extra></extra>",
            showlegend=(row == 1),
        ), row=row, col=1)

    # Panel title annotation
    fig.add_annotation(
        row=row, col=1,
        x=0.01, y=0.98,
        xref="x domain", yref="y domain",
        text=f"<b style='color:#00ff41;'>Anomalías {label}</b>",
        showarrow=False,
        font=dict(color="#00ff41", family="monospace", size=10),
        xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0.7)",
    )


def build_anomaly_chart(
    ticker: str = "GC=F",
    period: str = "5d",
    roll_window: int = 20,
    threshold: float = 2.0,
) -> tuple[go.Figure, list]:
    """Return (fig, anomaly_log_list)."""
    df5  = _download(ticker, "5m",  period)
    df15 = _download(ticker, "15m", period)

    if df5.empty or df15.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#000000", height=580,
                          annotations=[dict(text="No intraday data",
                                           x=0.5, y=0.5, xref="paper", yref="paper",
                                           font=dict(color="#00ff41", size=14))])
        return fig, []

    df5  = _detect_anomalies(df5,  roll_window, threshold)
    df15 = _detect_anomalies(df15, roll_window, threshold)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=["", ""],
        row_heights=[0.5, 0.5],
    )

    _add_anomaly_panel(fig, df5,  row=1, label="M5")
    _add_anomaly_panel(fig, df15, row=2, label="M15")

    # ── Styling ───────────────────────────────────────────────────────────────
    axis_style = dict(
        color="#444", gridcolor="#111111",
        showgrid=True, zeroline=True,
        zerolinecolor="#333333", zerolinewidth=1,
        tickfont=dict(size=8, color="#555", family="monospace"),
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#0a0a0a",
        height=580,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(
            font=dict(color="#888", family="monospace", size=9),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#222", x=0.01, y=0.99,
        ),
        title=dict(
            text=(
                f"<b style='color:#00ff41'>YFINANCE ANOMALY DR</b>"
                f"<span style='color:#444;font-size:10px;'>"
                f"  ·  {ticker}  ·  ventana={roll_window}  ·  umbral=±{threshold}σ  ·  señales de reversión a la media</span>"
            ),
            font=dict(color="#00ff41", family="monospace", size=12),
            x=0, xanchor="left",
        ),
    )
    for row in [1, 2]:
        fig.update_xaxes(**axis_style, row=row, col=1)
        fig.update_yaxes(**axis_style, tickformat=".4f", row=row, col=1)

    # ── Build anomaly log ─────────────────────────────────────────────────────
    log = []
    for label, df in [("M5", df5), ("M15", df15)]:
        anom = df[df["anomaly"]].tail(10)
        for ts, row in anom.iterrows():
            direction = "UP" if row["anomaly_up"] else "DOWN"
            log.append({
                "tf":        label,
                "timestamp": pd.Timestamp(ts).strftime("%m-%d %H:%M"),
                "ret":       float(row["ret"]),
                "direction": direction,
            })

    log.sort(key=lambda x: x["timestamp"], reverse=True)
    return fig, log[:20]


def render_anomaly_log(log: list) -> str:
    """Return HTML for the right-side anomaly log panel."""
    if not log:
        return "<div style='color:#333;font-family:monospace;font-size:10px;'>No anomalies detected</div>"

    rows = []
    for item in log:
        color = "#ff00ff" if item["direction"] == "UP" else "#00eeff"
        arrow = "▲" if item["direction"] == "UP" else "▼"
        rows.append(
            f"<div style='padding:3px 0;border-bottom:1px solid #111;"
            f"font-family:monospace;font-size:10px;line-height:1.4;'>"
            f"<span style='color:#444;'>{item['timestamp']}</span> "
            f"<span style='color:#00ff41;font-weight:bold;'>{item['tf']}</span> "
            f"<span style='color:{color};'>{arrow} {item['ret']:+.4f}</span>"
            f"</div>"
        )
    return "".join(rows)
