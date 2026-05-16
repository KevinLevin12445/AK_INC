import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from engines.yf_cache import yf_download


def _download_intraday(ticker: str = "GC=F", interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    try:
        raw = yf_download(ticker, period=period, interval=interval,
                          auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.dropna(subset=["Close", "Volume"])
        return raw
    except Exception:
        return pd.DataFrame()


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute session VWAP (resets daily) + total VWAP + SD bands."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df["typical"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tp_vol"]  = df["typical"] * df["Volume"]
    df["date"]    = df.index.date

    # Session-cumulative VWAP (resets each day)
    df["cum_tpvol_s"]  = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol_s"]    = df.groupby("date")["Volume"].cumsum()
    df["vwap_session"] = df["cum_tpvol_s"] / df["cum_vol_s"].replace(0, np.nan)

    # Total accumulated VWAP
    df["cum_tpvol_t"] = df["tp_vol"].cumsum()
    df["cum_vol_t"]   = df["Volume"].cumsum()
    df["vwap_total"]  = df["cum_tpvol_t"] / df["cum_vol_t"].replace(0, np.nan)

    # Session standard deviation bands
    df["dev2"]       = (df["typical"] - df["vwap_session"]) ** 2
    grp_cnt          = df.groupby("date").cumcount() + 1
    df["session_var"] = df.groupby("date")["dev2"].cumsum() / grp_cnt
    df["session_sd"]  = np.sqrt(df["session_var"]).fillna(0)

    return df


def build_vwap_chart(ticker: str = "GC=F", interval: str = "5m", period: str = "5d") -> go.Figure:
    df = _download_intraday(ticker, interval, period)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#000000", height=520,
                          annotations=[dict(text="No intraday data available",
                                           x=0.5, y=0.5, xref="paper", yref="paper",
                                           font=dict(color="#00ff41", size=14))])
        return fig

    df = compute_vwap(df)

    # Session boundary times
    dates      = sorted(df["date"].unique())
    session_starts = [df[df["date"] == d].index[0]  for d in dates]
    session_ends   = [df[df["date"] == d].index[-1] for d in dates]

    # SD band colors: ±1 magenta, ±2 orange, ±3 red, ±4 dark magenta
    SD_BANDS = [
        (4, "#aa00cc", "dot",   0.6, 0.02),
        (3, "#ff4444", "dash",  0.7, 0.03),
        (2, "#ff8800", "solid", 0.8, 0.05),
        (1, "#ff00ff", "solid", 0.9, 0.06),
    ]

    fig = go.Figure()

    # ── SD filled zones ──────────────────────────────────────────────────────
    for n, color, _, opacity, fill_alpha in SD_BANDS:
        u = df["vwap_session"] + n * df["session_sd"]
        l = df["vwap_session"] - n * df["session_sd"]
        idx_full = pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]).values
        y_fill   = pd.concat([u, l.iloc[::-1]]).values
        r, g, b  = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=idx_full, y=y_fill,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},{fill_alpha:.2f})",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── SD band lines ─────────────────────────────────────────────────────────
    for n, color, dash, width, _ in SD_BANDS:
        u = df["vwap_session"] + n * df["session_sd"]
        l = df["vwap_session"] - n * df["session_sd"]
        fig.add_trace(go.Scatter(
            x=df.index, y=u, mode="lines",
            name=f"+{n} SD",
            line=dict(color=color, width=width, dash=dash),
            showlegend=(n == 1),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=l, mode="lines",
            name=f"-{n} SD",
            line=dict(color=color, width=width, dash=dash),
            showlegend=False,
        ))

    # ── Session VWAP (green) ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["vwap_session"],
        mode="lines", name="VWAP Sesión",
        line=dict(color="#00ff41", width=1.8),
    ))

    # ── Total VWAP (white dashed) ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["vwap_total"],
        mode="lines", name="VWAP Total",
        line=dict(color="#ffffff", width=1, dash="dash"),
    ))

    # ── Price line (cyan) ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Precio",
        line=dict(color="#00ffff", width=1),
    ))

    # ── Session separators ────────────────────────────────────────────────────
    for st_time in session_starts[1:]:
        fig.add_vline(x=st_time, line=dict(color="#222222", width=1, dash="dot"))

    # ── SD labels on right side ───────────────────────────────────────────────
    last_ts = df.index[-1]
    last_vwap = float(df["vwap_session"].iloc[-1])
    last_sd   = float(df["session_sd"].iloc[-1])
    SD_COLOR  = {1: "#ff00ff", 2: "#ff8800", 3: "#ff4444", 4: "#aa00cc"}
    for n in [1, 2, 3, 4]:
        for sign, label in [(1, f"+{n} SD"), (-1, f"-{n} SD")]:
            fig.add_annotation(
                x=last_ts, y=last_vwap + sign * n * last_sd,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(color=SD_COLOR[n], size=9, family="monospace"),
                xanchor="left", xshift=8,
            )

    # ── Live metrics in title ─────────────────────────────────────────────────
    last_price = float(df["Close"].iloc[-1])
    last_time  = df.index[-1].strftime("%H:%M")
    vwap_sess  = float(df["vwap_session"].iloc[-1])
    vwap_tot   = float(df["vwap_total"].iloc[-1])

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=520,
        margin=dict(l=60, r=100, t=62, b=50),
        legend=dict(
            font=dict(color="#888", family="monospace", size=9),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#222",
            x=0.01, y=0.99,
        ),
        title=dict(
            text=(
                f"<b style='color:#00ff41'>VWAP + BANDAS SD  |  XAUUSD ({interval})</b><br>"
                f"<span style='font-size:10px;color:#888;'>"
                f"Hora: {last_time}  │  Precio: ${last_price:,.2f}"
                f"  │  VWAP Sesión: ${vwap_sess:,.2f}"
                f"  │  VWAP Total: ${vwap_tot:,.2f}"
                f"</span>"
            ),
            font=dict(color="#00ff41", family="monospace", size=12),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            color="#444", gridcolor="#111", showgrid=True,
            tickfont=dict(size=9, color="#666", family="monospace"),
        ),
        yaxis=dict(
            color="#444", gridcolor="#111", showgrid=True,
            tickfont=dict(size=9, color="#666", family="monospace"),
            tickformat="$,.2f",
        ),
    )
    return fig
