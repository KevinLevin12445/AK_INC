import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from engines.yf_cache import yf_download


def _download(ticker: str = "GC=F", interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    try:
        raw = yf_download(ticker, period=period, interval=interval,
                          auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.dropna(subset=["Close", "Volume", "Open", "High", "Low"])
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert(None)
        return raw
    except Exception:
        return pd.DataFrame()


def _session_volume_profile(
    session_df: pd.DataFrame,
    n_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Returns:
        bin_centers, bin_lo, bin_hi,
        buy_vol, sell_vol (per bin),
        poc_price, vah_price, val_price
    """
    prices   = session_df["Close"].values.astype(float)
    opens    = session_df["Open"].values.astype(float)
    highs    = session_df["High"].values.astype(float)
    lows     = session_df["Low"].values.astype(float)
    volumes  = session_df["Volume"].values.astype(float)

    p_min = lows.min()
    p_max = highs.max()
    if p_max <= p_min:
        p_max = p_min + 1.0

    bins        = np.linspace(p_min, p_max, n_bins + 1)
    bin_lo      = bins[:-1]
    bin_hi      = bins[1:]
    bin_centers = 0.5 * (bin_lo + bin_hi)

    buy_vol  = np.zeros(n_bins)
    sell_vol = np.zeros(n_bins)

    for i in range(n_bins):
        # Candles whose body overlaps this price bin
        mask = (lows <= bin_hi[i]) & (highs >= bin_lo[i])
        if not mask.any():
            continue
        v = volumes[mask]
        c = prices[mask]
        o = opens[mask]
        buy_vol[i]  = v[c >= o].sum()
        sell_vol[i] = v[c <  o].sum()

    total_vol = buy_vol + sell_vol
    poc_idx   = int(np.argmax(total_vol))
    poc_price = float(bin_centers[poc_idx])

    # Value area (70%)
    sorted_idx   = np.argsort(total_vol)[::-1]
    cum_vol      = np.cumsum(total_vol[sorted_idx])
    va_threshold = 0.70 * total_vol.sum()
    va_idx       = sorted_idx[: np.searchsorted(cum_vol, va_threshold) + 1]
    vah_price    = float(bin_centers[va_idx].max())
    val_price    = float(bin_centers[va_idx].min())

    return bin_centers, bin_lo, bin_hi, buy_vol, sell_vol, poc_price, vah_price, val_price


def build_volprofile_chart(
    ticker: str = "GC=F",
    interval: str = "15m",
    period: str = "5d",
    n_bins: int = 22,
    bar_time_fraction: float = 0.22,
) -> go.Figure:
    """
    Volume Profile Sessions chart:
    - Yellow price line
    - Per session: horizontal green/red bars (buy/sell vol per price bin)
    - POC (orange), VAH/VAL (white) horizontal lines per session
    - Session dividers
    """
    df = _download(ticker, interval, period)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#000000", height=560,
                          annotations=[dict(text="No intraday data available",
                                           x=0.5, y=0.5, xref="paper", yref="paper",
                                           font=dict(color="#00ff41", size=14))])
        return fig

    df["date"] = df.index.date
    sessions   = sorted(df["date"].unique())

    fig = go.Figure()

    # ── Yellow price line ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="XAUUSD",
        line=dict(color="#ffd700", width=1.2),
    ))

    # ── Per-session volume profile ────────────────────────────────────────────
    for day in sessions:
        sdf = df[df["date"] == day]
        if len(sdf) < 4:
            continue

        t_start = sdf.index[0]
        t_end   = sdf.index[-1]
        span    = (t_end - t_start).total_seconds()
        if span <= 0:
            continue

        bin_centers, bin_lo, bin_hi, buy_vol, sell_vol, poc, vah, val = \
            _session_volume_profile(sdf, n_bins=n_bins)

        total_vol = buy_vol + sell_vol
        max_vol   = total_vol.max()
        if max_vol == 0:
            continue

        # Bar width = fraction of session span (in seconds → timedelta)
        max_bar_sec = span * bar_time_fraction
        bar_width   = pd.Timedelta(seconds=max_bar_sec)

        # Draw bars as rectangles from t_start
        for i in range(n_bins):
            tv = total_vol[i]
            if tv == 0:
                continue
            bv = buy_vol[i]
            sv = sell_vol[i]
            bar_len = bar_width * (tv / max_vol)
            color   = "#00cc44" if bv >= sv else "#cc2222"  # green buy, red sell

            fig.add_shape(
                type="rect",
                x0=t_start,
                x1=t_start + bar_len,
                y0=float(bin_lo[i]),
                y1=float(bin_hi[i]),
                fillcolor=color.replace("#", "rgba(") + ",0.55)" if False else color,
                line=dict(width=0),
                opacity=0.55,
            )

        # POC line (orange) across full session
        fig.add_shape(
            type="line",
            x0=t_start, x1=t_end,
            y0=poc, y1=poc,
            line=dict(color="#ff8800", width=1.5, dash="solid"),
        )

        # VAH line (white)
        fig.add_shape(
            type="line",
            x0=t_start, x1=t_end,
            y0=vah, y1=vah,
            line=dict(color="#ffffff", width=1, dash="solid"),
        )

        # VAL line (white)
        fig.add_shape(
            type="line",
            x0=t_start, x1=t_end,
            y0=val, y1=val,
            line=dict(color="#ffffff", width=1, dash="solid"),
        )

        # POC annotation
        fig.add_annotation(
            x=t_end,
            y=poc,
            text=f"POC ${poc:,.0f}",
            showarrow=False,
            font=dict(color="#ff8800", size=8, family="monospace"),
            xanchor="left", xshift=4,
        )

        # Session separator
        fig.add_vline(
            x=t_start,
            line=dict(color="#222222", width=1, dash="dot"),
        )

    # ── Legend traces for POC/VAH/VAL ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name="POC", line=dict(color="#ff8800", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name="VAH/VAL", line=dict(color="#ffffff", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        name="Buy vol", marker=dict(color="#00cc44", size=8, symbol="square")
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        name="Sell vol", marker=dict(color="#cc2222", size=8, symbol="square")
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    last_price = float(df["Close"].iloc[-1])
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=560,
        margin=dict(l=70, r=60, t=54, b=50),
        title=dict(
            text=(
                f"<b style='color:#00ff41'>VOLUME PROFILE SESSIONS</b>"
                f"<span style='color:#888;font-size:10px;'>"
                f"  ·  {ticker} ({interval})  ·  Precio: ${last_price:,.2f}"
                f"  ·  barras = buy(verde)/sell(rojo)  ·  POC=naranja  VAH/VAL=blanco"
                f"</span>"
            ),
            font=dict(color="#00ff41", family="monospace", size=12),
            x=0, xanchor="left",
        ),
        legend=dict(
            font=dict(color="#888", family="monospace", size=9),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#222",
            x=0.01, y=0.99,
        ),
        xaxis=dict(
            color="#333", gridcolor="#111",
            tickfont=dict(size=9, color="#555", family="monospace"),
            showgrid=True,
        ),
        yaxis=dict(
            color="#333", gridcolor="#111",
            tickfont=dict(size=9, color="#555", family="monospace"),
            tickformat="$,.0f",
            showgrid=True,
        ),
    )
    return fig
