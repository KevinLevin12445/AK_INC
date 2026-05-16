import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import List
import warnings
warnings.filterwarnings("ignore")

RG_SCALE = [
    [0.00, "#8b0000"],
    [0.15, "#cc2200"],
    [0.30, "#dd4400"],
    [0.42, "#441100"],
    [0.50, "#0a0a0a"],
    [0.58, "#003311"],
    [0.70, "#007722"],
    [0.85, "#00aa33"],
    [1.00, "#00ff55"],
]

GOLD_COLOR_POSITIVE = "#ffd700"
GOLD_COLOR_NEGATIVE = "#cc8800"


def _clamp_color(val: float, lo: float = -3.0, hi: float = 3.0) -> float:
    return max(lo, min(hi, val))


def build_equity_treemap(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#000000", height=420,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                           xref="paper", yref="paper",
                                           font=dict(color="#00ff41", size=14))])
        return fig

    # Gold rows get special treatment
    has_gold_col = "is_gold" in df.columns

    labels, parents, values, colors, hovers, text_colors = (
        ["WORLD"], [""], [df["market_cap"].sum()], [0.0],
        ["<b>WORLD</b>"], ["#ffffff"],
    )

    for sector in df["sector"].unique():
        sdf      = df[df["sector"] == sector]
        sec_cap  = sdf["market_cap"].sum()
        sec_chg  = float((sdf["change_pct"] * sdf["market_cap"]).sum() / sec_cap) if sec_cap else 0.0
        labels.append(sector)
        parents.append("WORLD")
        values.append(sec_cap)
        colors.append(_clamp_color(sec_chg))
        hovers.append(f"<b>{sector}</b><br>Wtd Δ: {sec_chg:+.2f}%")
        text_colors.append("#ffd700" if sector == "Gold & Precious" else "#ffffff")

        for _, row in sdf.iterrows():
            chg_pct  = float(row["change_pct"])
            is_gold  = bool(row.get("is_gold", False)) if has_gold_col else False
            ticker   = row["ticker"]
            name     = row["name"]
            price    = row["price"]

            labels.append(ticker)
            parents.append(sector)
            values.append(float(row["market_cap"]))
            colors.append(_clamp_color(chg_pct))
            text_colors.append(GOLD_COLOR_POSITIVE if is_gold else "#ffffff")

            gold_tag = "★ GOLD · " if is_gold else ""
            hovers.append(
                f"<b>{gold_tag}{ticker}</b> — {name}<br>"
                f"Price: ${price:,.2f}<br>"
                f"Δ: {chg_pct:+.2f}%<br>"
                f"Est. MktCap: ${row['market_cap']}B"
            )

    norm_colors = [(c + 3.0) / 6.0 for c in colors]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        customdata=hovers,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(
            colors=norm_colors,
            colorscale=RG_SCALE,
            cmin=0, cmax=1,
            line=dict(color="#000000", width=1.5),
            showscale=True,
            colorbar=dict(
                title=dict(text="Δ%", font=dict(color="#888", size=9)),
                tickvals=[0, 0.5, 1.0],
                ticktext=["-3%", "0%", "+3%"],
                tickfont=dict(color="#888", size=9),
                len=0.6, thickness=10, x=1.01,
                bgcolor="#000000", bordercolor="#222",
            ),
        ),
        texttemplate=(
            "<b>%{label}</b><br>"
            "<span style='font-size:10px'>%{percentRoot:.1%}</span>"
        ),
        textfont=dict(family="monospace", size=11, color="#ffffff"),
        branchvalues="total",
        pathbar=dict(
            visible=True,
            thickness=20,
            textfont=dict(color="#00ff41", family="monospace", size=10),
        ),
        tiling=dict(packing="squarify", pad=1),
    ))

    # Gold banner metric at top-right
    gold_rows = df[df.get("is_gold", pd.Series(False, index=df.index))] if has_gold_col else pd.DataFrame()
    if not gold_rows.empty:
        gcf = gold_rows.iloc[0]
        chg = float(gcf["change_pct"])
        px  = float(gcf["price"])
        arrow  = "▲" if chg >= 0 else "▼"
        gcolor = GOLD_COLOR_POSITIVE if chg >= 0 else GOLD_COLOR_NEGATIVE
        gold_title_part = (
            f"<span style='color:{gcolor};font-size:13px;'>"
            f"  ★ XAUUSD ${px:,.2f}  {arrow}{chg:+.2f}%"
            f"</span>"
        )
    else:
        gold_title_part = ""

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=440,
        margin=dict(l=5, r=5, t=44, b=5),
        title=dict(
            text=(
                "<b style='color:#00ff41'>EQUITY HEATMAP</b>"
                "<span style='color:#444;font-size:10px'>  ·  size=market cap  ·  color=Δ%</span>"
                + gold_title_part
            ),
            font=dict(color="#00ff41", family="monospace", size=12),
            x=0, xanchor="left",
        ),
    )
    return fig


def build_geo_map(df: pd.DataFrame, gold_change_pct: float = 0.0, gold_price: float = 0.0) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#000000", height=420,
                          annotations=[dict(text="No data", x=0.5, y=0.5,
                                           xref="paper", yref="paper",
                                           font=dict(color="#00ff41", size=14))])
        return fig

    df = df.copy()
    df["norm"] = (df["change_pct"].clip(-3, 3) + 3) / 6.0

    choro_colors = [
        [0.00, "#8b0000"], [0.15, "#cc2200"], [0.30, "#dd4400"],
        [0.42, "#441100"], [0.50, "#111111"], [0.58, "#003311"],
        [0.70, "#007722"], [0.85, "#00aa33"], [1.00, "#00ff55"],
    ]

    fig = go.Figure()

    # ① Choropleth base layer — country ETF performance
    fig.add_trace(go.Choropleth(
        locations=df["iso3"],
        z=df["norm"],
        text=df.apply(
            lambda r: f"<b>{r['country']}</b><br>{r['ticker']}: {r['change_pct']:+.2f}%",
            axis=1
        ),
        hovertemplate="%{text}<extra></extra>",
        colorscale=choro_colors,
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="ETF Δ%", font=dict(color="#888", size=9)),
            tickvals=[0, 0.5, 1.0],
            ticktext=["-3%", "0%", "+3%"],
            tickfont=dict(color="#888", size=9),
            len=0.5, thickness=10, x=1.0,
            bgcolor="#000000", bordercolor="#222",
        ),
        marker=dict(line=dict(color="#1a1a1a", width=0.5)),
    ))

    # ② Country ETF bubbles (size = abs move)
    max_abs = df["change_pct"].abs().max() or 1.0
    df["bubble_sz"] = (df["change_pct"].abs() / max_abs * 18 + 4).clip(4, 22)

    fig.add_trace(go.Scattergeo(
        locations=df["iso3"],
        text=df["change_pct"].apply(lambda v: f"{v:+.2f}%"),
        mode="markers+text",
        marker=dict(
            size=df["bubble_sz"],
            color=df["norm"],
            colorscale=choro_colors,
            cmin=0, cmax=1,
            line=dict(color="#000000", width=1),
            opacity=0.82,
        ),
        textfont=dict(color="#ffffff", family="monospace", size=8),
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
        name="ETF",
    ))

    # ③ Gold producer overlay — gold-colored stars on major producing nations
    GOLD_PRODUCERS = {
        "CHN": ("China",         380, (35.0,  105.0)),
        "AUS": ("Australia",     310, (-25.0, 133.0)),
        "RUS": ("Russia",        300, (61.0,   95.0)),
        "CAN": ("Canada",        180, (56.0,  -96.0)),
        "USA": ("United States", 170, (37.0, -100.0)),
        "ZAF": ("South Africa",  100, (-29.0,  25.0)),
        "PER": ("Peru",          100, (-9.0,  -75.0)),
        "GHA": ("Ghana",          80, (7.9,    -1.0)),
        "IDN": ("Indonesia",      80, (-2.5,  117.0)),
        "UZB": ("Uzbekistan",     60, (41.0,   64.0)),
    }

    g_lats  = [v[2][0] for v in GOLD_PRODUCERS.values()]
    g_lons  = [v[2][1] for v in GOLD_PRODUCERS.values()]
    g_sizes = [max(8, min(26, v[1] / 15)) for v in GOLD_PRODUCERS.values()]
    g_texts = [
        f"<b>{v[0]}</b><br>Gold prod: ~{v[1]}t/yr"
        for v in GOLD_PRODUCERS.values()
    ]
    g_arrow = "▲" if gold_change_pct >= 0 else "▼"
    g_color = GOLD_COLOR_POSITIVE if gold_change_pct >= 0 else GOLD_COLOR_NEGATIVE

    fig.add_trace(go.Scattergeo(
        lat=g_lats,
        lon=g_lons,
        mode="markers",
        marker=dict(
            size=g_sizes,
            color=g_color,
            symbol="star",
            line=dict(color="#000000", width=0.5),
            opacity=0.90,
        ),
        text=g_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Gold Producers",
        showlegend=True,
    ))

    # ④ XAUUSD banner annotation on the map
    gold_banner = (
        f"★ XAUUSD  ${gold_price:,.2f}  {g_arrow}{gold_change_pct:+.2f}%"
        if gold_price > 0 else "★ XAUUSD"
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        geo=dict(
            bgcolor="#000000",
            landcolor="#111111",
            oceancolor="#050510",
            lakecolor="#050510",
            countrycolor="#1a1a1a",
            coastlinecolor="#1a1a1a",
            showland=True, showocean=True,
            showlakes=True, showcoastlines=True,
            showcountries=True,
            projection_type="natural earth",
            framecolor="#1a1a1a",
        ),
        height=420,
        margin=dict(l=0, r=0, t=44, b=0),
        legend=dict(
            x=0.01, y=0.01,
            font=dict(color="#888", family="monospace", size=9),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#222",
        ),
        title=dict(
            text=(
                "<b style='color:#00ff41'>GLOBAL CAPITAL MAP</b>"
                "<span style='color:#444;font-size:10px'>  ·  country ETF performance</span>"
                f"<span style='color:{g_color};font-size:12px;'>"
                f"    {gold_banner}</span>"
            ),
            font=dict(color="#00ff41", family="monospace", size=12),
            x=0, xanchor="left",
        ),
    )
    return fig


def render_alert_row(alert) -> str:
    color_map = {"HIGH": "#ff4444", "MEDIUM": "#ffd700", "LOW": "#00aaff"}
    bg_map    = {"HIGH": "rgba(80,0,0,0.35)", "MEDIUM": "rgba(60,50,0,0.35)", "LOW": "rgba(0,30,60,0.35)"}
    type_icon = {"price": "💹", "volatility": "⚡", "insider": "👤", "news": "📰"}
    color = color_map.get(alert.severity, "#888888")
    bg    = bg_map.get(alert.severity, "rgba(20,20,20,0.5)")
    icon  = type_icon.get(alert.alert_type, "•")
    return (
        f"<div style='padding:6px 8px;margin-bottom:4px;border-left:3px solid {color};"
        f"background:{bg};border-radius:2px;font-family:monospace;font-size:11px;line-height:1.4;'>"
        f"<span style='color:{color};font-weight:bold;'>[{alert.severity}]</span> "
        f"<span style='color:#888;'>{icon} {alert.alert_type.upper()} · {alert.timestamp}</span><br>"
        f"<span style='color:#cccccc;'>{alert.message}</span>"
        f"</div>"
    )


def render_news_item(item: dict) -> str:
    title = (
        (item.get("content") or {}).get("title")
        or item.get("title")
        or item.get("headline", "—")
    )
    publisher = (
        ((item.get("content") or {}).get("provider") or {}).get("displayName")
        or item.get("publisher")
        or item.get("source", "—")
    )
    link = (
        ((item.get("content") or {}).get("clickThroughUrl") or {}).get("url")
        or item.get("link", "#")
    )
    short = title[:90] + "…" if len(title) > 90 else title
    return (
        f"<div style='padding:5px 0;border-bottom:1px solid #111;font-family:monospace;font-size:10px;'>"
        f"<a href='{link}' target='_blank' style='color:#aaaaaa;text-decoration:none;'>{short}</a>"
        f"<span style='color:#444;'> · {publisher}</span>"
        f"</div>"
    )


def render_map_tab(
    sector_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    alerts: list,
    news_items: list,
    gold_change_pct: float = 0.0,
    gold_price: float = 0.0,
):
    left, right = st.columns([0.72, 0.28], gap="small")

    with left:
        fig_tree = build_equity_treemap(sector_df)
        st.plotly_chart(fig_tree, width="stretch")

        fig_geo = build_geo_map(geo_df, gold_change_pct=gold_change_pct, gold_price=gold_price)
        st.plotly_chart(fig_geo, width="stretch")

    with right:
        st.markdown(
            "<div style='font-family:monospace;font-size:11px;color:#00ff41;"
            "letter-spacing:2px;margin-bottom:8px;border-bottom:1px solid #003300;"
            "padding-bottom:4px;'>⚡ LIVE ALERTS</div>",
            unsafe_allow_html=True,
        )
        if alerts:
            html_alerts = "".join(render_alert_row(a) for a in alerts)
            st.markdown(html_alerts, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#333;font-family:monospace;font-size:10px;padding:8px;'>"
                "No active alerts</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='font-family:monospace;font-size:11px;color:#00ff41;"
            "letter-spacing:2px;margin:14px 0 8px;border-bottom:1px solid #003300;"
            "padding-bottom:4px;'>📰 GOLD NEWS FEED</div>",
            unsafe_allow_html=True,
        )
        if news_items:
            news_html = "".join(render_news_item(n) for n in news_items[:10])
            st.markdown(
                f"<div style='max-height:420px;overflow-y:auto;'>{news_html}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='color:#333;font-family:monospace;font-size:10px;padding:8px;'>"
                "No news available</div>",
                unsafe_allow_html=True,
            )
