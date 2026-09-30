"""Interactive Nifty Analyst explorer.

Run:  streamlit run app.py
Lets you play with indicator settings, analogue count, lookback and date range
and see similar-day detection + next-day outcome stats update live.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core.data as data  # noqa: E402
import core.settings as st_mod  # noqa: E402
from analysis import intraday  # noqa: E402
from analysis import patterns  # noqa: E402
from core import session  # noqa: E402

st.set_page_config(page_title="Nifty Analyst Explorer", layout="wide")

settings = st_mod.load_settings()


def render_intraday(settings, tf_label: str):
    import numpy as np

    import plotly.graph_objects as go  # noqa: F401
    from analysis import patterns  # noqa: F811

    hours = tf_label.endswith("h")
    minutes = int(tf_label[:-1]) * (60 if hours else 1)
    st.title(f"Nifty Analyst — intraday explorer ({tf_label})")
    bars = session.archive_bars("nifty", minutes, settings)
    if bars.empty:
        st.error("No intraday archive yet. Run `python scripts/update_data.py --intraday` first.")
        return
    daily = data.load_series("nifty", settings)
    if daily.empty:
        st.error("Daily nifty series missing (run scripts/update_data.py).")
        return

    features = settings["intraday"]["features"]
    weights = settings["intraday"]["weights"]
    cfg = settings["intraday"]

    with st.sidebar:
        k = st.slider("Analogue count (K)", 20, 400, cfg["k"], 10)
        look = st.slider("Z-score lookback (bars)", 100, 1500, cfg["standardize_lookback"], 50)

    @st.cache_data
    def compute(bars_key, daily_key, minutes, k, look):
        frame = intraday.build_features(bars, daily, settings)
        z = patterns.zscore_point_in_time(frame, features, look)
        session_dates = sorted({d.isoformat() for d in frame.index.date})
        return frame, z, session_dates

    bars_key = f"{len(bars)}-{bars.index[-1]}"
    daily_key = f"{len(daily)}-{daily.index[-1]}"
    frame, z, session_dates = compute(bars_key, daily_key, minutes, k, look)
    chosen = st.sidebar.selectbox(
        "Session", list(reversed(session_dates[:80])),
        index=0, format_func=lambda s: s[:10],
    )
    pos = int(np.flatnonzero(frame.index.date.astype(str) == chosen)[-1])
    sim, dist = patterns.find_analogues(
        frame, z, features, weights, pos, k, look, session_col="session"
    )
    if not sim:
        st.info("No analogues at this position yet.")
        return
    cohort = frame.iloc[sim]
    stats = intraday.cohort_metrics(cohort, cfg["horizon_bars"], cfg["primary_target"])
    stats["analogue_count"] = len(cohort)

    day = frame[frame.index.date.astype(str) == chosen]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Similar bars", len(cohort))
    c2.metric("Prob rest up", f"{stats['direction']['prob_up']*100:.1f}%")
    c3.metric("Median rest", f"{stats[cfg['primary_target']]['median']:+.2f}%")
    c4.metric("Median next bar", f"{stats.get('fwd_ret_1', {}).get('median', float('nan')):+.2f}%")

    fig = go.Figure(go.Candlestick(
        x=day.index, open=day["open"], high=day["high"], low=day["low"], close=day["close"], name="OHLC",
        increasing_line_color="#0a8f3c", decreasing_line_color="#d13438",
    ))
    pts = frame.iloc[sim]
    fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name=f"Top {len(pts)} analogues",
                             marker=dict(color="#1f6feb", size=4, opacity=0.5), yaxis="y"))
    fig.add_trace(go.Scatter(x=[day.index[-1]], y=[day["close"].iloc[-1]], mode="markers",
                             name="Position", marker=dict(color="#d13438", size=10, symbol="star")))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, title=f"Session {chosen} ({tf_label})")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Forward outcomes of similar bars")
    st.dataframe(pd.DataFrame({
        col: {"n": stats[col]["count"], "mean": round(stats[col]["mean"], 3),
              "median": round(stats[col]["median"], 3), "p5": round(stats[col]["p5"], 3),
              "p25": round(stats[col]["p25"], 3), "p75": round(stats[col]["p75"], 3),
              "p95": round(stats[col]["p95"], 3)}
        for col in [cfg["primary_target"], "fwd_ret_1"] + [f"fwd_ret_{h}" for h in cfg["horizon_bars"]]
        if col in stats
    }).T, width="stretch")

    st.subheader("Closest bars")
    view = cohort.copy().sort_index()
    view["ts"] = view.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(view[["ts", "fwd_ret_1", cfg["primary_target"]]].tail(30), width="stretch")


tf_choice = st.sidebar.radio("Timeframe", ["Daily", "5m", "15m", "30m", "1h"], index=0)

if tf_choice != "Daily":
    render_intraday(settings, tf_choice)
    st.stop()

df = data.load_series("nifty", settings)
if df.empty:
    st.error("No cached data. Run `python scripts/update_data.py` first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    k = st.slider(
        "Analogue count (K)",
        settings["analogues"]["k_min"], settings["analogues"]["k_max"],
        settings["analogues"]["k"], 10,
    )
    look = st.slider(
        "Z-score lookback (days)", 100, 1000,
        settings["analogues"]["standardize_lookback"], 25,
    )
    show_ma = st.multiselect(
        "MA overlays", settings["indicators"]["ma_periods"],
        default=list(settings["indicators"]["ma_periods"]),
    )
    show_rsi = st.toggle("Show RSI chart", value=True)
    show_vol = st.toggle("Show volatility chart", value=True)
    
    st.divider()
    st.header("Feature Selection")
    sel_mode = st.radio("Selection Mode", ["Manual", "Auto-select (Walk-forward)"], index=1)
    
    all_features = settings["analogues"]["features"]
    if sel_mode == "Manual":
        active_features = st.multiselect("Active features", all_features, default=all_features)
        backtest_days = 90
        top_n = 5
    else:
        backtest_days = st.slider("Backtest lookback (days)", 30, 150, 90, 10)
        top_n = st.slider("Top N features to select", 1, len(all_features), 5, 1)
        active_features = []


@st.cache_data
def get_base_data(ma_periods):
    ma_periods = tuple(ma_periods) if ma_periods else None
    frame = patterns.build_features(df, settings, ma_periods=ma_periods)
    avail = [f for f in settings["analogues"]["features"] if f in frame.columns]
    return frame, avail

@st.cache_data
def get_zscores(frame, avail, look):
    return patterns.zscore_point_in_time(frame, avail, look)

@st.cache_data
def do_auto_select(frame, z, avail, k, look, backtest_days, top_n):
    weights = settings["analogues"]["weights"]
    target_pos = len(frame) - 1
    return patterns.auto_select_features(frame, z, avail, weights, target_pos, k, look, backtest_days, top_n)

@st.cache_data
def get_analogues(frame, z, active, k, look):
    weights = settings["analogues"]["weights"]
    target_pos = len(frame) - 1
    return patterns.find_analogues(frame, z, active, weights, target_pos, k, look)

frame, avail_features = get_base_data(tuple(show_ma))
z = get_zscores(frame, avail_features, look)

if sel_mode == "Auto-select (Walk-forward)":
    with st.sidebar:
        with st.spinner("Running walk-forward backtest..."):
            best_feats, scores = do_auto_select(frame, z, avail_features, k, look, backtest_days, top_n)
            active_features = best_feats
        st.success("Auto-selection complete!")
        st.write("**Selected Indicators:**")
        for f in best_feats:
            st.write(f"- {f} (Edge: {scores[f]*100:.1f}%)")

if not active_features:
    active_features = avail_features[:5]

idxs, dists = get_analogues(frame, z, active_features, k, look)
cohort = frame.iloc[idxs].copy()
cohort["distance"] = dists
stats = patterns.outcome_stats(cohort)
stats["analogue_count"] = len(cohort)
scen = patterns.scenarios(stats, settings["sentiment"])
today = frame.iloc[-1]

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

st.title("Nifty Analyst — explorer")
d = stats["direction"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Similar setups", stats["analogue_count"])
c2.metric("Next-day positive", d["up"])
c3.metric("Next-day negative", d["down"])
c4.metric("Prob up", f"{d['prob_up']*100:.1f}%")
c5.metric("Median next move", f"{stats['nxt_ret']['median']:+.2f}%")
c6.metric("Stat. bias", scen["bias"])

st.subheader("Price + analogues")
if idxs:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=frame.index[-400:], open=frame["open"][-400:], high=frame["high"][-400:],
        low=frame["low"][-400:], close=frame["close"][-400:], name="OHLC"), row=1, col=1)
    for n in show_ma:
        fig.add_trace(go.Scatter(x=frame.index[-400:], y=frame[f"sma_{n}"][-400:],
                                 mode="lines", name=f"SMA {n}"), row=1, col=1)
    pts = frame.iloc[idxs]
    fig.add_trace(go.Scatter(x=pts.index, y=pts["close"], mode="markers", name=f"Top {len(pts)} analogues",
                             marker=dict(color="#1f6feb", size=5, opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[frame.index[-1]], y=[frame["close"].iloc[-1]], mode="markers",
                             name="Today", marker=dict(color="#d13438", size=11, symbol="star")), row=1, col=1)
    fig.add_trace(go.Bar(x=frame.index[-400:], y=frame["vol_20d"][-400:], name="Vol 20d"), row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, showlegend=True)
    st.plotly_chart(fig, width="stretch")

st.subheader("Today's setup")
st.dataframe(pd.DataFrame({
    "close": [today["close"]], "rsi": [round(today["rsi_14"], 1)],
    "atr%": [round(today["atr_pct"], 2)], "gap%": [round(today["gap_pct"], 2)],
    "range%": [round(today["range_pct"], 2)], "vol20": [round(today["vol_20d"], 1)],
    "ret1d%": [round(today["ret_1d"], 2)], "ret5d%": [round(today["ret_5d"], 2)],
    "ret20d%": [round(today["ret_20d"], 2)],
}), width="stretch")

st.subheader("Next-day outcome distribution of analogues")
grid = pd.DataFrame({
    col: {
        "n": stats[col]["count"], "mean": round(stats[col]["mean"], 3),
        "median": round(stats[col]["median"], 3), "p5": round(stats[col]["p5"], 3),
        "p25": round(stats[col]["p25"], 3), "p75": round(stats[col]["p75"], 3),
        "p95": round(stats[col]["p95"], 3),
    }
    for col in ["nxt_ret", "nxt_open_gap", "nxt_high_pct", "nxt_low_pct", "nxt_mad", "nxt_maf"]
}).T
st.dataframe(grid)

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Similar day next-day returns")
    hist = go.Figure(go.Histogram(x=cohort["nxt_ret"].dropna(), nbinsx=36, marker_color="#8fb0e0"))
    hist.add_vline(x=0, line_color="black")
    st.plotly_chart(hist, width="stretch")
with col_r:
    st.subheader("Scenarios")
    st.write("**Bullish** (prob {:.0f}%): P75 target {:.2f}%".format(
        scen["bullish_scenario"]["prob"] * 100, scen["bullish_scenario"]["target_median_ret"]))
    st.write("**Neutral**: band P25–P75  [{:.2f}%, {:.2f}%]".format(*scen["median_band"]))
    st.write("**Bearish** (prob {:.0f}%): P25 target {:.2f}%".format(
        scen["bearish_scenario"]["prob"] * 100, scen["bearish_scenario"]["target_median_ret"]))
    st.caption("Pure historical statistics — interpretation is added by the Nifty Analyst agent.")

if show_rsi:
    st.subheader("RSI")
    rs = go.Figure(go.Scatter(x=frame.index[-800:], y=frame["rsi_14"][-800:], name="RSI"))
    rs.add_hline(y=70, line_dash="dash", line_color="red")
    rs.add_hline(y=30, line_dash="dash", line_color="green")
    rs.update_yaxes(range=[0, 100])
    st.plotly_chart(rs, width="stretch")

st.subheader("Analogue list (top 30)")
st.dataframe(
    cohort[[*active_features[:6], "distance"]].assign(**{"date": cohort.index.strftime("%Y-%m-%d")})
    .sort_values("distance").head(30),
    width="stretch",
)
