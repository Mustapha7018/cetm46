import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config 
st.set_page_config(
    page_title="Global Climate Change Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data loading 
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.strip()
    df["Year"] = df["Year"].astype(int)
    return df

df_raw = load_data()

CONTINUOUS_COLS = [
    "Avg Temperature (°C)",
    "CO2 Emissions (Tons/Capita)",
    "Sea Level Rise (mm)",
    "Rainfall (mm)",
    "Population",
    "Renewable Energy (%)",
    "Extreme Weather Events",
    "Forest Area (%)",
]

METRIC_LABELS = {
    "CO2 Emissions (Tons/Capita)": "CO2 Emissions",
    "Avg Temperature (°C)": "Avg Temperature",
    "Renewable Energy (%)": "Renewable Energy",
    "Sea Level Rise (mm)": "Sea Level Rise",
    "Forest Area (%)": "Forest Area",
    "Extreme Weather Events": "Extreme Weather Events",
    "Rainfall (mm)": "Rainfall",
}

COLOR_SCALES = {
    "CO2 Emissions (Tons/Capita)": "Reds",
    "Avg Temperature (°C)": "RdYlBu_r",
    "Renewable Energy (%)": "Greens",
    "Sea Level Rise (mm)": "Blues",
    "Forest Area (%)": "Greens",
    "Extreme Weather Events": "Oranges",
    "Rainfall (mm)": "Blues",
}

# Session state 
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

# Sidebar 
with st.sidebar:
    st.markdown("## Dashboard Filters")

    _tcol_label, _tcol_toggle = st.columns([5, 1])
    with _tcol_label:
        st.markdown("Dark Mode")
    with _tcol_toggle:
        dark_mode = st.toggle("", value=st.session_state["dark_mode"], key="dark_toggle", label_visibility="collapsed")
    st.session_state["dark_mode"] = dark_mode

    st.markdown("---")

    year_min, year_max = int(df_raw["Year"].min()), int(df_raw["Year"].max())
    year_range = st.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )

    all_countries = sorted(df_raw["Country"].unique().tolist())
    ALL_OPTION = "All Countries"
    country_options = [ALL_OPTION] + all_countries

    raw_selection = st.multiselect(
        "Countries",
        options=country_options,
        default=[ALL_OPTION],
        placeholder="Select countries…",
    )

    if ALL_OPTION in raw_selection or not raw_selection:
        selected_countries = all_countries
    else:
        selected_countries = raw_selection

    primary_metric = st.selectbox(
        "Primary Metric (Choropleth / KPI focus)",
        options=list(METRIC_LABELS.keys()),
        index=0,
    )

    top_n = st.slider("Top N Countries", min_value=5, max_value=15, value=10)

    st.markdown("---")
    st.markdown(
        "<small>Data: Global Climate Indicators 2000–2024</small>",
        unsafe_allow_html=True,
    )

# Theme 
if dark_mode:
    T = {
        "plotly": "plotly_dark",
        "plot_bg": "rgba(13,33,55,0.6)",
        "grid": "#1e3a5f",
        "card_bg": "linear-gradient(135deg,#1e3a5f 0%,#0d2137 100%)",
        "card_border": "#2d5986",
        "card_label": "#8ab4d4",
        "card_value": "#ffffff",
        "delta_pos": "#4ade80",
        "delta_neg": "#f87171",
        "sidebar_bg": "#0d2137",
        "body_bg": "#0a1628",
        "text": "#e2f0fb",
        "subtext": "#6b8fa8",
        "section_accent": "#3b82f6",
        "geo_bg": "rgba(13,33,55,0.6)",
        "geo_land": "rgba(30,58,92,0.5)",
        "geo_coast": "#2d5986",
        "input_bg": "#0d2137",
        "hoverlabel_bg": "#1e3a5f",
        "hoverlabel_font": "#e2f0fb",
        "hoverlabel_border": "#2d5986",
    }
else:
    T = {
        "plotly": "plotly_white",
        "plot_bg": "rgba(240,246,255,0.85)",
        "grid": "#c7d9f0",
        "card_bg": "linear-gradient(135deg,#e8f4fd 0%,#d6eeff 100%)",
        "card_border": "#93c5fd",
        "card_label": "#1e5f8a",
        "card_value": "#0d2137",
        "delta_pos": "#16a34a",
        "delta_neg": "#dc2626",
        "sidebar_bg": "#e0eefa",
        "body_bg": "#f0f6ff",
        "text": "#0d2137",
        "subtext": "#2d5986",
        "section_accent": "#2563eb",
        "geo_bg": "rgba(240,246,255,0.8)",
        "geo_land": "rgba(195,220,255,0.6)",
        "geo_coast": "#93c5fd",
        "input_bg": "#ffffff",
        "hoverlabel_bg": "#ffffff",
        "hoverlabel_font": "#0d2137",
        "hoverlabel_border": "#93c5fd",
    }

# CSS injection 
st.markdown(f"""
<style>
  /* ── global ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {T['body_bg']} !important;
      color: {T['text']} !important;
  }}
  [data-testid="stAppViewContainer"] > .main {{
      background-color: {T['body_bg']} !important;
  }}
  /* ── sidebar ── */
  section[data-testid="stSidebar"] {{
      background: {T['sidebar_bg']} !important;
  }}
  section[data-testid="stSidebar"] * {{
      color: {T['text']} !important;
  }}
  /* ── metric cards ── */
  .metric-card {{
      background: {T['card_bg']};
      border: 1px solid {T['card_border']};
      border-radius: 12px;
      padding: 16px 20px;
      text-align: center;
  }}
  .metric-label  {{ color: {T['card_label']}; font-size: 13px; font-weight: 600; letter-spacing: .5px; }}
  .metric-value  {{ color: {T['card_value']}; font-size: 28px; font-weight: 700; margin: 4px 0; }}
  .metric-delta-pos {{ color: {T['delta_pos']}; font-size: 13px; }}
  .metric-delta-neg {{ color: {T['delta_neg']}; font-size: 13px; }}
  /* ── section headers ── */
  .section-header {{
      border-left: 4px solid {T['section_accent']};
      padding-left: 12px;
      margin: 32px 0 12px 0;
      color: {T['text']};
      font-size: 20px;
      font-weight: 600;
  }}
  /* ── headings & captions ── */
  h1, h2, h3, h4 {{ color: {T['text']} !important; }}
  [data-testid="stCaptionContainer"] p {{ color: {T['subtext']} !important; }}
  p {{ color: {T['text']}; }}
  /* ── plotly: all axis ticks, titles, legends, colorbars (primary + secondary) ── */
  .stPlotlyChart svg .xtick text,
  .stPlotlyChart svg .ytick text,
  .stPlotlyChart svg .x2tick text,
  .stPlotlyChart svg .y2tick text,
  .stPlotlyChart svg .x3tick text,
  .stPlotlyChart svg .y3tick text,
  .stPlotlyChart svg .g-xtitle text,
  .stPlotlyChart svg .g-ytitle text,
  .stPlotlyChart svg .g-x2title text,
  .stPlotlyChart svg .g-y2title text,
  .stPlotlyChart svg .g-x3title text,
  .stPlotlyChart svg .g-y3title text,
  .stPlotlyChart svg .legendtext,
  .stPlotlyChart svg .cbaxis text,
  .stPlotlyChart svg .cbtitle text,
  .stPlotlyChart svg .colorbar text {{
      fill: {T['text']} !important;
  }}
  /* ── sidebar columns: vertically center toggle row ── */
  section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
      align-items: center !important;
  }}
</style>
""", unsafe_allow_html=True)

if not dark_mode:
    st.markdown("""
<style>
  /* ── light mode: select/multiselect box background ── */
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
      background-color: #ffffff !important;
      border-color: #93c5fd !important;
  }
  /* search input text */
  section[data-testid="stSidebar"] [data-baseweb="select"] input {
      color: #0d2137 !important;
      background-color: transparent !important;
  }
  /* placeholder / hint text */
  section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
      color: #2d5986 !important;
      opacity: 1 !important;
  }
  /* selected single value text (selectbox) */
  section[data-testid="stSidebar"] [data-baseweb="select"] [data-id="placeholder"],
  section[data-testid="stSidebar"] [data-baseweb="select"] > div > div > div:not([data-baseweb="tag"]) {
      color: #0d2137 !important;
  }
</style>
""", unsafe_allow_html=True)

# Filter dataframe
df = df_raw[
    (df_raw["Year"] >= year_range[0])
    & (df_raw["Year"] <= year_range[1])
    & (df_raw["Country"].isin(selected_countries))
].copy()

# Apply top-N filter globally across all charts
if df["Country"].nunique() > top_n:
    _top_n_countries = (
        df.groupby("Country")[primary_metric]
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )
    df = df[df["Country"].isin(_top_n_countries)]

# Header 
st.title("🌍 Global Climate Change Dashboard (2000–2024)")
st.markdown(
    "An interactive exploration of how temperatures, emissions, sea levels, and renewable energy "
    "have evolved across the globe over two and a half decades."
)
st.markdown("---")

# KPI cards 
def make_kpi(label, col_name, fmt=".2f", suffix=""):
    if df.empty or col_name not in df.columns:
        return (
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>—</div>"
            f"</div>"
        )
    val = df[col_name].mean()
    global_val = df_raw[col_name].mean()
    delta = val - global_val
    sign = "▲" if delta >= 0 else "▼"
    delta_class = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{val:{fmt}}{suffix}</div>"
        f"<div class='{delta_class}'>{sign} {abs(delta):{fmt}}{suffix} vs global avg</div>"
        f"</div>"
    )

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(make_kpi("Avg Temperature", "Avg Temperature (°C)", ".1f", " °C"), unsafe_allow_html=True)
with k2:
    st.markdown(make_kpi("CO2 per Capita", "CO2 Emissions (Tons/Capita)", ".2f", " T"), unsafe_allow_html=True)
with k3:
    st.markdown(make_kpi("Sea Level Rise", "Sea Level Rise (mm)", ".1f", " mm"), unsafe_allow_html=True)
with k4:
    st.markdown(make_kpi("Renewable Energy", "Renewable Energy (%)", ".1f", "%"), unsafe_allow_html=True)

# Section 1: Changes over time — Line Chart 
st.markdown("<div class='section-header'>How has the climate changed over time?</div>", unsafe_allow_html=True)
st.caption("Communication purpose: **Changes over time** — dual-axis line chart")

_line_secondary = (
    "CO2 Emissions (Tons/Capita)"
    if primary_metric != "CO2 Emissions (Tons/Capita)"
    else "Avg Temperature (°C)"
)
_pm_label = METRIC_LABELS.get(primary_metric, primary_metric)
_sec_label = METRIC_LABELS.get(_line_secondary, _line_secondary)

yearly = (
    df.groupby("Year")[[primary_metric, _line_secondary]]
    .mean()
    .reset_index()
)

fig_line = make_subplots(specs=[[{"secondary_y": True}]])
fig_line.add_trace(
    go.Scatter(
        x=yearly["Year"],
        y=yearly[primary_metric],
        name=_pm_label,
        mode="lines+markers",
        line=dict(color="#f87171", width=2.5),
        marker=dict(size=5),
    ),
    secondary_y=False,
)
fig_line.add_trace(
    go.Scatter(
        x=yearly["Year"],
        y=yearly[_line_secondary],
        name=_sec_label,
        mode="lines+markers",
        line=dict(color="#60a5fa", width=2.5, dash="dot"),
        marker=dict(size=5),
    ),
    secondary_y=True,
)
fig_line.update_layout(
    template=T["plotly"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T["plot_bg"],
    font=dict(color=T["text"]),
    hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20),
    height=380,
)
_axis_font = dict(color=T["text"])
fig_line.update_yaxes(title_text=_pm_label, secondary_y=False, gridcolor=T["grid"], title_font=_axis_font, tickfont=_axis_font)
fig_line.update_yaxes(title_text=_sec_label, secondary_y=True, title_font=_axis_font, tickfont=_axis_font)
fig_line.update_xaxes(gridcolor=T["grid"], title_font=_axis_font, tickfont=_axis_font)
st.plotly_chart(fig_line, use_container_width=True)

# ── Section 2: Comparing categories — Bar Chart ───────────────────────────────
st.markdown(
    f"<div class='section-header'>Top countries by {METRIC_LABELS.get(primary_metric, primary_metric)}</div>",
    unsafe_allow_html=True,
)
st.caption("Communication purpose: **Comparing categories** — horizontal bar, colored by primary metric")

bar_data = (
    df.groupby("Country")[primary_metric]
    .mean()
    .reset_index()
    .sort_values(primary_metric, ascending=False)
    .head(top_n)
    .sort_values(primary_metric, ascending=True)
)

fig_bar = px.bar(
    bar_data,
    x=primary_metric,
    y="Country",
    orientation="h",
    color=primary_metric,
    color_continuous_scale=COLOR_SCALES.get(primary_metric, "Blues"),
    labels={primary_metric: METRIC_LABELS.get(primary_metric, primary_metric)},
    template=T["plotly"],
)
fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T["plot_bg"],
    font=dict(color=T["text"]),
    hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
    margin=dict(l=20, r=20, t=20, b=20),
    height=max(350, top_n * 28),
    coloraxis_colorbar=dict(
        title=dict(text=METRIC_LABELS.get(primary_metric, primary_metric), font=dict(color=T["text"])),
        tickfont=dict(color=T["text"]),
    ),
    yaxis=dict(tickfont=dict(size=11)),
)
fig_bar.update_xaxes(gridcolor=T["grid"])
fig_bar.update_yaxes(gridcolor=T["grid"])
st.plotly_chart(fig_bar, use_container_width=True)

# ── Section 3: Part-to-whole — Donut + Grouped Bar ───────────────────────────
st.markdown("<div class='section-header'>Where does energy come from?</div>", unsafe_allow_html=True)
st.caption("Communication purpose: **Part-to-whole** — donut chart + grouped bar")

col_donut, col_gbar = st.columns([1, 2])

ENERGY_HEIGHT = 480  # increased height

with col_donut:
    avg_renewables = df["Renewable Energy (%)"].mean() if not df.empty else 0
    avg_fossil = 100 - avg_renewables
    fig_donut = px.pie(
        names=["Renewable", "Non-Renewable"],
        values=[avg_renewables, avg_fossil],
        hole=0.55,
        color_discrete_sequence=["#4ade80", "#f87171"],
        template=T["plotly"],
        title="Avg Energy Mix (filtered)",
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]),
        hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
        margin=dict(l=10, r=10, t=60, b=20),
        height=ENERGY_HEIGHT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_gbar:
    top_countries = (
        df.groupby("Country")[primary_metric]
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )
    gbar_data = (
        df[df["Country"].isin(top_countries)]
        .groupby("Country")[primary_metric]
        .mean()
        .reset_index()
        .sort_values(primary_metric, ascending=False)
    )
    fig_gbar = px.bar(
        gbar_data,
        x="Country",
        y=primary_metric,
        color=primary_metric,
        color_continuous_scale=COLOR_SCALES.get(primary_metric, "Blues"),
        template=T["plotly"],
        title=f"{METRIC_LABELS.get(primary_metric, primary_metric)} — Top {top_n} Countries",
        labels={primary_metric: METRIC_LABELS.get(primary_metric, primary_metric)},
    )
    fig_gbar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["text"]),
        hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
        margin=dict(l=10, r=10, t=60, b=60),
        height=ENERGY_HEIGHT,
        showlegend=False,
        coloraxis_showscale=False,
    )
    fig_gbar.update_xaxes(tickangle=-35, gridcolor=T["grid"])
    fig_gbar.update_yaxes(gridcolor=T["grid"])
    st.plotly_chart(fig_gbar, use_container_width=True)

# ── Section 4: Connections — Scatter Plot ────────────────────────────────────
st.markdown("<div class='section-header'>Relationships & connections</div>", unsafe_allow_html=True)
st.caption(
    f"Communication purpose: **Plotting connections & relationships** — bubble chart with trendline · "
    f"Y: **{METRIC_LABELS.get(primary_metric, primary_metric)}**"
)

_scatter_x = (
    "Renewable Energy (%)"
    if primary_metric != "Renewable Energy (%)"
    else "CO2 Emissions (Tons/Capita)"
)
_scatter_color = "CO2 Emissions (Tons/Capita)"
if _scatter_color == primary_metric or _scatter_color == _scatter_x:
    _scatter_color = "Avg Temperature (°C)"
_scatter_cols = list({primary_metric, _scatter_x, _scatter_color, "Population"})

scatter_data = (
    df.groupby("Country")[_scatter_cols]
    .mean()
    .reset_index()
    .dropna()
)

if not scatter_data.empty:
    fig_scatter = px.scatter(
        scatter_data,
        x=_scatter_x,
        y=primary_metric,
        size="Population",
        color=_scatter_color,
        color_continuous_scale=COLOR_SCALES.get(_scatter_color, "Reds"),
        hover_name="Country",
        trendline="ols",
        size_max=50,
        labels={
            _scatter_x: METRIC_LABELS.get(_scatter_x, _scatter_x),
            primary_metric: METRIC_LABELS.get(primary_metric, primary_metric),
            _scatter_color: METRIC_LABELS.get(_scatter_color, _scatter_color),
        },
        template=T["plotly"],
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["text"]),
        hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=450,
        coloraxis_colorbar=dict(
            title=dict(text=METRIC_LABELS.get(_scatter_color, _scatter_color), font=dict(color=T["text"])),
            tickfont=dict(color=T["text"]),
        ),
    )
    fig_scatter.update_xaxes(gridcolor=T["grid"])
    fig_scatter.update_yaxes(gridcolor=T["grid"])
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough data to render scatter plot for current selection.")

# ── Section 5: Geo-spatial — Choropleth ──────────────────────────────────────
st.markdown("<div class='section-header'>Global map of climate indicators</div>", unsafe_allow_html=True)
st.caption(
    f"Communication purpose: **Mapping geo-spatial data** — choropleth · "
    f"metric: **{METRIC_LABELS.get(primary_metric, primary_metric)}**"
)

choro_data = df.groupby("Country")[primary_metric].mean().reset_index()
choro_data.columns = ["Country", primary_metric]

cscale = COLOR_SCALES.get(primary_metric, "Viridis")

fig_choro = px.choropleth(
    choro_data,
    locations="Country",
    locationmode="country names",
    color=primary_metric,
    color_continuous_scale=cscale,
    labels={primary_metric: METRIC_LABELS.get(primary_metric, primary_metric)},
    template=T["plotly"],
)
fig_choro.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=T["text"]),
    hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
    geo=dict(
        bgcolor=T["geo_bg"],
        lakecolor=T["geo_bg"],
        landcolor=T["geo_land"],
        showframe=False,
        showcoastlines=True,
        coastlinecolor=T["geo_coast"],
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=540,
    coloraxis_colorbar=dict(
        title=dict(text=METRIC_LABELS.get(primary_metric, primary_metric), font=dict(color=T["text"])),
        tickfont=dict(color=T["text"]),
    ),
)
st.plotly_chart(fig_choro, use_container_width=True)

# ── Section 6: Correlation Heatmap ───────────────────────────────────────────
st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
st.caption("Pearson correlations across all continuous indicators — spot hidden relationships")

_col_short = dict(zip(CONTINUOUS_COLS, ["Temp", "CO2", "Sea Lvl", "Rainfall", "Population", "Renewables", "Ext. Weather", "Forest"]))
_heat_cols = [primary_metric] + [c for c in CONTINUOUS_COLS if c != primary_metric]
_heat_labels = [_col_short[c] for c in _heat_cols]

corr_data = df[CONTINUOUS_COLS].dropna()
if len(corr_data) >= 3:
    corr_matrix = corr_data[_heat_cols].corr().round(2)
    fig_heat = px.imshow(
        corr_matrix.values,
        x=_heat_labels,
        y=_heat_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=True,
        aspect="auto",
        template=T["plotly"],
    )
    fig_heat.update_traces(textfont_size=11)
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["text"]),
        hoverlabel=dict(bgcolor=T["hoverlabel_bg"], font_color=T["hoverlabel_font"], bordercolor=T["hoverlabel_border"]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=460,
        coloraxis_colorbar=dict(
            title=dict(text="Pearson r", font=dict(color=T["text"])),
            tickfont=dict(color=T["text"]),
        ),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Not enough data to compute correlations for current selection.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
_footer_subtext = T["subtext"]
_footer_accent = T["section_accent"]
st.markdown(
    f"<small style='color:{_footer_subtext}'>Built with Streamlit · Plotly · Pandas &nbsp;|&nbsp; "
    f"Dataset: <a href='https://www.kaggle.com/datasets/adilshamim8/temperature' "
    f"target='_blank' style='color:{_footer_accent};'>Global Climate Change Indicators 2000–2024</a> on Kaggle</small>",
    unsafe_allow_html=True,
)
