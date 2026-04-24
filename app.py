import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SalesIQ — Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS — Dark Professional SaaS Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] label,
.stSelectbox label, .stSlider label, .stRadio label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8892b0 !important;
    margin-bottom: 0.3rem !important;
}
div[data-baseweb="select"] > div {
    background-color: #1a1d27 !important;
    border: 1px solid #252840 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
}
hr {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 1rem 0;
}

/* ---------- Run button ---------- */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f6ef7 0%, #6c47ff 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: opacity 0.2s ease, transform 0.1s ease;
    box-shadow: 0 4px 20px rgba(79, 110, 247, 0.35);
}
div.stButton > button:hover  { opacity: 0.88; transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0px); opacity: 1; }

/* ---------- KPI cards ---------- */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #13161f, #1a1d2b);
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #4f6ef7; }
[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #8892b0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.75rem !important;
    font-weight: 500 !important;
    color: #e2e8f0 !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.85rem !important;
}
.stSuccess { background-color: rgba(34,197,94,0.1)  !important; border-left: 3px solid #22c55e !important; }
.stInfo    { background-color: rgba(79,110,247,0.1)  !important; border-left: 3px solid #4f6ef7 !important; }
.stWarning { background-color: rgba(251,191,36,0.08) !important; border-left: 3px solid #fbbf24 !important; }

/* ---------- Download button ---------- */
[data-testid="stDownloadButton"] > button {
    background: transparent;
    border: 1px solid #4f6ef7;
    color: #818cf8;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.5rem 1rem;
    transition: all 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(79,110,247,0.12);
    color: #c7d2fe;
}

/* ---------- Section tags ---------- */
.section-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f6ef7;
    background: rgba(79,110,247,0.1);
    border: 1px solid rgba(79,110,247,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    margin-bottom: 0.5rem;
}

/* ---------- Top banner ---------- */
.top-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e2130;
}
.brand-name {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #e2e8f0;
}
.brand-name span { color: #4f6ef7; }
.brand-tagline {
    font-size: 0.78rem;
    color: #8892b0;
    margin-top: 0.2rem;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #4ade80;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 20px;
    padding: 0.35rem 0.8rem;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.3; }
}

/* ---------- Chart cards ---------- */
.chart-card {
    background: linear-gradient(145deg, #13161f, #1a1d2b);
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}

/* ---------- Footer ---------- */
.caption-text {
    font-size: 0.72rem;
    color: #4a5568;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB DARK THEME HELPER
# ============================================================
def apply_chart_style(ax, fig):
    fig.patch.set_facecolor('#13161f')
    ax.set_facecolor('#13161f')
    ax.tick_params(colors='#8892b0', labelsize=8)
    ax.xaxis.label.set_color('#8892b0')
    ax.yaxis.label.set_color('#8892b0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2235')
    ax.grid(color='#1e2235', linestyle='--', linewidth=0.6, alpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))


# ============================================================
# TOP BANNER
# ============================================================
st.markdown("""
<div class="top-banner">
    <div>
        <div class="brand-name">Sales<span>IQ</span></div>
        <div class="brand-tagline">Smart demand forecasting · ML-powered predictions</div>
    </div>
    <div class="live-badge">
        <div class="dot"></div> LIVE MODEL
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD & PREPARE DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

    # Drop columns not used for modelling
    df = df.drop([
        'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
        'Product ID', 'Product Name', 'City', 'Country',
        'Postal Code', 'Ship Date'
    ], axis=1)

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('Order Date').reset_index(drop=True)

    # Date features
    df['year']        = df['Order Date'].dt.year
    df['month']       = df['Order Date'].dt.month
    df['day']         = df['Order Date'].dt.day
    df['day_of_week'] = df['Order Date'].dt.dayofweek
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)

    # NOTE: rolling means are intentionally NOT computed here at the global level
    # because they must be computed per-group (after filtering) to be meaningful.
    # Placeholder columns are added so the DataFrame shape is consistent.
    df['rolling_mean_7']  = np.nan
    df['rolling_mean_14'] = np.nan

    return df


df = load_data()


# ============================================================
# ENCODE CATEGORICAL COLUMNS
# ============================================================
cat_cols = ['Ship Mode', 'Segment', 'State', 'Region', 'Category', 'Sub-Category']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# ============================================================
# LOAD TRAINED MODELS
# ============================================================
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgboost_model.json")

rf_model = joblib.load("random_forest.pkl")


# ============================================================
# SIDEBAR — CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ CONTROLS")

    st.markdown('<div class="section-tag">Filters</div>', unsafe_allow_html=True)
    region   = st.selectbox("Region",   encoders['Region'].classes_)
    category = st.selectbox("Category", encoders['Category'].classes_)
    segment  = st.selectbox("Segment",  encoders['Segment'].classes_)

    st.markdown("---")
    st.markdown('<div class="section-tag">Parameters</div>', unsafe_allow_html=True)
    discount     = st.slider("Discount", 0.0, 0.5, 0.1, step=0.01, format="%.2f")
    ship_mode    = st.selectbox("Ship Mode",    encoders['Ship Mode'].classes_)
    sub_category = st.selectbox("Sub-Category", encoders['Sub-Category'].classes_)

    st.markdown("---")
    st.markdown('<div class="section-tag">Forecast Settings</div>', unsafe_allow_html=True)
    forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 30)

    mode = st.radio(
        "Model Mode",
        ["Auto (Best Model)", "Manual Selection"],
        label_visibility="visible"
    )
    if mode == "Manual Selection":
        model_choice = st.selectbox("Select Model", ["XGBoost", "Random Forest"])
    else:
        model_choice = None

    st.markdown("---")
    run_forecast = st.button("🚀 Run Forecast")


# ============================================================
# FILTER DATA & ENGINEER FEATURES PER GROUP
# ============================================================
filtered = df[
    (df['Region']   == encoders['Region'].transform([region])[0]) &
    (df['Category'] == encoders['Category'].transform([category])[0]) &
    (df['Segment']  == encoders['Segment'].transform([segment])[0])
].copy()

filtered = filtered.sort_values('Order Date').reset_index(drop=True)

# Lag features (require sort by date, hence done after filtering)
filtered['lag_1'] = filtered['Sales'].shift(1)
filtered['lag_7'] = filtered['Sales'].shift(7)

# Rolling mean features — computed on this filtered group's Sales history
# min_periods=1 avoids extra NaNs at the start of the series
filtered['rolling_mean_7']  = filtered['Sales'].rolling(window=7,  min_periods=1).mean()
filtered['rolling_mean_14'] = filtered['Sales'].rolling(window=14, min_periods=1).mean()

# Drop rows with NaN introduced by lag shifts (lag_7 needs at least 7 rows)
filtered = filtered.dropna().reset_index(drop=True)

# Feature matrix and target — drop non-feature columns
X_raw = filtered.drop(columns=['Sales', 'Order Date'])
y     = filtered['Sales']

# ---------------------------------------------------------------
# CRITICAL: reorder columns to EXACTLY match the training order.
# rf_model.feature_names_in_ holds the column list sklearn saw at fit time.
# XGBoost also stores feature names; we use the RF order as the canonical one
# (both models must have been trained on the same feature set).
# ---------------------------------------------------------------
FEATURE_ORDER = list(rf_model.feature_names_in_)

# Safety check — warn if any expected column is missing
missing = [c for c in FEATURE_ORDER if c not in X_raw.columns]
extra   = [c for c in X_raw.columns  if c not in FEATURE_ORDER]

if missing:
    st.error(
        f"❌ Feature mismatch — columns missing from current data that the model "
        f"was trained on: `{missing}`. Check your feature engineering pipeline."
    )
    st.stop()

# Reindex to training order (drops any extra cols that weren't in training)
X = X_raw[FEATURE_ORDER]

# Debug expander — remove in production if desired
with st.expander("🔍 Feature Debug", expanded=False):
    st.caption("Expected feature order (from rf_model.feature_names_in_):")
    st.code(FEATURE_ORDER)
    st.caption(f"Current X columns ({len(X.columns)}): match ✅" if not extra else
               f"Extra cols ignored: {extra}")


# ============================================================
# MODEL EVALUATION ON FILTERED DATA
# ============================================================
rf_pred  = rf_model.predict(X)
xgb_pred = xgb_model.predict(X)

rf_mae  = np.mean(np.abs(y - rf_pred))
xgb_mae = np.mean(np.abs(y - xgb_pred))

# Auto-select best model by MAE
if xgb_mae < rf_mae:
    best_model, best_name, best_pred, best_mae = xgb_model, "XGBoost",       xgb_pred, xgb_mae
else:
    best_model, best_name, best_pred, best_mae = rf_model,  "Random Forest", rf_pred,  rf_mae

# Apply user's model mode choice
if mode == "Auto (Best Model)":
    selected_model, selected_name, selected_pred = best_model, best_name, best_pred
else:
    if model_choice == "XGBoost":
        selected_model, selected_name, selected_pred = xgb_model, "XGBoost",       xgb_pred
    else:
        selected_model, selected_name, selected_pred = rf_model,  "Random Forest", rf_pred


# ============================================================
# SECTION 1 — MODEL SELECTION BADGE
# ============================================================
st.markdown('<div class="section-tag">Model Selection</div>', unsafe_allow_html=True)
st.markdown("#### 🏆 Active Model")

if mode == "Auto (Best Model)":
    st.success(f"**{best_name}** selected automatically — MAE: **{best_mae:.2f}**")
else:
    st.info(f"Manual override: **{selected_name}** in use")

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 2 — KPI METRICS
# ============================================================
st.markdown('<div class="section-tag">Performance Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Random Forest MAE", f"{rf_mae:,.2f}",  help="Mean Absolute Error — Random Forest")
c2.metric("XGBoost MAE",       f"{xgb_mae:,.2f}", help="Mean Absolute Error — XGBoost")
c3.metric("Filtered Records",  f"{len(y):,}",      help="Records after applying sidebar filters")
c4.metric("Avg Actual Sales",  f"{y.mean():,.1f}", help="Mean sales value in filtered dataset")

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 3 — DIAGNOSTIC CHARTS
# ============================================================
st.markdown('<div class="section-tag">Model Diagnostics</div>', unsafe_allow_html=True)
st.markdown("#### 📊 Actual Sales vs Model Predictions")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("**📈 Actual Sales**")
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(y.values, color='#4f6ef7', linewidth=1.4, alpha=0.9)
    ax.fill_between(range(len(y)), y.values, alpha=0.12, color='#4f6ef7')
    apply_chart_style(ax, fig)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(f"**🤖 {selected_name} Predictions**")
    fig2, ax2 = plt.subplots(figsize=(5, 2.6))
    ax2.plot(selected_pred, color='#a78bfa', linewidth=1.4, alpha=0.9)
    ax2.fill_between(range(len(selected_pred)), selected_pred, alpha=0.12, color='#a78bfa')
    apply_chart_style(ax2, fig2)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FORECAST FUNCTION
# ============================================================
def generate_forecast(model, last_data, days, user_inputs):
    """
    Autoregressively forecast `days` steps ahead.

    At each step, lag_1, lag_7, rolling_mean_7, and rolling_mean_14
    are recomputed from the growing `current` DataFrame so that
    the feature vector exactly matches what the model was trained on.

    Parameters
    ----------
    model       : fitted sklearn/XGBoost model
    last_data   : DataFrame — seed window (must include Sales & Order Date columns)
    days        : int — number of days to forecast
    user_inputs : dict — encoded sidebar values for Discount, Ship Mode, Sub-Category
    """
    forecast = []
    current  = last_data.copy().reset_index(drop=True)

    for _ in range(days):
        # Start from the last known row as a template
        row = current.iloc[-1:].copy()

        # Overwrite controllable sidebar inputs
        row['Discount']     = user_inputs['discount']
        row['Ship Mode']    = user_inputs['ship_mode']
        row['Sub-Category'] = user_inputs['sub_category']

        # --- Lag features ---
        row['lag_1'] = current['Sales'].iloc[-1]
        row['lag_7'] = (
            current['Sales'].iloc[-7]
            if len(current) >= 7
            else current['Sales'].iloc[-1]
        )

        # --- Rolling mean features ---
        row['rolling_mean_7'] = (
            current['Sales'].iloc[-7:].mean()
            if len(current) >= 7
            else current['Sales'].mean()
        )
        row['rolling_mean_14'] = (
            current['Sales'].iloc[-14:].mean()
            if len(current) >= 14
            else current['Sales'].mean()
        )

        # Drop non-feature columns before prediction, then reorder to training order
        pred = model.predict(
            row.drop(columns=['Sales', 'Order Date'])[FEATURE_ORDER]
        )[0]

        # Append predicted row so the next iteration can use it
        new_row          = row.copy()
        new_row['Sales'] = pred
        current          = pd.concat([current, new_row], ignore_index=True)

        forecast.append(float(pred))

    return forecast


# ============================================================
# SECTION 4 — FORECAST OUTPUT (only when button is clicked)
# ============================================================
if run_forecast:
    if len(filtered) <= 10:
        st.warning(
            "⚠️ Not enough data for the selected filters. "
            "Try adjusting Region, Category, or Segment."
        )
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-tag">Forecast Output</div>',
                    unsafe_allow_html=True)
        st.markdown(f"#### 🔮 {forecast_days}-Day Sales Forecast · {selected_name}")

        # Encoded sidebar values
        user_inputs = {
            "discount":     discount,
            "ship_mode":    encoders['Ship Mode'].transform([ship_mode])[0],
            "sub_category": encoders['Sub-Category'].transform([sub_category])[0]
        }

        # Seed window: last 20 rows of filtered data
        last_window     = filtered.tail(20).copy()
        forecast_values = generate_forecast(
            selected_model, last_window, forecast_days, user_inputs
        )
        future_dates = pd.date_range(
            start=filtered['Order Date'].max(), periods=forecast_days, freq='D'
        )

        # --- Forecast chart ---
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(10, 3.2))

        ax3.plot(future_dates, forecast_values,
                 color='#4ade80', linewidth=2, label='Forecast')
        ax3.fill_between(future_dates, forecast_values,
                         alpha=0.15, color='#4ade80')

        # Highlight the peak day
        peak_idx = int(np.argmax(forecast_values))
        ax3.scatter(
            future_dates[peak_idx], forecast_values[peak_idx],
            color='#fbbf24', s=70, zorder=5,
            label=f'Peak: {forecast_values[peak_idx]:,.1f}'
        )

        ax3.set_xlabel("Date", fontsize=8)
        ax3.set_ylabel("Forecasted Sales ($)", fontsize=8)
        ax3.legend(fontsize=8, facecolor='#1a1d27',
                   edgecolor='#252840', labelcolor='#8892b0')
        apply_chart_style(ax3, fig3)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Forecast KPI row ---
        trend_val   = forecast_values[-1] - forecast_values[0]
        trend_label = "📈 Upward" if trend_val > 0 else "📉 Downward"
        trend_delta = f"+{trend_val:,.1f}" if trend_val > 0 else f"{trend_val:,.1f}"

        fa, fb, fc, fd = st.columns(4)
        fa.metric("Avg Forecast", f"${np.mean(forecast_values):,.2f}")
        fb.metric("Peak Sales",   f"${max(forecast_values):,.2f}")
        fc.metric("Min Sales",    f"${min(forecast_values):,.2f}")
        fd.metric("Trend",        trend_label, delta=trend_delta)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Download CSV ---
        forecast_df = pd.DataFrame({
            "Date":         future_dates.strftime('%Y-%m-%d'),
            "Forecast ($)": forecast_values
        })
        st.download_button(
            label="⬇ Download Forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name=(
                f"forecast_{selected_name.lower().replace(' ', '_')}"
                f"_{forecast_days}d.csv"
            ),
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;">
    <div class="caption-text">SalesIQ · ML Forecasting Dashboard</div>
    <div class="caption-text">XGBoost · Random Forest · Streamlit</div>
</div>
""", unsafe_allow_html=True)