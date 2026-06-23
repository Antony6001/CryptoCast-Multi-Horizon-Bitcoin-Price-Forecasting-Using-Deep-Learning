#import required python libraries

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ["PYTHONHASHSEED"]        = "42"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models    import Sequential, Model
from tensorflow.keras.layers    import (
    SimpleRNN, LSTM, Conv1D, MaxPooling1D, Dense, Dropout,
    Flatten, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Input, Add,
)
from tensorflow.keras.optimizers import Adam



# PAGE CONFIGURATION
st.set_page_config(
    page_title="₿ Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide",
)


st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#f7931a; text-align:center; }
    .sub-title   { font-size:1.1rem; color:#888; text-align:center; margin-top:-10px; }
    .metric-card { background:#1e1e2e; border-radius:12px; padding:14px 20px; text-align:center; }
    .metric-val  { font-size:1.6rem; font-weight:700; color:#f7931a; }
    .metric-lbl  { font-size:.78rem; color:#aaa; }
    .badge       { display:inline-block; padding:3px 10px; border-radius:20px;
                   font-size:.75rem; font-weight:700; margin:2px; }
    .badge-gold  { background:#f7931a22; color:#f7931a; border:1px solid #f7931a; }
    .badge-blue  { background:#4361ee22; color:#4361ee; border:1px solid #4361ee; }
    .badge-green { background:#2ecc7122; color:#2ecc71; border:1px solid #2ecc71; }
    .badge-red   { background:#e74c3c22; color:#e74c3c; border:1px solid #e74c3c; }
    .badge-org   { background:#e67e2222; color:#e67e22; border:1px solid #e67e22; }
    div[data-testid="stTabs"] button { font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">₿ Bitcoin Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Forecasting · RNN · LSTM · CNN · Transformer</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# SIDEBAR — CONTROLS


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    selected_models = st.multiselect(
        "Models to Train",
        ["RNN", "LSTM", "CNN", "Transformer"],
        default=["RNN", "LSTM"],
        help="Select one or more architectures to compare.",
    )
    selected_horizons = st.multiselect(
        "Forecast Horizons",
        ["1-Day", "3-Day", "7-Day"],
        default=["1-Day"],
    )
    train_ratio = st.slider("Train / Test Split", 0.60, 0.90, 0.80, 0.05)
    window_size = st.slider("Lookback Window (days)", 30, 120, 60, 10)
    epochs      = st.slider("Max Epochs", 20, 150, 60, 10)
    batch_size  = st.select_slider("Batch Size", [16, 32, 64, 128], value=32)
    n_show      = st.slider("Test Samples to Plot", 50, 300, 150, 25)

    st.markdown("---")
    st.markdown("**Simulated Data Period**")
    start_yr = st.selectbox("Start Year", [2015, 2016, 2017, 2018], index=0)
    end_yr   = st.selectbox("End Year",   [2022, 2023, 2024], index=2)

    st.markdown("---")
    run_btn = st.button("🚀 Train & Evaluate", use_container_width=True, type="primary")


# DATA GENERATION


@st.cache_data
def generate_data(start_yr, end_yr):
    np.random.seed(42)
    dates  = pd.date_range(start=f"{start_yr}-01-01", end=f"{end_yr}-12-31", freq="D")
    n      = len(dates)
    price  = np.abs(200 + np.cumsum(np.random.randn(n) * 300))
    df = pd.DataFrame({
        "Date":     dates,
        "Price":    np.round(price, 2),
        "Open":     np.round(price * (1 + np.random.uniform(-0.01, 0.01, n)), 2),
        "High":     np.round(price * (1 + np.random.uniform(0.00,  0.02, n)), 2),
        "Low":      np.round(price * (1 - np.random.uniform(0.00,  0.02, n)), 2),
        "Volume":   np.round(np.random.uniform(1e9, 5e10, n), 0),
        "Change %": np.round(np.random.uniform(-5, 5, n), 2),
    })
    df["Date"]  = pd.to_datetime(df["Date"])
    df          = df.sort_values("Date").reset_index(drop=True)
    df["Price"] = df["Price"].ffill()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def create_sequences(data, window=60, horizon=1):
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window : i + window + horizon, 0])
    return np.array(X), np.array(y)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_rnn(window, n_feat, horizon):
    m = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(window, n_feat), activation="tanh"),
        Dropout(0.2),
        SimpleRNN(32, return_sequences=False, activation="tanh"),
        Dropout(0.2),
        Dense(horizon),
    ], name=f"RNN_h{horizon}")
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m

def build_lstm(window, n_feat, horizon):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, n_feat)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(horizon),
    ], name=f"LSTM_h{horizon}")
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m

def build_cnn(window, n_feat, horizon):
    m = Sequential([
        Conv1D(64, 3, activation="relu", padding="same", input_shape=(window, n_feat)),
        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(32, 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(horizon),
    ], name=f"CNN_h{horizon}")
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m

def build_transformer(window, n_feat, horizon, d_model=64, n_heads=4, ff_dim=128, drop=0.1):
    inp  = Input(shape=(window, n_feat))
    x    = Dense(d_model)(inp)
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=drop)(x, x)
    x    = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ff   = Dense(ff_dim, activation="relu")(x)
    ff   = Dense(d_model)(ff)
    ff   = Dropout(drop)(ff)
    x    = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(64, activation="relu")(x)
    x    = Dropout(drop)(x)
    out  = Dense(horizon)(x)
    m    = Model(inp, out, name=f"Transformer_h{horizon}")
    m.compile(optimizer=Adam(1e-4), loss="mse", metrics=["mae"])
    return m

BUILDERS = {"RNN": build_rnn, "LSTM": build_lstm, "CNN": build_cnn, "Transformer": build_transformer}
COLORS   = {"RNN": "#4361ee", "LSTM": "#2ecc71", "CNN": "#e67e22", "Transformer": "#e74c3c"}
HORIZON_MAP = {"1-Day": 1, "3-Day": 3, "7-Day": 7}

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + METRICS
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, X_tr, y_tr, epochs, batch):
    cb = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
    ]
    return model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                     validation_split=0.10, callbacks=cb, verbose=0)

def compute_metrics(model, X_te, y_te, price_scaler):
    y_pred_s = model.predict(X_te, verbose=0)
    y_true   = price_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    y_pred   = price_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2,
                y_true=y_true, y_pred=y_pred, errors=y_true - y_pred)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREVIEW TAB (always visible)
# ─────────────────────────────────────────────────────────────────────────────
df_raw = generate_data(start_yr, end_yr)

tab_data, tab_results, tab_compare = st.tabs(["📊 Dataset", "📈 Model Results", "🏆 Comparison"])

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days",     f"{len(df_raw):,}")
    c2.metric("Min Price",      f"${df_raw['Price'].min():,.0f}")
    c3.metric("Max Price",      f"${df_raw['Price'].max():,.0f}")
    c4.metric("Avg Daily Chg",  f"{df_raw['Change %'].mean():.2f}%")

    st.subheader("Simulated Bitcoin Price History")
    fig_raw, ax_raw = plt.subplots(figsize=(12, 3.5))
    fig_raw.patch.set_facecolor("#0e1117")
    ax_raw.set_facecolor("#0e1117")
    ax_raw.plot(df_raw["Date"], df_raw["Price"], color="#f7931a", linewidth=1)
    ax_raw.fill_between(df_raw["Date"], df_raw["Price"], alpha=0.15, color="#f7931a")
    ax_raw.set_xlabel("Date", color="#aaa")
    ax_raw.set_ylabel("Price (USD)", color="#aaa")
    ax_raw.tick_params(colors="#aaa")
    ax_raw.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for spine in ax_raw.spines.values():
        spine.set_edgecolor("#333")
    st.pyplot(fig_raw)

    st.subheader("Raw Data Sample")
    st.dataframe(df_raw.head(20), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not selected_models:
        st.error("Please select at least one model.")
        st.stop()
    if not selected_horizons:
        st.error("Please select at least one horizon.")
        st.stop()

    # Prepare scaled data
    FEATURE_COLS = ["Price", "Open", "High", "Low", "Volume", "Change %"]
    scaler       = MinMaxScaler()
    scaled_data  = scaler.fit_transform(df_raw[FEATURE_COLS])
    price_scaler = MinMaxScaler()
    price_scaler.fit(df_raw[["Price"]])

    split_idx  = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:split_idx]
    test_data  = scaled_data[split_idx:]
    N_FEAT     = scaled_data.shape[1]

    all_results = {}   # {model_name: {horizon_label: metrics_dict}}
    total_jobs  = len(selected_models) * len(selected_horizons)
    progress    = st.progress(0, text="Initializing…")
    job_i       = 0

    for model_name in selected_models:
        all_results[model_name] = {}
        for h_label in selected_horizons:
            h    = HORIZON_MAP[h_label]
            job_i += 1
            progress.progress(job_i / total_jobs,
                              text=f"Training {model_name} · {h_label} ({job_i}/{total_jobs})")
            X_tr, y_tr = create_sequences(train_data, window_size, h)
            X_te, y_te = create_sequences(test_data,  window_size, h)
            tf.random.set_seed(42)
            m    = BUILDERS[model_name](window_size, N_FEAT, h)
            hist = train_model(m, X_tr, y_tr, epochs, batch_size)
            res  = compute_metrics(m, X_te, y_te, price_scaler)
            res["hist"] = hist
            all_results[model_name][h_label] = res

    progress.empty()
    st.success(f"✅ Trained {total_jobs} model(s) successfully!")

    # ── Results Tab ──────────────────────────────────────────────────────────
    with tab_results:
        for model_name in selected_models:
            badge_cls = {"RNN":"badge-blue","LSTM":"badge-green",
                         "CNN":"badge-org","Transformer":"badge-red"}[model_name]
            st.markdown(f'<span class="badge {badge_cls}">{model_name}</span>',
                        unsafe_allow_html=True)

            # Metric cards
            cols = st.columns(len(selected_horizons))
            for col, h_label in zip(cols, selected_horizons):
                r = all_results[model_name][h_label]
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-weight:700;color:#ccc;margin-bottom:6px">{h_label}</div>
                        <div class="metric-val">${r['MAE']:,.0f}</div>
                        <div class="metric-lbl">MAE</div>
                        <hr style="border-color:#333;margin:8px 0">
                        <div style="color:#ccc;font-size:.85rem">RMSE: ${r['RMSE']:,.0f}</div>
                        <div style="color:#ccc;font-size:.85rem">MAPE: {r['MAPE']:.2f}%</div>
                        <div style="color:#ccc;font-size:.85rem">R²: {r['R2']:.4f}</div>
                    </div>""", unsafe_allow_html=True)

            # Forecast + Loss plots
            n_h   = len(selected_horizons)
            color = COLORS[model_name]
            fig, axes = plt.subplots(2, n_h, figsize=(6 * n_h, 8))
            fig.patch.set_facecolor("#0e1117")
            if n_h == 1:
                axes = np.array(axes).reshape(2, 1)
            fig.suptitle(f"{model_name} — Forecast & Training Loss",
                         fontsize=13, fontweight="bold", color="#fff")

            for i, h_label in enumerate(selected_horizons):
                r    = all_results[model_name][h_label]
                hist = r["hist"]
                # Forecast
                ax = axes[0, i]; ax.set_facecolor("#0e1117")
                ax.plot(r["y_true"][:n_show],  color="#fff",   linewidth=1.5, label="Actual")
                ax.plot(r["y_pred"][:n_show],  color=color,    linewidth=1.1, linestyle="--", label="Predicted")
                ax.set_title(f"{h_label}  MAE=${r['MAE']:,.0f}  R²={r['R2']:.3f}",
                             fontsize=9, color="#ccc")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
                ax.tick_params(colors="#aaa")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                for s in ax.spines.values(): s.set_edgecolor("#333")
                # Loss
                ax2 = axes[1, i]; ax2.set_facecolor("#0e1117")
                ep  = len(hist.history["loss"])
                ax2.plot(hist.history["loss"],     color=color,    linewidth=1.4, label="Train")
                ax2.plot(hist.history["val_loss"], color="#f7931a", linewidth=1.2,
                         linestyle="--", label="Val")
                ax2.set_title(f"{h_label}  ({ep} epochs)", fontsize=9, color="#ccc")
                ax2.set_xlabel("Epoch", color="#aaa")
                ax2.set_ylabel("MSE Loss", color="#aaa")
                ax2.legend(fontsize=7); ax2.grid(True, alpha=0.15)
                ax2.tick_params(colors="#aaa")
                for s in ax2.spines.values(): s.set_edgecolor("#333")

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("---")

    # ── Comparison Tab ───────────────────────────────────────────────────────
    with tab_compare:
        # Build summary table
        rows = []
        for model_name in selected_models:
            for h_label in selected_horizons:
                r = all_results[model_name][h_label]
                rows.append({
                    "Model":    model_name,
                    "Horizon":  h_label,
                    "MAE ($)":  round(r["MAE"],  1),
                    "RMSE ($)": round(r["RMSE"], 1),
                    "MAPE (%)": round(r["MAPE"], 3),
                    "R²":       round(r["R2"],   4),
                })
        df_res = pd.DataFrame(rows)

        st.subheader("📋 Metrics Summary")
        st.dataframe(df_res.style.background_gradient(subset=["R²"], cmap="Greens")
                              .background_gradient(subset=["MAE ($)"], cmap="Reds_r"),
                     use_container_width=True)

        # Best per horizon
        st.subheader("🥇 Best Model per Horizon (by MAE)")
        for h_label in selected_horizons:
            sub  = df_res[df_res["Horizon"] == h_label]
            if sub.empty: continue
            best = sub.loc[sub["MAE ($)"].idxmin()]
            col  = COLORS.get(best["Model"], "#f7931a")
            st.markdown(
                f"**{h_label}** → "
                f'<span style="color:{col};font-weight:700">{best["Model"]}</span> '
                f'· MAE=${best["MAE ($)"]:,.0f} · MAPE={best["MAPE (%)"]}% · R²={best["R²"]}',
                unsafe_allow_html=True,
            )

        # MAE Bar chart
        if len(selected_horizons) > 0 and len(selected_models) > 1:
            st.subheader("📊 MAE Comparison Chart")
            n_h2 = len(selected_horizons)
            fig2, axes2 = plt.subplots(1, n_h2, figsize=(6 * n_h2, 4.5))
            fig2.patch.set_facecolor("#0e1117")
            if n_h2 == 1:
                axes2 = [axes2]
            fig2.suptitle("MAE per Model per Horizon (lower = better)",
                          fontsize=12, fontweight="bold", color="#fff")
            for ax2, h_label in zip(axes2, selected_horizons):
                ax2.set_facecolor("#0e1117")
                sub   = df_res[df_res["Horizon"] == h_label]
                maes  = sub["MAE ($)"].values
                names = sub["Model"].values
                bars  = ax2.bar(names, maes,
                                color=[COLORS[m] for m in names],
                                width=0.5, edgecolor="#222")
                best_i = int(np.argmin(maes))
                bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(3)
                for bar, v in zip(bars, maes):
                    ax2.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + max(maes) * 0.01,
                             f"${v:,.0f}", ha="center", va="bottom",
                             fontsize=9, color="#ddd")
                ax2.set_title(f"{h_label}", fontsize=11, fontweight="bold", color="#ccc")
                ax2.set_ylabel("MAE (USD)", color="#aaa")
                ax2.set_ylim(0, max(maes) * 1.18)
                ax2.grid(axis="y", alpha=0.2)
                ax2.tick_params(colors="#aaa")
                for s in ax2.spines.values(): s.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig2)

        # Overlay plot (if multiple models, single horizon)
        if len(selected_models) > 1 and len(selected_horizons) >= 1:
            st.subheader("📉 All Models Overlay")
            h_label = selected_horizons[0]
            fig3, ax3 = plt.subplots(figsize=(13, 4.5))
            fig3.patch.set_facecolor("#0e1117")
            ax3.set_facecolor("#0e1117")
            first_model = selected_models[0]
            ax3.plot(all_results[first_model][h_label]["y_true"][:n_show],
                     color="#fff", linewidth=2, label="Actual", zorder=6)
            lstyles = ["--", "-.", ":", "-"]
            for i, model_name in enumerate(selected_models):
                ax3.plot(all_results[model_name][h_label]["y_pred"][:n_show],
                         color=COLORS[model_name], linestyle=lstyles[i % 4],
                         linewidth=1.2, label=model_name, alpha=0.85)
            ax3.set_title(f"All Models · {h_label} Horizon",
                          fontsize=11, fontweight="bold", color="#ccc")
            ax3.set_xlabel("Test Sample Index", color="#aaa")
            ax3.set_ylabel("Bitcoin Price ($)", color="#aaa")
            ax3.legend(fontsize=9); ax3.grid(True, alpha=0.15)
            ax3.tick_params(colors="#aaa")
            ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            for s in ax3.spines.values(): s.set_edgecolor("#333")
            plt.tight_layout()
            st.pyplot(fig3)

else:
    with tab_results:
        st.info("👈 Configure your settings in the sidebar and click **Train & Evaluate** to begin.")
    with tab_compare:
        st.info("👈 Train at least two models to compare results here.")
