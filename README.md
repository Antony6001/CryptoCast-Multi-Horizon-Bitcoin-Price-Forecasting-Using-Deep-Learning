# 📈 CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

> A deep learning system for forecasting Bitcoin prices across multiple time horizons using CNN, RNN, LSTM, and Transformer architectures.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Use Cases](#-business-use-cases)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Models Implemented](#-models-implemented)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results & Visualizations](#-results--visualizations)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Skills Demonstrated](#-skills-demonstrated)

---

## 🧩 Problem Statement

Bitcoin prices are highly volatile and governed by complex temporal patterns that traditional statistical methods fail to capture. This project builds and compares deep learning architectures that learn from **60-day historical price sequences** to forecast Bitcoin closing prices across three horizons:

| Horizon | Description |
|---------|-------------|
| **1D** | Next-day price forecast |
| **3D** | 3-day ahead price forecast |
| **7D** | 7-day ahead price forecast |

---

## 💼 Business Use Cases

- **Crypto Trading Platforms** — Short-term price signaling for retail and institutional traders
- **Algorithmic Trading Systems** — Multi-horizon decision making for automated strategies
- **Risk Management Tools** — Anticipate short-term volatility windows
- **Investment Analytics Dashboards** — Predictive insights for portfolio managers
- **Educational Platforms** — Real-world time-series deep learning demonstration

---

## 📊 Dataset

The dataset contains historical Bitcoin OHLCV (Open, High, Low, Close, Volume) data.

| Column | Description |
|--------|-------------|
| `Date` | Trading date |
| `Price` | Closing price *(target variable)* |
| `Open` | Opening price |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |
| `Vol.` | Trading volume |
| `Change %` | Daily percentage change |

📥 **[Download Dataset](https://drive.google.com/file/d/17t7BMoeh-EDKKEGy5uLRyA_FRtcUB7Gd/view?usp=sharing)** *(link)*

### Preprocessing Pipeline

```
Raw Data → Sort by Date → Handle Missing Values → MinMaxScaler → Sliding Window (60 days) → Train/Test Split
```

- **Sequence length:** 60 past days as input
- **Split strategy:** Time-based (no shuffling to prevent data leakage)
- **Scaling:** `MinMaxScaler` on all features

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────┐
│              Raw Bitcoin OHLCV Data          │
└────────────────────┬────────────────────────┘
                     │
              Preprocessing
         (Scale, Sort, Split)
                     │
        ┌────────────▼────────────┐
        │   Sliding Window (60d)  │
        └────────────┬────────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
    1D-CNN          RNN           LSTM      Transformer
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                          │
              Multi-Horizon Predictions
              ┌──────┬──────┬──────┐
              │  1D  │  3D  │  7D  │
              └──────┴──────┴──────┘
                          │
                    Evaluation &
                    Visualization
```

---

## 🤖 Models Implemented

### 1. 1D Convolutional Neural Network (1D-CNN)
- Captures **local temporal patterns** via convolutional filters
- Fast training; effective for detecting short-term trends
- Best for: identifying recurring price patterns within windows

### 2. Recurrent Neural Network (RNN)
- Models **sequential dependencies** in time-series data
- Serves as the baseline temporal model
- Best for: understanding basic sequential learning

### 3. Long Short-Term Memory (LSTM)
- Handles **long-term dependencies** with gating mechanisms
- Mitigates the vanishing gradient problem
- Best for: capturing multi-week price memory effects

### 4. Transformer (Time-Series Attention)
- Uses **self-attention** to weigh all positions in the sequence
- Captures global dependencies without recurrence
- Best for: long sequences with complex inter-day relationships

| Model | Strength | Training Speed |
|-------|----------|----------------|
| 1D-CNN | Short-term trend detection | ⚡ Fastest |
| RNN | Basic temporal learning | 🔄 Fast |
| LSTM | Long-term dependency capture | 🐢 Moderate |
| Transformer | Global context awareness | 🐌 Slowest |

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |

Visualizations include:
- Actual vs Predicted price curves per horizon
- Error distribution plots
- Forecast horizon comparison across models
- Training/validation loss curves

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
TensorFlow >= 2.x  OR  PyTorch >= 2.x
```

### Installation

```bash
git clone https://github.com/Antony6001/CryptoCast-Multi-Horizon-Bitcoin-Price-Forecasting-Using-Deep-Learning.git
cd cryptocast
pip install -r requirements.txt
```

### Run

```bash
# Data preprocessing
python src/preprocess.py

# Train all models
python src/train.py --model all --horizons 1 3 7

# Evaluate and generate plots
python src/evaluate.py
```

Or run the full pipeline in the notebook:

```bash
jupyter notebook notebooks/CryptoCast_Full_Pipeline.ipynb
```

---

## 📁 Project Structure

```
cryptocast/
├── data/
│   └── bitcoin_prices.csv          # Raw dataset
├── notebooks/
│   └── CryptoCast_Full_Pipeline.ipynb
├── src/
│   ├── preprocess.py               # Data loading, scaling, sequence generation
│   ├── models/
│   │   ├── cnn_model.py            # 1D-CNN architecture
│   │   ├── rnn_model.py            # Simple RNN
│   │   ├── lstm_model.py           # LSTM
│   │   └── transformer_model.py   # Time-series Transformer
│   ├── train.py                    # Training loop
│   └── evaluate.py                 # Metrics + visualizations
├── outputs/
│   ├── plots/                      # Forecast & loss curves
│   └── results/                    # CSV of model metrics
├── requirements.txt
└── README.md
```

---

## 🧠 Skills Demonstrated

- **Time Series Analysis** — Sliding window, temporal train-test split, stationarity awareness
- **Sequence Modeling** — Multi-step input/output sequence design
- **Deep Learning** — CNN, RNN, LSTM, Transformer implementations
- **Multi-Step Prediction** — Separate and multi-output forecasting strategies
- **Model Comparison** — Systematic benchmarking across architectures
- **Financial ML** — Avoiding lookahead bias, evaluating forecast quality
- **Real-World System Design** — Modular, reproducible ML pipeline

---

## 🙌 Acknowledgements

- Bitcoin price data sourced from [Kaggle / Investing.com]
- Inspired by research on deep learning for financial time series forecasting

## 👨‍💻 Author

Antony R

Data Science & Machine Learning Enthusiast

Bitcoin Forecasting | Deep Learning | Time Series Analytics