# 📊 SalesIQ — ML Sales Forecasting Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)


**A professional ML-powered sales forecasting dashboard built with Streamlit, XGBoost, and Random Forest — designed for real-world demand planning.**

 🚀 Live Demo = https://futureml01-gcmrafbxrox2oharasgdat.streamlit.app/

</div>

---

## ✨ Features

- 🤖 **Dual ML Models** — XGBoost and Random Forest with automatic best-model selection
- 🔮 **Autoregressive Forecasting** — Predicts up to 60 days ahead using rolling lag features
- 📈 **Interactive Charts** — Dark-themed matplotlib visuals with fill areas and peak markers
- 🎛️ **Sidebar Controls** — Filter by Region, Category, Segment; adjust Discount, Ship Mode, Sub-Category
- 📊 **Live KPI Metrics** — MAE scores, record counts, avg sales, trend direction
- ⬇️ **CSV Export** — Download forecast results with one click
- 🐛 **Feature Debug Panel** — Inspect model feature order vs current data in real time
- 🎨 **Professional Dark UI** — SaaS-grade design with DM Sans + DM Mono fonts

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────┐
│  SalesIQ                              🟢 LIVE MODEL  │
│  Smart demand forecasting · ML-powered predictions   │
├──────────────┬──────────────────────────────────────┤
│              │  📊 Model Diagnostics                 │
│  ⚙️ Controls │  ┌────────────┐  ┌────────────┐      │
│  ─────────── │  │ Actual     │  │ Predicted  │      │
│  Region      │  │ Sales ↗    │  │ RF / XGB ↗ │      │
│  Category    │  └────────────┘  └────────────┘      │
│  Segment     │                                       │
│  ─────────── │  🔮 30-Day Forecast                   │
│  Discount    │  ┌─────────────────────────────────┐  │
│  Ship Mode   │  │         ╭──╮    ★ Peak          │  │
│  Sub-Cat     │  │    ╭────╯  ╰────────────        │  │
│  ─────────── │  └─────────────────────────────────┘  │
│  🚀 Forecast │  Avg: $245  Peak: $891  Trend: 📈     │
└──────────────┴──────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
FUTURE_ML_01/
│
├── 📄 app.py                    # Main Streamlit dashboard
├── 📦 xgboost_model.json        # Trained XGBoost model
├── 📦 random_forest.pkl         # Trained Random Forest model
├── 📊 Sample - Superstore.csv   # Source dataset
├── 📋 requirements.txt          # Python dependencies
└── 📝 README.md                 # You are here
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/salesiq-dashboard.git
cd salesiq-dashboard
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Model Files

Make sure these files are in the **root directory** (same folder as `app.py`):

| File | Description |
|------|-------------|
| `xgboost_model.json` | Trained XGBoost regressor |
| `random_forest.pkl` | Trained Random Forest regressor |
| `Sample - Superstore.csv` | Superstore sales dataset |

### 5️⃣ Run the App

```bash
streamlit run app.py
```

## 📦 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
```

Install all at once:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn xgboost joblib
```

---

## 📊 Dataset

This project uses the **Sample Superstore** dataset — a widely used retail sales dataset containing:

| Column | Description |
|--------|-------------|
| `Order Date` | Date the order was placed |
| `Region` | US sales region (East, West, Central, South) |
| `Category` | Product category (Furniture, Technology, Office Supplies) |
| `Segment` | Customer segment (Consumer, Corporate, Home Office) |
| `Sales` | 💰 Target variable — order sales amount |
| `Discount` | Discount applied to the order |
| `Ship Mode` | Shipping method selected |
| `Sub-Category` | Product sub-category |

---

## 🧠 ML Pipeline

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_1` | Sales value from 1 day prior |
| `lag_7` | Sales value from 7 days prior |
| `rolling_mean_7` | 7-day rolling average of Sales |
| `rolling_mean_14` | 14-day rolling average of Sales |
| `year`, `month`, `day` | Extracted from Order Date |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | Binary flag for Sat/Sun |

### Models

```
📌 Random Forest Regressor
   └── Ensemble of decision trees
   └── Robust to outliers
   └── Good baseline performance

📌 XGBoost Regressor
   └── Gradient boosted trees
   └── Handles non-linear patterns
   └── Often outperforms RF on tabular data
```

### Auto Model Selection

The dashboard automatically computes **MAE (Mean Absolute Error)** on the filtered data for both models and selects the winner:

```python
if xgb_mae < rf_mae:
    selected = "XGBoost"   # 🏆 Lower error wins
else:
    selected = "Random Forest"
```

---

## 🔮 Forecasting Logic

The forecast is **autoregressive** — each predicted day feeds into the next:

```
Day 1 → predict → append to window
Day 2 → predict (using Day 1 prediction as lag) → append
Day 3 → predict (using Days 1–2 as rolling mean) → ...
```

At each step, `lag_1`, `lag_7`, `rolling_mean_7`, and `rolling_mean_14` are recalculated from the growing prediction window — ensuring the feature vector always matches what the model was trained on.

---

## 🎛️ Dashboard Controls

| Control | Options | Effect |
|---------|---------|--------|
| **Region** | East, West, Central, South | Filters data by US region |
| **Category** | Furniture, Technology, Office Supplies | Filters product category |
| **Segment** | Consumer, Corporate, Home Office | Filters customer segment |
| **Discount** | 0.00 – 0.50 | Sets discount for forecast |
| **Ship Mode** | First/Second/Standard/Same Day | Sets shipping for forecast |
| **Sub-Category** | 17 options | Sets product sub-category |
| **Forecast Horizon** | 7 – 60 days | How far ahead to predict |
| **Model Mode** | Auto / Manual | Auto picks best MAE model |

---

## 🐛 Known Issues & Fixes

### ❗ `Feature names must be in the same order as they were in fit`
The app automatically reads `rf_model.feature_names_in_` and reorders columns before every prediction. If you retrain models, ensure the same feature set and order is used.

### ❗ `Not enough data for selected filters`
Some Region + Category + Segment combinations have very few records. Try a broader filter combination.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repo
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m 'Add some AmazingFeature'

# 4. Push to the branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request
```

---

## 👨‍💻 Author

**Richa Sinha**

- 💼 LinkedIn: https://www.linkedin.com/in/richa-sinha-b2ab57330/
- 📧 Email: rs8537329@gmail.com.com

---

## 🌟 Show Your Support

If this project helped you, please consider giving it a ⭐ on GitHub — it really helps!

---

