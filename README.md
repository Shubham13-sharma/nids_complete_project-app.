# 🛡️ Network Intrusion Detection System (NIDS)

> Machine Learning-based real-time intrusion detection using NSL-KDD dataset  
> **Accuracy: 98.8% | Precision: 98.99% | Recall: 98.91% | F1: 98.95%**

---

## 📁 Project Structure

```
nids_project/
├── nids_project.py        # Core ML pipeline (train + inference)
├── app.py                 # Streamlit web dashboard
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   ├── KDDTrain_.txt      # NSL-KDD training set  (125,973 records)
│   └── KDDTest_.txt       # NSL-KDD test set      (22,544 records)
└── models/                # Auto-created when you train
    ├── nids_model.pkl
    ├── nids_preprocessor.pkl
    └── nids_metrics.json
```

---

## ⚡ Quick Setup (5 steps)

### Step 1 — Clone / download the project

Place all files in a folder, e.g. `nids_project/`.

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Requires Python ≥ 3.9**

### Step 4 — Add dataset files

Copy `KDDTrain_.txt` and `KDDTest_.txt` into the `data/` folder:

```
nids_project/
└── data/
    ├── KDDTrain_.txt
    └── KDDTest_.txt
```

> Download from: https://www.unb.ca/cic/datasets/nsl.html  
> (Both files are included if you received them with this project.)

### Step 5 — Run the project

#### Option A: Command-line training pipeline

```bash
python nids_project.py --train data/KDDTrain_.txt --test data/KDDTest_.txt
```

This will:
1. Load and preprocess the NSL-KDD dataset
2. Train the Random Forest classifier (100 trees)
3. Run 5-fold cross-validation
4. Print precision / recall / accuracy / F1 + confusion matrix
5. Show the top-10 feature importances
6. Save `models/nids_model.pkl` and `models/nids_preprocessor.pkl`
7. Run a 20-packet live inference simulation

#### Option B: Streamlit web dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

Dashboard tabs:
| Tab | What you'll see |
|---|---|
| **Overview** | Accuracy/Precision/Recall/F1 cards, radar chart, confusion matrix, baseline comparison |
| **Classifier** | Input network traffic parameters manually, choose presets (DoS / PortScan / Normal), click Classify |
| **Live Monitor** | Simulates N packets of real-time traffic and shows a live log table + threat-rate pie chart |
| **Analysis** | NSL-KDD attack distribution, feature importance ranking, system architecture diagram |
| **Database** | Saved prediction history, counts, charts, CSV export, SQL query hints |

### Database support

- **SQLite is enabled by default** and stores logs in `data/nids_logs.db`
- Every manual classification and live simulation result is automatically saved
- The database stores prediction metadata plus the raw traffic record as JSON
- **MySQL is optional** if you want to connect from Workbench instead of using SQLite

To use MySQL:

```bash
pip install mysql-connector-python
```

Then choose **MySQL** in the sidebar, enter your connection details, and click **Connect**

---

## 🧠 How It Works

### Algorithm 1 — Preprocessing pipeline

```
Raw NSL-KDD data
  → Remove duplicates & nulls
  → Label-encode categorical features (protocol_type, service, flag)
  → Standardize numerical features  z = (x − μ) / σ
  → Assign binary label: normal=0, any attack=1
  → Return feature matrix F
```

### Algorithm 2 — Real-time inference

```
Network packet Iₜ
  → Preprocess (encode + normalize)  →  feature vector Fₜ
  → Forward-propagate through Random Forest
  → Get probabilities {p_normal, p_attack}
  → If p_attack ≥ τ (0.5) → label = "Attack"  → generate alert
  → Update logs & dashboard
```

### Model architecture

- **100 decision trees** trained with bagging (bootstrap aggregation)
- **max_depth = 20**, balanced class weights
- **Majority voting** for final prediction + confidence score
- Achieves **98.8% accuracy** on the NSL-KDD test set

---

## 📊 Performance Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 98.80% |
| Precision | 98.99% |
| Recall    | 98.91% |
| F1-Score  | 98.95% |

### Baseline Comparison

| Method                  | Accuracy |
|-------------------------|----------|
| Logistic Regression IDS | 88%      |
| KNN-based IDS           | 89%      |
| SVM-based IDS           | 91%      |
| Deep Learning IDS       | 94%      |
| **Random Forest (Ours)**| **98.8%**|

---

## 🔧 CLI Reference

```bash
python nids_project.py [OPTIONS]

Options:
  --train PATH     Path to KDDTrain_.txt   [default: data/KDDTrain_.txt]
  --test  PATH     Path to KDDTest_.txt    [default: data/KDDTest_.txt]
  --model-dir DIR  Where to save artifacts [default: models/]
```

---

## 📦 Dependencies

| Package         | Version  | Purpose                        |
|-----------------|----------|--------------------------------|
| scikit-learn    | 1.4.2    | Random Forest, preprocessing   |
| pandas          | 2.2.2    | Data loading & manipulation    |
| numpy           | 1.26.4   | Numerical operations           |
| streamlit       | 1.35.0   | Web dashboard                  |
| plotly          | 5.22.0   | Interactive charts             |
| matplotlib      | 3.9.0    | Static charts / Jupyter        |
| seaborn         | 0.13.2   | Heatmaps                       |
| notebook        | 7.2.0    | Jupyter notebook support       |
| tqdm            | 4.66.4   | Progress bars                  |
| colorama        | 0.4.6    | Coloured terminal output       |

---

## 🗂️ Files Included

| File                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| `nids_project.py`     | Full ML pipeline: loader, preprocessor, model, inference|
| `app.py`              | Streamlit dashboard with training, monitoring, and DB logging |
| `NIDS_Dashboard.jsx`  | React dashboard component (for web app integration)      |
| `cyber__1_.ipynb`     | Jupyter notebook — exploratory analysis                  |
| `IDS_R_PAPER.pdf`     | Research paper this project is based on                  |
| `requirements.txt`    | Python package requirements                              |

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |
| `FileNotFoundError: data/KDDTrain_.txt` | Copy dataset files to the `data/` folder |
| Streamlit not opening | Try `streamlit run app.py --server.port 8502` |
| Training is slow | Normal for 125K rows; takes ~15–30s on a laptop |
| `models/` folder missing | Created automatically on first training run |

---

## 📚 References

1. Tavallaee et al., *"A detailed analysis of the KDD CUP 99 dataset"*, IEEE 2009
2. Sharafaldin et al., *"Toward generating a new intrusion detection dataset"*, 2018
3. Buczak & Guven, *"A survey of data mining and ML methods for cyber security"*, IEEE 2016
