"""
=======================================================================
NIDS Dashboard  —  Streamlit Web Application  (Advanced DB Edition)
=======================================================================
Run:  cd F:\nids_complete_project\nids_project
streamlit run app.py

NEW in this version:
  ✅  Advanced dashboard with training, classification, simulation, analysis
  ✅  Database tab — full view of saved predictions
  ✅  Auto-save every Classify + Simulation result to DB
  ✅  SQLite works out of the box (no server required)
  ✅  Optional MySQL Workbench connection from the sidebar
  ✅  Saved raw traffic payload for each prediction
=======================================================================
"""

import json
import os
import pickle
import random
import re
import sqlite3
import time
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nids_project import (
    CATEGORICAL_FEATURES, FEATURE_COLS, NSL_KDD_COLUMNS,
    NIDSInferencePipeline, NIDSModel, NIDSPreprocessor, load_dataset,
)

# ── MySQL soft import — app still runs if driver missing ────────────────
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main  { background-color: #060d1f; }
    .stApp { background: linear-gradient(135deg, #060d1f 0%, #0a1628 100%); }
    h1, h2, h3 { color: #00f5a0 !important; font-family: 'IBM Plex Mono', monospace; }
    .metric-card {
        background: linear-gradient(135deg, #0d1f3c, #0a1628);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .attack-badge {
        background:#ff6b6b33; border:1px solid #ff6b6b;
        color:#ff6b6b; border-radius:8px; padding:4px 12px; font-weight:bold;
    }
    .normal-badge {
        background:#00f5a033; border:1px solid #00f5a0;
        color:#00f5a0; border-radius:8px; padding:4px 12px; font-weight:bold;
    }
    div[data-testid="stMetricValue"] { font-size:2rem !important; color:#00f5a0 !important; }
    .db-ok  { color:#00f5a0; font-weight:700; font-size:13px; }
    .db-off { color:#ff6b6b; font-weight:700; font-size:13px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "model_loaded":  False,
    "model":         None,
    "preprocessor":  None,
    "pipeline":      None,
    "metrics":       None,
    "live_logs":     [],
    "alert_count":   0,
    "normal_count":  0,
    "last_result":   None,
    "last_record":   None,
    "attack_threshold": 0.50,
    # DB
    "db_backend":    "SQLite",
    "db_connected":  False,
    "db_conn":       None,
    "db_host":       "localhost",
    "db_port":       3306,
    "db_user":       "root",
    "db_password":   "",
    "db_name":       "nids_db",
    "db_path":       "data/nids_logs.db",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════════════
# MYSQL HELPERS
# ════════════════════════════════════════════════════════════════════════

def _safe_db_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
        raise ValueError("Database name may only contain letters, numbers, and underscores.")
    return name


def _create_logs_table(conn, backend: str):
    cur = conn.cursor()
    if backend == "SQLite":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nids_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                protocol TEXT DEFAULT '',
                service TEXT DEFAULT '',
                flag TEXT DEFAULT '',
                src_bytes INTEGER DEFAULT 0,
                dst_bytes INTEGER DEFAULT 0,
                duration_sec INTEGER DEFAULT 0,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                p_attack REAL NOT NULL,
                p_normal REAL NOT NULL,
                source TEXT DEFAULT 'manual',
                record_json TEXT DEFAULT '{}'
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nids_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                logged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                protocol VARCHAR(10) DEFAULT '',
                service VARCHAR(50) DEFAULT '',
                flag VARCHAR(15) DEFAULT '',
                src_bytes BIGINT DEFAULT 0,
                dst_bytes BIGINT DEFAULT 0,
                duration_sec INT DEFAULT 0,
                label VARCHAR(10) NOT NULL,
                confidence FLOAT NOT NULL,
                p_attack FLOAT NOT NULL,
                p_normal FLOAT NOT NULL,
                source VARCHAR(20) DEFAULT 'manual',
                record_json LONGTEXT
            )
        """)
    conn.commit()
    cur.close()


def db_connect(backend, host=None, port=None, user=None, password=None, dbname=None, path=None):
    """Connect to SQLite or MySQL and create the logs table if missing."""
    try:
        if backend == "SQLite":
            db_path = path or st.session_state.db_path
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            _create_logs_table(conn, "SQLite")
            return conn, "OK"

        if not MYSQL_AVAILABLE:
            return None, "mysql-connector-python not installed.\nRun: pip install mysql-connector-python"

        safe_name = _safe_db_name(dbname or st.session_state.db_name)
        conn = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            connection_timeout=6,
            autocommit=False,
        )
        cur = conn.cursor()
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{safe_name}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cur.execute(f"USE `{safe_name}`")
        cur.close()
        _create_logs_table(conn, "MySQL")
        return conn, "OK"
    except Exception as exc:
        return None, str(exc)


def _use_db(conn):
    if st.session_state.db_backend != "MySQL":
        return
    try:
        conn.cursor().execute(f"USE `{st.session_state.db_name}`")
    except Exception:
        pass


def db_insert(conn, record: dict, result: dict, source: str = "manual") -> bool:
    if conn is None:
        return False
    try:
        _use_db(conn)
        cur = conn.cursor()
        payload = (
            str(record.get("protocol_type", "")),
            str(record.get("service", "")),
            str(record.get("flag", "")),
            int(record.get("src_bytes", 0)),
            int(record.get("dst_bytes", 0)),
            int(record.get("duration", 0)),
            result["label"],
            float(result["confidence"]),
            float(result["p_attack"]),
            float(result["p_normal"]),
            source,
            json.dumps(record, default=str),
        )
        if st.session_state.db_backend == "SQLite":
            cur.execute("""
                INSERT INTO nids_logs
                    (protocol, service, flag, src_bytes, dst_bytes,
                     duration_sec, label, confidence, p_attack, p_normal, source, record_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, payload)
        else:
            cur.execute("""
                INSERT INTO nids_logs
                    (protocol, service, flag, src_bytes, dst_bytes,
                     duration_sec, label, confidence, p_attack, p_normal, source, record_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, payload)
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False


def db_fetch(conn, limit: int = 300) -> pd.DataFrame:
    if conn is None:
        return pd.DataFrame()
    try:
        _use_db(conn)
        if st.session_state.db_backend == "SQLite":
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM nids_logs ORDER BY logged_at DESC, id DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
        else:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT * FROM nids_logs ORDER BY logged_at DESC, id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def db_count(conn) -> int:
    if conn is None:
        return 0
    try:
        _use_db(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nids_logs")
        n = cur.fetchone()[0]
        cur.close()
        return int(n)
    except Exception:
        return 0


def db_clear(conn) -> bool:
    if conn is None:
        return False
    try:
        _use_db(conn)
        cur = conn.cursor()
        if st.session_state.db_backend == "SQLite":
            cur.execute("DELETE FROM nids_logs")
            cur.execute("DELETE FROM sqlite_sequence WHERE name='nids_logs'")
        else:
            cur.execute("TRUNCATE TABLE nids_logs")
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False


def db_ping(conn) -> bool:
    if conn is None:
        return False
    try:
        if st.session_state.db_backend == "SQLite":
            conn.execute("SELECT 1")
            return True
        conn.ping(reconnect=True, attempts=2, delay=1)
        return True
    except Exception:
        return False


def ensure_default_db_connection():
    if st.session_state.db_connected and db_ping(st.session_state.db_conn):
        return
    if st.session_state.db_backend != "SQLite":
        return
    conn, msg = db_connect("SQLite", path=st.session_state.db_path)
    if conn:
        st.session_state.db_conn = conn
        st.session_state.db_connected = True
    else:
        st.session_state.db_conn = None
        st.session_state.db_connected = False
        st.session_state.db_error = msg


ensure_default_db_connection()


# ════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════
MODEL_PATH   = "models/nids_model.pkl"
PREPROC_PATH = "models/nids_preprocessor.pkl"
METRICS_PATH = "models/nids_metrics.json"


def _try_load_saved():
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH):
        m = NIDSModel.load(MODEL_PATH)
        p = NIDSPreprocessor.load(PREPROC_PATH)
        st.session_state.model        = m
        st.session_state.preprocessor = p
        st.session_state.pipeline     = NIDSInferencePipeline(m, p)
        st.session_state.model_loaded = True
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                st.session_state.metrics = json.load(f)
        return True
    return False


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ NIDS Control Panel")
    st.markdown("---")

    st.markdown("### 🎯 Detection")
    attack_threshold = st.slider(
        "Attack Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.attack_threshold),
        step=0.01,
        help="Lower values flag more traffic as attacks. Higher values are stricter.",
    )
    st.session_state.attack_threshold = attack_threshold

    if st.session_state.pipeline is not None:
        st.session_state.pipeline.THRESHOLD = attack_threshold

    st.caption(f"Current rule: show intrusion when P(Attack) ≥ {attack_threshold:.2f}")
    st.markdown("---")

    # ── Model ─────────────────────────────────────────────────────
    st.markdown("### 🤖 Model")
    if st.session_state.model_loaded:
        st.success("✅ Model ready")
    else:
        if _try_load_saved():
            st.success("✅ Pre-trained model loaded")
        else:
            st.warning("⚠️ No model found — upload dataset to train")

    st.markdown("---")

    # ── Train ─────────────────────────────────────────────────────
    st.markdown("### 📂 Train Model")
    train_file = st.file_uploader("Upload KDDTrain_.txt", type=["txt", "csv"])
    test_file  = st.file_uploader("Upload KDDTest_.txt (optional)", type=["txt", "csv"])

    if st.button("🚀 Train & Evaluate", disabled=(train_file is None), type="primary"):
        with st.spinner("Training Random Forest … (~30 sec)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tf:
                tf.write(train_file.read())
                train_path = tf.name

            df_train     = load_dataset(train_path)
            preprocessor = NIDSPreprocessor()
            X, y         = preprocessor.fit_transform(df_train)

            if test_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tf2:
                    tf2.write(test_file.read())
                    test_path = tf2.name
                df_test  = load_dataset(test_path)
                X_test   = preprocessor.transform(df_test)
                y_test   = (df_test["label"] != "normal").astype(int).values
                X_train, y_train = X, y
            else:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)

            model   = NIDSModel(n_estimators=100, max_depth=20)
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)

            os.makedirs("models", exist_ok=True)
            model.save(MODEL_PATH)
            preprocessor.save(PREPROC_PATH)
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)

            st.session_state.model        = model
            st.session_state.preprocessor = preprocessor
            st.session_state.pipeline     = NIDSInferencePipeline(model, preprocessor)
            st.session_state.metrics      = metrics
            st.session_state.model_loaded = True

        st.success("✅ Training complete!")
        st.rerun()

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MYSQL WORKBENCH CONNECTION PANEL
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🗄️ Database")

    db_backend = st.selectbox(
        "Database Engine",
        ["SQLite", "MySQL"],
        index=0 if st.session_state.db_backend == "SQLite" else 1,
        key="s_backend",
    )

    with st.expander("⚙️ Connection Settings", expanded=not st.session_state.db_connected):
        if db_backend == "SQLite":
            db_path = st.text_input(
                "SQLite File",
                value=st.session_state.db_path,
                key="s_path",
            )
            st.caption("💡 SQLite needs no server. The file is created automatically.")
            db_host = st.session_state.db_host
            db_port = st.session_state.db_port
            db_user = st.session_state.db_user
            db_pass = st.session_state.db_password
            db_name = st.session_state.db_name
        else:
            if not MYSQL_AVAILABLE:
                st.warning("MySQL driver missing. Install `mysql-connector-python` to use MySQL.")
            db_host = st.text_input("Host", value=st.session_state.db_host, key="s_host")
            db_port = st.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=st.session_state.db_port,
                step=1,
                key="s_port",
            )
            db_user = st.text_input("Username", value=st.session_state.db_user, key="s_user")
            db_pass = st.text_input(
                "Password",
                value=st.session_state.db_password,
                type="password",
                key="s_pass",
            )
            db_name = st.text_input("Database", value=st.session_state.db_name, key="s_name")
            db_path = st.session_state.db_path
            st.caption("💡 Database and table are created automatically.")

    btn1, btn2 = st.columns(2)

    with btn1:
        if st.button("🔌 Connect", use_container_width=True, type="primary"):
            conn, msg = db_connect(
                db_backend,
                host=db_host,
                port=int(db_port),
                user=db_user,
                password=db_pass,
                dbname=db_name,
                path=db_path,
            )
            if conn:
                st.session_state.db_backend = db_backend
                st.session_state.db_conn = conn
                st.session_state.db_connected = True
                st.session_state.db_host = db_host
                st.session_state.db_port = int(db_port)
                st.session_state.db_user = db_user
                st.session_state.db_password = db_pass
                st.session_state.db_name = db_name
                st.session_state.db_path = db_path
                st.success(f"✅ Connected to {db_backend}!")
            else:
                st.session_state.db_connected = False
                st.session_state.db_conn = None
                st.error(f"❌ {msg}")

    with btn2:
        if st.button("⛔ Disconnect", use_container_width=True):
            if st.session_state.db_conn:
                try:
                    st.session_state.db_conn.close()
                except Exception:
                    pass
            st.session_state.db_conn = None
            st.session_state.db_connected = False
            st.info("Disconnected.")

    if st.session_state.db_connected and db_ping(st.session_state.db_conn):
        n_rows = db_count(st.session_state.db_conn)
        db_target = (
            st.session_state.db_path
            if st.session_state.db_backend == "SQLite"
            else st.session_state.db_name
        )
        st.markdown(
            f'<span class="db-ok">● Connected → {st.session_state.db_backend}: {db_target} ({n_rows} rows)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="db-off">● Not connected</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
**Model**: Random Forest (100 trees)  
**Dataset**: NSL-KDD (125K records)  
**Accuracy**: 98.8%  
**Database**: SQLite default + optional MySQL
    """)


# ════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════════════════════
st.markdown("# 🛡️ Network Intrusion Detection System")
st.markdown("*Machine Learning-based real-time threat detection — NSL-KDD Dataset*")
st.markdown("---")

if st.session_state.pipeline is not None:
    st.session_state.pipeline.THRESHOLD = float(st.session_state.attack_threshold)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 Classifier",
    "📡 Live Monitor",
    "📈 Analysis",
    "🗄️ Database",
])


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════
with tab1:
    metrics = st.session_state.metrics

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        c2.metric("🔬 Precision", f"{metrics['precision']*100:.2f}%")
        c3.metric("📡 Recall",    f"{metrics['recall']*100:.2f}%")
        c4.metric("⚡ F1-Score",  f"{metrics['f1']*100:.2f}%")
    else:
        st.info("Train a model to see live metrics. Showing paper baseline below.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Accuracy",  "98.80%", help="From research paper")
        c2.metric("🔬 Precision", "98.99%")
        c3.metric("📡 Recall",    "98.91%")
        c4.metric("⚡ F1-Score",  "98.95%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🕸️ Performance Radar")
        vals = [98.8, 98.99, 98.91, 98.95, 98.70]
        if metrics:
            cm   = metrics["confusion_matrix"]
            tn, fp = cm[0][0], cm[0][1]
            spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
            vals = [metrics["accuracy"]*100, metrics["precision"]*100,
                    metrics["recall"]*100,   metrics["f1"]*100, spec]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals,
            theta=["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"],
            fill="toself", name="Performance",
            line_color="#00f5a0", fillcolor="rgba(0,245,160,0.15)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[85, 100], showticklabels=True)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#74b9ff", margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        st.subheader("🔲 Confusion Matrix")
        cm_vals = [[1916, 26], [28, 2539]]
        if metrics:
            cm_vals = metrics["confusion_matrix"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_vals,
            x=["Pred: Normal", "Pred: Attack"],
            y=["Actual: Normal", "Actual: Attack"],
            text=cm_vals, texttemplate="%{text}",
            colorscale=[[0, "#0a1628"], [0.5, "#0070f3"], [1, "#00f5a0"]],
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#74b9ff", margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("📊 Baseline Model Comparison")
    rf_acc = metrics["accuracy"] if metrics else 0.988
    baseline_df = pd.DataFrame({
        "Method":   ["Logistic Reg.", "KNN", "SVM", "Deep Learning", "Random Forest (Ours)"],
        "Accuracy": [0.88, 0.89, 0.91, 0.94, rf_acc],
    })
    fig_bar = px.bar(baseline_df, x="Method", y="Accuracy", color="Accuracy",
                     color_continuous_scale=["#1e3a5f", "#0070f3", "#00f5a0"],
                     range_y=[0.85, 1.0], text_auto=".3f")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#74b9ff")
    st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFIER
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Real-Time Traffic Classifier")
    if not st.session_state.model_loaded:
        st.warning("Please train or load a model first (use sidebar).")
    else:
        PRESETS = {
            "Normal HTTP Traffic": dict(
                duration=2, protocol_type="tcp", service="http", flag="SF",
                src_bytes=12983, dst_bytes=500, count=1, serror_rate=0.0, same_srv_rate=1.0,
                logged_in=1, dst_host_count=128, dst_host_srv_count=64,
                dst_host_same_srv_rate=1.0, rerror_rate=0.0, diff_srv_rate=0.0,
                dst_host_serror_rate=0.0,
            ),
            "DoS (Neptune) Attack": dict(
                duration=0, protocol_type="tcp", service="private", flag="S0",
                src_bytes=0, dst_bytes=0, count=511, serror_rate=1.0, same_srv_rate=1.0,
                logged_in=0, dst_host_count=255, dst_host_srv_count=10,
                dst_host_same_srv_rate=0.04, rerror_rate=0.0, diff_srv_rate=0.0,
                dst_host_serror_rate=1.0, srv_count=511, srv_serror_rate=1.0,
                dst_host_srv_serror_rate=1.0,
            ),
            "Port Scan (Probe)": dict(
                duration=0, protocol_type="tcp", service="private", flag="REJ",
                src_bytes=0, dst_bytes=0, count=229, serror_rate=0.0, same_srv_rate=0.06,
                logged_in=0, dst_host_count=255, dst_host_srv_count=18,
                dst_host_same_srv_rate=0.08, rerror_rate=1.0, diff_srv_rate=0.85,
                dst_host_serror_rate=0.0,
            ),
            "Brute Force (R2L)": dict(
                duration=0, protocol_type="tcp", service="ftp_data", flag="SF",
                src_bytes=491, dst_bytes=0, count=15, serror_rate=0.0, same_srv_rate=0.07,
                logged_in=0, dst_host_count=35, dst_host_srv_count=12,
                dst_host_same_srv_rate=0.12, rerror_rate=0.0, diff_srv_rate=0.7,
                dst_host_serror_rate=0.0, num_failed_logins=5, hot=3,
                num_compromised=1, is_guest_login=1,
            ),
            "Guaranteed Demo Attack": dict(
                duration=0, protocol_type="tcp", service="private", flag="S0",
                src_bytes=0, dst_bytes=0, count=511, srv_count=511,
                serror_rate=1.0, srv_serror_rate=1.0, same_srv_rate=1.0,
                logged_in=0, dst_host_count=255, dst_host_srv_count=10,
                dst_host_same_srv_rate=0.04, diff_srv_rate=0.0, rerror_rate=0.0,
                dst_host_serror_rate=1.0, dst_host_srv_serror_rate=1.0,
            ),
        }

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**⚡ Quick Presets**")
            preset   = st.selectbox("Load a preset", ["Custom"] + list(PRESETS.keys()))
            defaults = PRESETS.get(preset, {})

            duration      = st.number_input("Duration (sec)", 0, 58329,
                                             int(defaults.get("duration", 0)))
            protocol_type = st.selectbox("Protocol", ["tcp", "udp", "icmp"],
                index=["tcp","udp","icmp"].index(defaults.get("protocol_type","tcp")))

            svc_opts = ["http","ftp_data","private","smtp","ssh","domain_u","eco_i","telnet"]
            svc_val  = defaults.get("service","http")
            service  = st.selectbox("Service", svc_opts,
                index=svc_opts.index(svc_val) if svc_val in svc_opts else 0)

            flag_opts = ["SF","S0","REJ","RSTO","S1","S2","OTH"]
            flag_val  = defaults.get("flag","SF")
            flag      = st.selectbox("Flag", flag_opts,
                index=flag_opts.index(flag_val) if flag_val in flag_opts else 0)

            src_bytes     = st.number_input("Source Bytes", 0, 5_000_000,
                                             int(defaults.get("src_bytes", 5000)))
            dst_bytes     = st.number_input("Dest Bytes",   0, 5_000_000,
                                             int(defaults.get("dst_bytes", 1200)))
            count         = st.slider("Connection Count", 0, 511,
                                       int(defaults.get("count", 10)))
            serror_rate   = st.slider("SYN Error Rate",   0.0, 1.0,
                                       float(defaults.get("serror_rate", 0.0)), 0.01)
            same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0,
                                       float(defaults.get("same_srv_rate", 0.9)), 0.01)
            cls_threshold = st.slider(
                "Classifier Threshold",
                0.10, 0.90,
                float(st.session_state.attack_threshold),
                0.01,
                help="Lower this for demo mode if you want attack presets to trigger more easily.",
            )
            st.session_state.attack_threshold = cls_threshold
            st.session_state.pipeline.THRESHOLD = cls_threshold

            if st.button("▶  Classify Traffic", type="primary", use_container_width=True):
                record = {c: 0 for c in FEATURE_COLS}
                record.update({
                    "duration": duration, "protocol_type": protocol_type,
                    "service": service, "flag": flag,
                    "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                    "count": count, "serror_rate": serror_rate,
                    "same_srv_rate": same_srv_rate,
                    "logged_in": int(defaults.get("logged_in", 1)),
                    "dst_host_count": int(defaults.get("dst_host_count", 128)),
                    "dst_host_srv_count": int(defaults.get("dst_host_srv_count", 64)),
                    "dst_host_same_srv_rate": float(defaults.get("dst_host_same_srv_rate", same_srv_rate)),
                    "rerror_rate": float(defaults.get("rerror_rate", 0.0)),
                    "diff_srv_rate": float(defaults.get("diff_srv_rate", 0.0)),
                    "dst_host_serror_rate": float(defaults.get("dst_host_serror_rate", serror_rate)),
                    "num_failed_logins": int(defaults.get("num_failed_logins", 0)),
                    "hot": int(defaults.get("hot", 0)),
                    "num_compromised": int(defaults.get("num_compromised", 0)),
                    "is_guest_login": int(defaults.get("is_guest_login", 0)),
                    "srv_count": int(defaults.get("srv_count", count)),
                    "srv_serror_rate": float(defaults.get("srv_serror_rate", serror_rate)),
                    "dst_host_srv_serror_rate": float(defaults.get("dst_host_srv_serror_rate", serror_rate)),
                })
                with st.spinner("Running Random Forest inference …"):
                    result = st.session_state.pipeline.predict(record)

                st.session_state.last_result = result
                st.session_state.last_record = record

                if st.session_state.db_connected:
                    ok = db_insert(st.session_state.db_conn, record, result, "manual")
                    st.toast("✅ Saved to database" if ok else "⚠️ DB save failed",
                             icon="🗄️" if ok else "⚠️")

        with col2:
            st.markdown("**🔎 Prediction Result**")
            res = st.session_state.last_result

            if res:
                if res["label"] == "Attack":
                    st.error(f"⚠️  **INTRUSION DETECTED** — confidence {res['confidence']}%")
                else:
                    st.success(f"✅  **NORMAL TRAFFIC** — confidence {res['confidence']}%")
                    st.caption(
                        f"This is below the current threshold. "
                        f"P(Attack) = {res['p_attack']:.2f}, threshold = {st.session_state.attack_threshold:.2f}."
                    )

                st.progress(int(res["confidence"]))
                m1, m2 = st.columns(2)
                m1.metric("P(Attack)", f"{res['p_attack']*100:.1f}%")
                m2.metric("P(Normal)", f"{res['p_normal']*100:.1f}%")

                st.markdown("---")
                st.caption(f"Decision threshold: P(Attack) ≥ {st.session_state.attack_threshold:.2f}")
                if st.session_state.db_connected:
                    if st.button("💾 Save This Result to Database",
                                 use_container_width=True):
                        ok = db_insert(
                            st.session_state.db_conn,
                            st.session_state.last_record,
                            res, "manual"
                        )
                        if ok:
                            st.success("✅ Prediction saved to the database!")
                        else:
                            st.error("❌ Save failed — check DB connection in sidebar.")
                else:
                    st.info("🔌 Connect a database in the sidebar to enable saving.")

                st.caption("Model: Random Forest (100 trees) · Dataset: NSL-KDD")
            else:
                st.info("👈 Configure parameters and click **Classify Traffic**.")

            st.markdown("**📊 Top Feature Importances**")
            if (st.session_state.model and
                    getattr(st.session_state.model, "feature_importances_", None)):
                top   = st.session_state.model.top_features(8)
                fi_df = pd.DataFrame(list(top.items()), columns=["Feature", "Importance"])
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="tealgrn")
                fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)",
                                     font_color="#74b9ff", margin=dict(t=10, b=10))
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.caption("Feature importances appear after model training.")


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📡 Real-Time Traffic Monitor")
    st.caption("Simulates the real-time inference pipeline (Algorithm 2 from paper)")

    if not st.session_state.model_loaded:
        st.warning("Please train or load a model first.")
    else:
        col_l, col_r = st.columns([1, 3])

        with col_l:
            n_packets  = st.slider("Packets to simulate", 5, 100, 20)
            attack_pct = st.slider("Simulated attack %",  10, 70,  40)

            if st.button("▶ Run Simulation", type="primary", use_container_width=True):
                protocols    = ["tcp", "udp", "icmp"]
                svc_normal   = ["http", "smtp", "ftp_data", "ssh", "telnet"]
                flags_attack = ["S0", "REJ"]
                new_logs     = []
                prog         = st.progress(0, text="Simulating …")

                for i in range(n_packets):
                    is_atk = random.random() < (attack_pct / 100)
                    if is_atk:
                        rec = {
                            "duration": 0, "protocol_type": "tcp",
                            "service": "private", "flag": random.choice(flags_attack),
                            "src_bytes": 0, "dst_bytes": 0,
                            "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                            "num_failed_logins": 0, "logged_in": 0,
                            "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
                            "num_root": 0, "num_file_creations": 0, "num_shells": 0,
                            "num_access_files": 0, "num_outbound_cmds": 0,
                            "is_host_login": 0, "is_guest_login": 0,
                            "count": random.randint(200, 511),
                            "srv_count": random.randint(200, 511),
                            "serror_rate": round(random.uniform(0.8, 1.0), 2),
                            "srv_serror_rate": round(random.uniform(0.8, 1.0), 2),
                            "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
                            "same_srv_rate": round(random.uniform(0.9, 1.0), 2),
                            "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
                            "dst_host_count": 255, "dst_host_srv_count": 10,
                            "dst_host_same_srv_rate": 0.04, "dst_host_diff_srv_rate": 0.06,
                            "dst_host_same_src_port_rate": 0.0,
                            "dst_host_srv_diff_host_rate": 0.0,
                            "dst_host_serror_rate": round(random.uniform(0.8, 1.0), 2),
                            "dst_host_srv_serror_rate": round(random.uniform(0.8, 1.0), 2),
                            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
                        }
                    else:
                        rec = {
                            "duration": random.randint(1, 100),
                            "protocol_type": random.choice(protocols),
                            "service": random.choice(svc_normal), "flag": "SF",
                            "src_bytes": random.randint(100, 20000),
                            "dst_bytes": random.randint(0, 10000),
                            "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                            "num_failed_logins": 0, "logged_in": 1,
                            "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
                            "num_root": 0, "num_file_creations": 0, "num_shells": 0,
                            "num_access_files": 0, "num_outbound_cmds": 0,
                            "is_host_login": 0, "is_guest_login": 0,
                            "count": random.randint(1, 20),
                            "srv_count": random.randint(1, 20),
                            "serror_rate": 0.0, "srv_serror_rate": 0.0,
                            "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
                            "same_srv_rate": round(random.uniform(0.7, 1.0), 2),
                            "diff_srv_rate": round(random.uniform(0.0, 0.3), 2),
                            "srv_diff_host_rate": 0.0,
                            "dst_host_count": random.randint(10, 255),
                            "dst_host_srv_count": random.randint(10, 255),
                            "dst_host_same_srv_rate": round(random.uniform(0.5, 1.0), 2),
                            "dst_host_diff_srv_rate": round(random.uniform(0.0, 0.2), 2),
                            "dst_host_same_src_port_rate": round(random.uniform(0, 0.5), 2),
                            "dst_host_srv_diff_host_rate": 0.0,
                            "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
                            "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
                        }

                    res = st.session_state.pipeline.predict(rec)
                    new_logs.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "protocol":  rec["protocol_type"],
                        "service":   rec["service"],
                        "flag":      rec["flag"],
                        "src_bytes": rec["src_bytes"],
                        "label":     res["label"],
                        "confidence": res["confidence"],
                    })
                    if st.session_state.db_connected:
                        db_insert(st.session_state.db_conn, rec, res, "simulation")

                    prog.progress((i + 1) / n_packets,
                                  text=f"Packet {i+1}/{n_packets} …")

                prog.empty()
                st.session_state.live_logs    = new_logs
                st.session_state.alert_count  = sum(1 for l in new_logs if l["label"] == "Attack")
                st.session_state.normal_count = sum(1 for l in new_logs if l["label"] == "Normal")
                if st.session_state.db_connected:
                    st.toast(f"✅ {len(new_logs)} packets saved to the database", icon="🗄️")

            if st.button("🗑️ Clear Logs", use_container_width=True):
                st.session_state.live_logs    = []
                st.session_state.alert_count  = 0
                st.session_state.normal_count = 0
                st.rerun()

        with col_r:
            total = st.session_state.alert_count + st.session_state.normal_count
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📦 Total Packets", total)
            m2.metric("🔴 Intrusions",    st.session_state.alert_count)
            m3.metric("🟢 Normal",        st.session_state.normal_count)
            m4.metric("⚡ Threat Rate",
                      f"{st.session_state.alert_count/total*100:.1f}%"
                      if total else "0%")

        if st.session_state.live_logs:
            st.markdown("**📋 Inference Log**")
            log_df = pd.DataFrame(st.session_state.live_logs)

            def _clr(val):
                return "color:#ff6b6b;font-weight:700" if val == "Attack" else "color:#00f5a0"

            st.dataframe(
                log_df.style.map(_clr, subset=["label"]),
                use_container_width=True, height=300,
            )
            csv_bytes = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Log as CSV",
                data=csv_bytes,
                file_name=f"nids_sim_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True,
            )
            pie_df = pd.DataFrame({
                "Type":  ["Normal", "Attack"],
                "Count": [st.session_state.normal_count, st.session_state.alert_count],
            })
            fig_pie = px.pie(pie_df, names="Type", values="Count",
                             color_discrete_sequence=["#00f5a0", "#ff6b6b"], hole=0.4)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#74b9ff")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("👈 Click **Run Simulation** to start.")


# ════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Dataset & Attack Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**NSL-KDD Attack Distribution**")
        attack_dist = pd.DataFrame({
            "Type":  ["Normal","Neptune","Guess Passwd","Mscan","Warezmaster",
                      "Apache2","Satan","Others"],
            "Count": [9711, 4657, 1231, 996, 944, 737, 735, 2472],
            "Color": ["#00f5a0","#f05454","#f5a623","#e056fd",
                      "#6c5ce7","#fd79a8","#e17055","#74b9ff"],
        })
        fig_att = px.pie(attack_dist, names="Type", values="Count",
                         color_discrete_sequence=attack_dist["Color"].tolist(), hole=0.35)
        fig_att.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#74b9ff")
        st.plotly_chart(fig_att, use_container_width=True)

    with col_b:
        st.markdown("**Feature Importance Ranking**")
        fi_data = pd.DataFrame([
            {"Feature": "src_bytes",              "Importance": 0.1613},
            {"Feature": "dst_bytes",              "Importance": 0.1300},
            {"Feature": "dst_host_same_srv_rate", "Importance": 0.0742},
            {"Feature": "dst_host_diff_srv_rate", "Importance": 0.0698},
            {"Feature": "service",                "Importance": 0.0682},
            {"Feature": "dst_host_rerror_rate",   "Importance": 0.0650},
            {"Feature": "dst_host_srv_count",     "Importance": 0.0601},
            {"Feature": "duration",               "Importance": 0.0344},
            {"Feature": "protocol_type",          "Importance": 0.0332},
        ])
        if (st.session_state.model_loaded and
                getattr(st.session_state.model, "feature_importances_", None)):
            top     = st.session_state.model.top_features(9)
            fi_data = pd.DataFrame(list(top.items()), columns=["Feature", "Importance"])

        fig_fi2 = px.bar(fi_data.sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", color="Importance",
                         color_continuous_scale=["#1e3a5f", "#0070f3", "#00f5a0"])
        fig_fi2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#74b9ff")
        st.plotly_chart(fig_fi2, use_container_width=True)

    st.markdown("---")
    st.subheader("🏗️ System Architecture")
    arch_cols = st.columns(5)
    arch = [
        ("📥", "Data Acquisition",   "NSL-KDD\n41 features\n125K+ records",    "#74b9ff"),
        ("⚙️", "Preprocessing",       "OHE encoding\nz-score norm\nCorr filter", "#ffd93d"),
        ("🔧", "Feature Engineering", "Corr. analysis\nImportance rank\nRFE",    "#e056fd"),
        ("🌲", "RF Classifier",       "100 trees\nmax_depth=20\nMajority vote",  "#00f5a0"),
        ("🗄️", "Database + Dashboard", "τ=0.5 threshold\nAlerts & logs\nDB save", "#ff6b6b"),
    ]
    for col, (icon, title, desc, color) in zip(arch_cols, arch):
        col.markdown(f"""
        <div style="background:{color}15;border:1px solid {color}44;border-radius:10px;
                    padding:14px;text-align:center;height:160px;">
            <div style="font-size:28px">{icon}</div>
            <div style="font-size:11px;font-weight:700;color:{color};margin:6px 0">{title}</div>
            <div style="font-size:10px;color:#4a6fa5;white-space:pre-line">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Paper: 'Network Intrusion Detection System Using Machine Learning' · "
               "Dataset: NSL-KDD · Model: Random Forest · Accuracy: 98.8%")


# ════════════════════════════════════════════════════════════════════════
# TAB 5 — DATABASE
# ════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🗄️ Prediction Database")

    if not st.session_state.db_connected:
        st.warning("🔌 Connect to SQLite or MySQL using the sidebar panel first.")
        st.markdown("---")
        st.markdown("### 📋 Step-by-step: Database Setup")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("""
**Option 1 — SQLite (Recommended)**  
Pick `SQLite` in the sidebar and keep the default file path.

**Option 2 — MySQL**  
Install the driver once:
```bash
pip install mysql-connector-python
```

Then open MySQL Workbench and start your local server.

**MySQL sidebar fields**

| Field    | Value |
|----------|-------|
| Host     | `localhost` |
| Port     | `3306` |
| Username | `root` |
| Password | your MySQL password |
| Database | `nids_db` |
            """)
        with col_i2:
            st.markdown("""
**Step 4 — Click 🔌 Connect**  
The database file or table is created automatically.

**Step 5 — Use the dashboard**  
Every Classify / Simulation result
is saved to the DB automatically.

**Step 6 — Verify in SQL tools**
```sql
SELECT * FROM nids_logs
ORDER BY logged_at DESC
LIMIT 50;
```
            """)

    else:
        if not db_ping(st.session_state.db_conn):
            st.session_state.db_connected = False
            st.error("❌ Connection lost. Reconnect in the sidebar.")
        else:
            total_rows = db_count(st.session_state.db_conn)
            db_target = (
                st.session_state.db_path
                if st.session_state.db_backend == "SQLite"
                else f"{st.session_state.db_host}:{st.session_state.db_port}/{st.session_state.db_name}"
            )
            st.success(
                f"✅ Connected to **{st.session_state.db_backend}** at "
                f"`{db_target}` "
                f"— **{total_rows}** total records"
            )

            # Controls
            ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
            with ctrl1:
                row_limit = st.number_input("Rows to load", 10, 1000, 200, step=10)
            with ctrl2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("🔄 Refresh", use_container_width=True, key="db_refresh")
            with ctrl3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Clear All Records", use_container_width=True):
                    if db_clear(st.session_state.db_conn):
                        st.success("✅ Table cleared.")
                        st.rerun()
                    else:
                        st.error("❌ Could not clear table.")

            db_df = db_fetch(st.session_state.db_conn, limit=int(row_limit))

            if db_df.empty:
                st.info("No records yet. Use **Classifier** or **Live Monitor** to generate data.")
            else:
                # Summary
                n_total  = len(db_df)
                n_attack = int((db_df["label"] == "Attack").sum()) if "label" in db_df.columns else 0
                n_normal = n_total - n_attack
                n_manual = int((db_df["source"] == "manual").sum()) if "source" in db_df.columns else 0
                n_sim    = int((db_df["source"] == "simulation").sum()) if "source" in db_df.columns else 0

                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("📦 Total",      n_total)
                d2.metric("🔴 Attacks",    n_attack)
                d3.metric("🟢 Normal",     n_normal)
                d4.metric("🖱️ Manual",     n_manual)
                d5.metric("🤖 Simulation", n_sim)

                st.markdown("---")

                # Coloured table
                def _lbl_clr(val):
                    return ("color:#ff6b6b;font-weight:700"
                            if val == "Attack" else "color:#00f5a0")

                show_cols = [c for c in
                             ["id","logged_at","protocol","service","flag",
                              "src_bytes","dst_bytes","label","confidence",
                              "p_attack","source","record_json"]
                             if c in db_df.columns]
                st.dataframe(
                    db_df[show_cols].style.map(_lbl_clr, subset=["label"]),
                    use_container_width=True, height=360,
                )

                # Charts
                st.markdown("---")
                ch1, ch2 = st.columns(2)
                with ch1:
                    pie_db = px.pie(db_df, names="label",
                                    color_discrete_sequence=["#00f5a0","#ff6b6b"],
                                    hole=0.4, title="Attack vs Normal (DB total)")
                    pie_db.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#74b9ff")
                    st.plotly_chart(pie_db, use_container_width=True)
                with ch2:
                    if "source" in db_df.columns:
                        src_df = db_df["source"].value_counts().reset_index()
                        src_df.columns = ["Source","Count"]
                        fig_src = px.bar(src_df, x="Source", y="Count", color="Count",
                                         color_continuous_scale=["#1e3a5f","#00f5a0"],
                                         title="Records by Source")
                        fig_src.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                               plot_bgcolor="rgba(0,0,0,0)",
                                               font_color="#74b9ff")
                        st.plotly_chart(fig_src, use_container_width=True)

                # Download
                csv_exp = db_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export DB Logs as CSV",
                    data=csv_exp,
                    file_name=f"nids_db_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True,
                )

                # Workbench SQL hint
                sql_label = "SQLite" if st.session_state.db_backend == "SQLite" else "MySQL Workbench"
                with st.expander(f"💡 Useful {sql_label} queries"):
                    st.code(f"""
-- Open your SQL client → New Query tab → paste and run:

{f"USE {st.session_state.db_name};" if st.session_state.db_backend == "MySQL" else ""}

-- View all records
SELECT * FROM nids_logs ORDER BY logged_at DESC;

-- Only attack records
SELECT * FROM nids_logs WHERE label = 'Attack' ORDER BY logged_at DESC;

-- Count by label
SELECT label, COUNT(*) AS total FROM nids_logs GROUP BY label;

-- Count by source
SELECT source, label, COUNT(*) AS cnt
FROM nids_logs GROUP BY source, label;

-- High confidence attacks
SELECT * FROM nids_logs
WHERE label='Attack' AND confidence > 90
ORDER BY confidence DESC;
""", language="sql")
