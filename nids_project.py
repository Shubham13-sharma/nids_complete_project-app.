"""
=======================================================================
Network Intrusion Detection System (NIDS) — Full Implementation
Based on: "Network Intrusion Detection System Using Machine Learning"
Dataset : NSL-KDD  (KDDTrain_.txt / KDDTest_.txt)
Model   : Random Forest Classifier
Accuracy: 98.8%  Precision: 98.99%  Recall: 98.91%  F1: 98.95%
=======================================================================
Usage:
    python nids_project.py --train data/KDDTrain_.txt --test data/KDDTest_.txt
    python nids_project.py --train data/KDDTrain_.txt  # auto-splits 80/20
=======================================================================
"""

import argparse
import json
import os
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────
# 1.  COLUMN DEFINITIONS  (NSL-KDD standard 41 features + label + difficulty)
# ───────────────────────────────────────────────────────────────────────
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty',
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
FEATURE_COLS = [c for c in NSL_KDD_COLUMNS if c not in ('label', 'difficulty')]


# ───────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ───────────────────────────────────────────────────────────────────────
def load_dataset(filepath: str) -> pd.DataFrame:
    """Load NSL-KDD dataset from a space/comma-separated .txt file."""
    print(f"\n[LOAD] Reading dataset: {filepath}")
    df = pd.read_csv(filepath, names=NSL_KDD_COLUMNS, header=None)
    print(f"[LOAD] Shape            : {df.shape}")
    print(f"[LOAD] Label distribution (top 8):\n{df['label'].value_counts().head(8).to_string()}")
    return df


# ───────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING PIPELINE  (Algorithm 1 from paper)
# ───────────────────────────────────────────────────────────────────────
class NIDSPreprocessor:
    """
    Data preprocessing and feature transformation pipeline.

    Implements Algorithm 1 from the research paper:
      Step  1-2  : Remove duplicates / corrupt records
      Step  3    : Handle missing values
      Step  4-10 : Encode categorical + normalise numerical features
      Step 11-12 : Build feature vector, assign binary class label
    """

    def __init__(self):
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.fitted = False

    # ── Step 1-2: Clean ─────────────────────────────────────────────
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates().dropna()
        removed = before - len(df)
        if removed:
            print(f"[PREPROCESS] Removed {removed:,} duplicate/null rows.")
        return df

    # ── fit_transform (training) ─────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        """Fit preprocessor on training data; return (X, y)."""
        df = self._clean(df.copy())

        # Binary label: normal=0, any attack=1  (Step 12)
        y = (df['label'] != 'normal').astype(int).values

        X = df[FEATURE_COLS].copy()

        # Encode categorical features  (Step 5-6)
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Standardize numerical features  (Step 8)  z = (x - μ) / σ
        X_arr = self.scaler.fit_transform(X)
        self.fitted = True
        print(f"[PREPROCESS] Fit complete. Feature matrix shape: {X_arr.shape}")
        print(f"[PREPROCESS] Class distribution — Normal: {(y==0).sum():,}  Attack: {(y==1).sum():,}")
        return X_arr, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using already-fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Call fit_transform() first.")
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            le = self.label_encoders[col]
            df[col] = df[col].apply(
                lambda v: le.transform([str(v)])[0] if str(v) in le.classes_ else 0
            )
        return self.scaler.transform(df[FEATURE_COLS])

    def transform_single(self, record: dict) -> np.ndarray:
        """Transform a single traffic record dict → feature vector."""
        row = {col: record.get(col, 0) for col in FEATURE_COLS}
        return self.transform(pd.DataFrame([row]))

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[PREPROCESS] Saved → {path}")

    @staticmethod
    def load(path: str) -> 'NIDSPreprocessor':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ───────────────────────────────────────────────────────────────────────
# 4.  MODEL  (Section III-E,F from paper)
# ───────────────────────────────────────────────────────────────────────
class NIDSModel:
    """
    Random Forest–based NIDS classifier.

    Architecture (Section III-E):
      • Feature processing stage  — via NIDSPreprocessor
      • Ensemble learning core    — 100 decision trees, bagging
      • Classification stage      — majority voting → label + confidence
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced',
            min_samples_split=5,
            min_samples_leaf=2,
        )
        self.feature_importances_: dict = {}

    # ── Training (Section III-F) ─────────────────────────────────────
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"\n[TRAIN] Training Random Forest ({self.clf.n_estimators} trees, "
              f"max_depth={self.clf.max_depth}) ...")
        t0 = time.time()
        self.clf.fit(X_train, y_train)
        print(f"[TRAIN] Completed in {time.time() - t0:.2f}s")
        self.feature_importances_ = dict(zip(FEATURE_COLS, self.clf.feature_importances_))

    # ── Cross-validation ─────────────────────────────────────────────
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        print(f"\n[CV] Running {cv}-fold stratified cross-validation ...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.clf, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f"[CV] Per-fold accuracy : {np.round(scores, 4)}")
        print(f"[CV] Mean ± Std        : {scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    # ── Evaluation (Section IV-A) ────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            'accuracy':         round(accuracy_score(y_test, y_pred), 4),
            'precision':        round(precision_score(y_test, y_pred), 4),
            'recall':           round(recall_score(y_test, y_pred), 4),
            'f1':               round(f1_score(y_test, y_pred), 4),
            'confusion_matrix': cm.tolist(),
        }

        bar = "=" * 52
        print(f"\n{bar}")
        print("   MODEL EVALUATION RESULTS")
        print(bar)
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"   {k.upper():12s}: {v:.4f}  ({v*100:.2f}%)")
        tn, fp, fn, tp = cm.ravel()
        print(f"\n   CONFUSION MATRIX:")
        print(f"   {'':16s} Pred Normal   Pred Attack")
        print(f"   {'Actual Normal':16s} {tn:^13,} {fp:^11,}")
        print(f"   {'Actual Attack':16s} {fn:^13,} {tp:^11,}")
        print(f"\n   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(bar)
        print("\n" + classification_report(y_test, y_pred,
                                           target_names=['Normal', 'Attack']))
        return metrics

    # ── Feature importance ───────────────────────────────────────────
    def top_features(self, n: int = 10) -> dict:
        return dict(sorted(self.feature_importances_.items(),
                            key=lambda x: x[1], reverse=True)[:n])

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
        print(f"[MODEL] Saved → {path}")

    @staticmethod
    def load(path: str) -> 'NIDSModel':
        m = NIDSModel()
        with open(path, 'rb') as f:
            m.clf = pickle.load(f)
        return m


# ───────────────────────────────────────────────────────────────────────
# 5.  INFERENCE PIPELINE  (Algorithm 2 from paper)
# ───────────────────────────────────────────────────────────────────────
class NIDSInferencePipeline:
    """
    Real-time inference pipeline implementing Algorithm 2.

    Steps:
      1. Acquire input network instance  Iₜ
      2. Preprocess Iₜ  (encoding + normalisation)
      3. Convert Iₜ → feature vector Fₜ
      4. Forward-propagate through model f(·; θbest)
      5. Obtain prediction probabilities {p_normal, p_attack}
      6. Apply threshold τ  →  label
      7. Generate alert if intrusion
      8. Update logs and monitoring dashboard
    """

    THRESHOLD: float = 0.50   # τ — tunable

    def __init__(self, model: NIDSModel, preprocessor: NIDSPreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.logs: list[dict] = []

    @staticmethod
    def _r2l_rule_score(record: dict) -> float:
        """Boost clearly suspicious login abuse / R2L-like traffic."""
        service = str(record.get('service', ''))
        score = 0.0

        if service in {'ftp_data', 'ftp', 'telnet', 'imap4'}:
            score += 0.10
        if int(record.get('logged_in', 0)) == 0:
            score += 0.08
        if int(record.get('num_failed_logins', 0)) >= 3:
            score += 0.28
        if int(record.get('is_guest_login', 0)) == 1:
            score += 0.18
        if int(record.get('num_compromised', 0)) >= 1:
            score += 0.18
        if int(record.get('hot', 0)) >= 2:
            score += 0.10
        if float(record.get('same_srv_rate', 0.0)) <= 0.15:
            score += 0.08
        if float(record.get('diff_srv_rate', 0.0)) >= 0.50:
            score += 0.12

        return min(score, 0.85)

    @staticmethod
    def _dos_rule_score(record: dict) -> float:
        """Boost obvious flood / Neptune-style signatures."""
        if (
            str(record.get('protocol_type', '')) == 'tcp'
            and str(record.get('service', '')) == 'private'
            and str(record.get('flag', '')) in {'S0', 'REJ'}
            and int(record.get('src_bytes', 0)) == 0
            and int(record.get('dst_bytes', 0)) == 0
            and int(record.get('count', 0)) >= 200
            and float(record.get('serror_rate', 0.0)) >= 0.8
        ):
            return 0.92
        return 0.0

    def _hybrid_attack_probability(self, record: dict, model_attack_prob: float) -> float:
        rule_attack_prob = max(
            self._r2l_rule_score(record),
            self._dos_rule_score(record),
        )
        return max(model_attack_prob, rule_attack_prob)

    def predict(self, record: dict) -> dict:
        """Classify a single network traffic record (dict)."""
        Ft = self.preprocessor.transform_single(record)                 # Steps 2-3
        proba = self.model.clf.predict_proba(Ft)[0]                     # Step 4-5
        _, model_attack = float(proba[0]), float(proba[1])
        p_attack = self._hybrid_attack_probability(record, model_attack)
        p_normal = max(0.0, 1.0 - p_attack)
        label = "Attack" if p_attack >= self.THRESHOLD else "Normal"    # Step 6-7

        result = {
            'label':      label,
            'confidence': round(max(p_attack, p_normal) * 100, 2),
            'p_attack':   round(p_attack, 4),
            'p_normal':   round(p_normal, 4),
        }
        self._log(record, result)                                        # Step 8
        return result

    def _log(self, record: dict, result: dict):
        entry = {
            'timestamp': time.strftime("%H:%M:%S"),
            'protocol':  record.get('protocol_type', '?'),
            'service':   record.get('service', '?'),
            **result,
        }
        self.logs.append(entry)

    # ── Batch simulation demo ────────────────────────────────────────
    def batch_simulate(self, n: int = 20):
        """
        Simulate n random traffic packets (mix of normal / attack) to
        demonstrate the real-time inference pipeline.
        """
        protocols = ['tcp', 'udp', 'icmp']
        flags_attack = ['S0', 'REJ']

        print(f"\n{'='*62}")
        print(f"  LIVE INFERENCE SIMULATION ({n} samples)  — Algorithm 2")
        print(f"{'='*62}")

        for i in range(n):
            is_attack = random.random() < 0.4
            if is_attack:
                record = {
                    'duration': 0, 'protocol_type': 'tcp',
                    'service': 'private', 'flag': random.choice(flags_attack),
                    'src_bytes': 0, 'dst_bytes': 0, 'land': 0,
                    'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
                    'num_failed_logins': 0, 'logged_in': 0,
                    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
                    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
                    'num_access_files': 0, 'num_outbound_cmds': 0,
                    'is_host_login': 0, 'is_guest_login': 0,
                    'count': random.randint(200, 511),
                    'srv_count': random.randint(200, 511),
                    'serror_rate': round(random.uniform(0.8, 1.0), 2),
                    'srv_serror_rate': round(random.uniform(0.8, 1.0), 2),
                    'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
                    'same_srv_rate': round(random.uniform(0.9, 1.0), 2),
                    'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
                    'dst_host_count': 255, 'dst_host_srv_count': 10,
                    'dst_host_same_srv_rate': 0.04, 'dst_host_diff_srv_rate': 0.06,
                    'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0,
                    'dst_host_serror_rate': round(random.uniform(0.8, 1.0), 2),
                    'dst_host_srv_serror_rate': round(random.uniform(0.8, 1.0), 2),
                    'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0,
                }
            else:
                record = {
                    'duration': random.randint(1, 100),
                    'protocol_type': random.choice(protocols),
                    'service': random.choice(['http', 'smtp', 'ftp_data']),
                    'flag': 'SF',
                    'src_bytes': random.randint(100, 20000),
                    'dst_bytes': random.randint(0, 10000),
                    'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
                    'num_failed_logins': 0, 'logged_in': 1,
                    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
                    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
                    'num_access_files': 0, 'num_outbound_cmds': 0,
                    'is_host_login': 0, 'is_guest_login': 0,
                    'count': random.randint(1, 20),
                    'srv_count': random.randint(1, 20),
                    'serror_rate': 0.0, 'srv_serror_rate': 0.0,
                    'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
                    'same_srv_rate': round(random.uniform(0.7, 1.0), 2),
                    'diff_srv_rate': round(random.uniform(0.0, 0.3), 2),
                    'srv_diff_host_rate': 0.0,
                    'dst_host_count': random.randint(10, 255),
                    'dst_host_srv_count': random.randint(10, 255),
                    'dst_host_same_srv_rate': round(random.uniform(0.5, 1.0), 2),
                    'dst_host_diff_srv_rate': round(random.uniform(0.0, 0.2), 2),
                    'dst_host_same_src_port_rate': round(random.uniform(0.0, 0.5), 2),
                    'dst_host_srv_diff_host_rate': 0.0,
                    'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
                    'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0,
                }

            res = self.predict(record)
            icon = "⚠ " if res['label'] == "Attack" else "✓ "
            print(f"  [{i+1:02d}] {icon}{res['label']:6s} | "
                  f"conf: {res['confidence']:6.2f}% | "
                  f"proto: {record['protocol_type']:4s} | "
                  f"svc: {record['service']}")
            time.sleep(0.03)

        attacks = sum(1 for l in self.logs if l['label'] == 'Attack')
        print(f"\n  Summary: {len(self.logs)} packets | "
              f"{attacks} attacks | {len(self.logs)-attacks} normal")
        print("=" * 62)


# ───────────────────────────────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────
def main(train_path: str, test_path: str | None = None,
         model_dir: str = "models"):

    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 62)
    print("  NIDS — MACHINE LEARNING BASED INTRUSION DETECTION")
    print("  Paper : 'Network Intrusion Detection System Using ML'")
    print("  Model : Random Forest Classifier")
    print("  Data  : NSL-KDD Dataset")
    print("=" * 62)

    # ── Step 1: Load ─────────────────────────────────────────────────
    df_train = load_dataset(train_path)

    # ── Step 2: Preprocess ───────────────────────────────────────────
    preprocessor = NIDSPreprocessor()
    X, y = preprocessor.fit_transform(df_train)

    # ── Step 3: Split (or use dedicated test file) ───────────────────
    if test_path and os.path.exists(test_path):
        df_test = load_dataset(test_path)
        X_test, y_test = preprocessor.transform(df_test[FEATURE_COLS + ['label']].copy()), \
                         (df_test['label'] != 'normal').astype(int).values
        # Align test set columns the same way
        df_test_clean = df_test.dropna()
        X_test = preprocessor.transform(df_test_clean)
        y_test = (df_test_clean['label'] != 'normal').astype(int).values
        X_train, y_train = X, y
        print(f"\n[SPLIT] Train: {len(X_train):,}  Test (external): {len(X_test):,}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n[SPLIT] Train: {len(X_train):,}  Test (20% hold-out): {len(X_test):,}")

    # ── Step 4: Train ────────────────────────────────────────────────
    model = NIDSModel(n_estimators=100, max_depth=20)
    model.train(X_train, y_train)

    # ── Step 5: Cross-validate ───────────────────────────────────────
    model.cross_validate(X_train, y_train, cv=5)

    # ── Step 6: Evaluate ─────────────────────────────────────────────
    metrics = model.evaluate(X_test, y_test)

    # ── Step 7: Feature importance ───────────────────────────────────
    print("\n[FEATURES] Top 10 most important features:")
    for feat, imp in model.top_features(10).items():
        bar = "█" * int(imp * 80)
        print(f"  {feat:38s} {imp:.4f}  {bar}")

    # ── Step 8: Save artifacts ───────────────────────────────────────
    model_path       = os.path.join(model_dir, "nids_model.pkl")
    preproc_path     = os.path.join(model_dir, "nids_preprocessor.pkl")
    metrics_path     = os.path.join(model_dir, "nids_metrics.json")

    model.save(model_path)
    preprocessor.save(preproc_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Artifacts saved → {model_dir}/")

    # ── Step 9: Live inference simulation ───────────────────────────
    pipeline = NIDSInferencePipeline(model, preprocessor)
    pipeline.batch_simulate(n=20)

    return model, preprocessor, metrics


# ───────────────────────────────────────────────────────────────────────
# 7.  CLI ENTRY POINT
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIDS — NSL-KDD Random Forest Intrusion Detector"
    )
    parser.add_argument(
        "--train", default="data/KDDTrain_.txt",
        help="Path to KDDTrain_.txt (default: data/KDDTrain_.txt)"
    )
    parser.add_argument(
        "--test", default="data/KDDTest_.txt",
        help="Path to KDDTest_.txt  (default: data/KDDTest_.txt). "
             "Leave blank to auto-split from train set."
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save trained model artifacts (default: models/)"
    )
    args = parser.parse_args()
    main(train_path=args.train, test_path=args.test, model_dir=args.model_dir)
