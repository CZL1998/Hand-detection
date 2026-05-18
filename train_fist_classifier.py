"""
Train an SVM classifier on data collected by collect_fist_data.py.

Usage:
    python train_fist_classifier.py

Input:  fist_data.npz
Output: fist_classifier.pkl   (auto-loaded by classifier.py at import time)
"""
import sys
import numpy as np

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.metrics import classification_report
    import joblib
except ImportError:
    print("[ERROR] scikit-learn is required:  pip install scikit-learn")
    sys.exit(1)

LABELS    = ["E", "M", "N", "S", "T"]
DATA_PATH = "fist_data.npz"
SAVE_PATH = "fist_classifier.pkl"


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        data = np.load(DATA_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"[ERROR] '{DATA_PATH}' not found.")
        print("Run collect_fist_data.py first to gather training samples.")
        sys.exit(1)

    X      = data["X"].astype(np.float64)
    y      = data["y"].astype(int)
    labels = list(data.get("labels", np.array(LABELS)))

    print(f"Loaded {len(X)} samples  ({X.shape[1]}-d features)")
    for i, lt in enumerate(labels):
        n = (y == i).sum()
        status = "" if n >= 20 else "  ← low sample count, consider collecting more"
        print(f"  {lt}: {n}{status}")

    if len(X) < len(labels) * 10:
        print("\n[WARN] Very few samples. Accuracy may be low.")

    # ── Scale features ────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    clf_cv = SVC(kernel="rbf", C=10, gamma="scale")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf_cv, X_scaled, y, cv=cv)
    print(f"\n5-fold CV accuracy: {scores.mean():.1%}  (± {scores.std():.1%})")

    # ── Hold-out evaluation ───────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y)

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    clf.fit(X_tr, y_tr)

    print("\nHold-out classification report:")
    print(classification_report(y_te, clf.predict(X_te), target_names=labels))

    # ── Retrain on full dataset for deployment ────────────────────────────────
    clf.fit(X_scaled, y)

    bundle = {"scaler": scaler, "clf": clf, "labels": labels}
    joblib.dump(bundle, SAVE_PATH)
    print(f"Model saved → '{SAVE_PATH}'")
    print("Restart digital_twin.py or main.py to pick up the new model.")


if __name__ == "__main__":
    main()
