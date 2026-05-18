"""
Train a full 24-class ASL letter classifier on data from collect_data.py.

Usage:
    python train_classifier.py

Input:  asl_data.npz
Output: asl_classifier.pkl   (auto-loaded by classifier.py at import time)
"""
import sys
import numpy as np

try:
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.metrics import classification_report
    import joblib
except ImportError:
    print("[ERROR] scikit-learn is required:  pip install scikit-learn")
    sys.exit(1)

DATA_PATH = "asl_data.npz"
SAVE_PATH = "asl_classifier.pkl"
MIN_SAMPLES_PER_CLASS = 20


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        data = np.load(DATA_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"[ERROR] '{DATA_PATH}' not found.")
        print("Run collect_data.py first.")
        sys.exit(1)

    X      = data["X"].astype(np.float64)
    y      = data["y"].astype(int)
    labels = list(data["labels"])

    print(f"Loaded {len(X)} samples  ({X.shape[1]}-d features,  {len(labels)} classes)\n")

    warn = False
    for i, lt in enumerate(labels):
        n      = (y == i).sum()
        note   = "  ← low" if n < MIN_SAMPLES_PER_CLASS else ""
        print(f"  {lt}: {n:>4}{note}")
        if n < MIN_SAMPLES_PER_CLASS:
            warn = True
    if warn:
        print("\n[WARN] Some classes have few samples — accuracy may be reduced.")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split ────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Try SVM ───────────────────────────────────────────────────────────────
    print("\n── SVM (RBF kernel) ──────────────────────────────────────")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm_cv = cross_val_score(svm, X_scaled, y, cv=cv).mean()
    print(f"5-fold CV: {svm_cv:.1%}")
    svm.fit(X_tr, y_tr)
    svm_te = svm.score(X_te, y_te)
    print(f"Hold-out : {svm_te:.1%}")

    # ── Try MLP ───────────────────────────────────────────────────────────────
    print("\n── MLP (128-64) ──────────────────────────────────────────")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                        random_state=42, early_stopping=True)
    mlp_cv = cross_val_score(mlp, X_scaled, y, cv=cv).mean()
    print(f"5-fold CV: {mlp_cv:.1%}")
    mlp.fit(X_tr, y_tr)
    mlp_te = mlp.score(X_te, y_te)
    print(f"Hold-out : {mlp_te:.1%}")

    # ── Pick the better model ─────────────────────────────────────────────────
    if svm_cv >= mlp_cv:
        best, best_name = svm, "SVM"
    else:
        best, best_name = mlp, "MLP"

    print(f"\nSelected: {best_name}  (CV {max(svm_cv, mlp_cv):.1%})")

    # ── Full classification report ────────────────────────────────────────────
    print("\nHold-out classification report:")
    print(classification_report(y_te, best.predict(X_te), target_names=labels))

    # ── Retrain on all data ───────────────────────────────────────────────────
    best.fit(X_scaled, y)

    bundle = {"scaler": scaler, "clf": best, "labels": labels, "model": best_name}
    joblib.dump(bundle, SAVE_PATH)
    print(f"Model saved → '{SAVE_PATH}'")
    print("Restart digital_twin.py / main.py to use the new classifier.")


if __name__ == "__main__":
    main()
