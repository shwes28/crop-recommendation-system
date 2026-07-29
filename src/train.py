"""
train.py — Training pipeline for Random Forest, SVM, and Naive Bayes classifiers.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_models():
    """Return dict of untrained model instances."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=2, random_state=42, n_jobs=-1,
        ),
        "SVM": CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
            ensemble=False,
        ),
        "Naive Bayes": GaussianNB(),
    }


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    classes: list[str],
    save: bool = True,
) -> dict:
    """
    Train all models, evaluate on test set, return results dict.

    Returns
    -------
    results : dict with keys = model names, values = dict of metrics
    """
    models   = build_models()
    results  = {}

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  Training : {name}")
        print(f"{'='*55}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        report = classification_report(y_test, y_pred,
                                       target_names=classes, output_dict=True)

        print(f"  Test Accuracy : {acc*100:.2f}%")
        print(f"  CV Accuracy   : {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=classes))

        results[name] = {
            "model":       model,
            "accuracy":    acc,
            "cv_mean":     cv.mean(),
            "cv_std":      cv.std(),
            "report":      report,
            "y_pred":      y_pred,
            "confusion":   confusion_matrix(y_test, y_pred),
        }

        # Feature importances only for RF
        if hasattr(model, "feature_importances_"):
            results[name]["feature_importances"] = model.feature_importances_

        if save:
            MODELS_DIR.mkdir(exist_ok=True)
            fname = name.lower().replace(" ", "_") + ".pkl"
            joblib.dump(model, MODELS_DIR / fname)
            print(f"  Saved → models/{fname}")

    # Also save the scaler and label encoder separately
    return results


def plot_confusion_matrix(confusion: np.ndarray, classes: list[str],
                          model_name: str) -> plt.Figure:
    """Return a styled confusion matrix figure."""
    BG      = "#0d1117"
    TEXT    = "#e6edf3"
    GRID    = "#30363d"

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                  display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True,
              xticks_rotation=45)

    ax.set_title(f"{model_name} — Confusion Matrix",
                 color=TEXT, pad=15, fontsize=13, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    fig.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    """Grouped bar chart comparing accuracy and CV score."""
    BG      = "#0d1117"
    SURFACE = "#161b22"
    TEXT    = "#e6edf3"
    GRID    = "#30363d"
    COLORS  = ["#58a6ff", "#3fb950", "#f78166"]

    names    = list(results.keys())
    accs     = [results[n]["accuracy"] * 100 for n in names]
    cv_means = [results[n]["cv_mean"]  * 100 for n in names]
    cv_stds  = [results[n]["cv_std"]   * 100 for n in names]

    x    = np.arange(len(names))
    width = 0.35

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": GRID, "axes.grid": True,
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)

    bars1 = ax.bar(x - width/2, accs,     width, label="Test Accuracy",
                   color=COLORS[0], alpha=0.85, edgecolor=GRID)
    bars2 = ax.bar(x + width/2, cv_means, width, label="CV Accuracy (5-fold)",
                   color=COLORS[1], alpha=0.85, edgecolor=GRID,
                   yerr=cv_stds, capsize=5,
                   error_kw={"ecolor": TEXT, "linewidth": 1.5})

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 102)
    ax.set_title("Model Comparison — Test vs Cross-Validation Accuracy",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.3, labelcolor=TEXT)
    fig.tight_layout()
    return fig
