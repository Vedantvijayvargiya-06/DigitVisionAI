"""
DigitVision AI - Handwritten Digit Recognition
Minor Project 7: CNN Image Classification and Streamlit Application

Student : Vedant Vijay
Dataset : MNIST (handwritten digits 0-9)

The file has two parts that share the same functions:
    PART A - CNN image classification (load, explore, preprocess, build, train, evaluate)
    PART B - Streamlit application that uses the CNN trained in Part A

HOW TO RUN
    1. Install libraries:
           pip install tensorflow streamlit numpy matplotlib seaborn scikit-learn pillow
    2. Part A (CNN workflow in the terminal, saves graphs in ./part_a_outputs):
           python VedantVijay_CNN_Streamlit_Project.py --train
    3. Part B (web application):
           streamlit run VedantVijay_CNN_Streamlit_Project.py
       On the first launch the app trains the CNN (a few minutes) and keeps it in
       the system temp folder, so later launches open instantly. No model file is needed.

References used for learning: Keras and Streamlit official documentation.
"""

import argparse
import gzip
import io
import json
import os
import tempfile
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep the terminal clean

import matplotlib

matplotlib.use("Agg")  # draw figures without a pop-up window
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from streamlit.runtime.scriptrunner import get_script_run_ctx
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# SHARED SETTINGS
# =============================================================================
APP_NAME = "DigitVision AI"
CLASS_NAMES = [str(digit) for digit in range(10)]
IMG_SIZE = 28
EPOCHS = 8
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
SEED = 42

CACHE_DIR = os.path.join(tempfile.gettempdir(), "digitvision_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "digitvision_cnn.keras")
METRICS_PATH = os.path.join(CACHE_DIR, "digitvision_metrics.json")

# Backup download source (used only if the Keras download is blocked)
MNIST_MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

MAX_UPLOAD_MB = 5
COLORS = {"primary": "#4F46E5", "accent": "#0EA5E9", "good": "#16A34A",
          "bad": "#DC2626", "muted": "#64748B", "text": "#0F172A"}


# =============================================================================
# PART A - CNN IMAGE CLASSIFICATION
# =============================================================================

# ---- A1. Load the dataset ----------------------------------------------------
def _download_idx_file(file_name):
    """Download one MNIST file from the backup mirror (only if not already saved)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, file_name)
    if not os.path.exists(local_path):
        with urllib.request.urlopen(MNIST_MIRROR + file_name, timeout=60) as response:
            data = response.read()
        with open(local_path + ".part", "wb") as file:
            file.write(data)
        os.replace(local_path + ".part", local_path)
    with gzip.open(local_path, "rb") as file:
        return file.read()


def _load_mnist_from_mirror():
    """Read the raw IDX files: images start at byte 16, labels at byte 8."""
    raw = {key: _download_idx_file(name) for key, name in MNIST_FILES.items()}
    x_train = np.frombuffer(raw["x_train"], dtype=np.uint8, offset=16).reshape(-1, IMG_SIZE, IMG_SIZE)
    x_test = np.frombuffer(raw["x_test"], dtype=np.uint8, offset=16).reshape(-1, IMG_SIZE, IMG_SIZE)
    y_train = np.frombuffer(raw["y_train"], dtype=np.uint8, offset=8)
    y_test = np.frombuffer(raw["y_test"], dtype=np.uint8, offset=8)
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    """Load MNIST with Keras; if the download fails, use the backup mirror."""
    try:
        return keras.datasets.mnist.load_data()
    except Exception as error:  # network / firewall problems
        print(f"Keras download failed ({error}). Trying backup mirror...")
        return _load_mnist_from_mirror()


# ---- A2. Explore the dataset -------------------------------------------------
def explore_dataset(x_train, y_train, x_test, y_test):
    """Print the basic facts about the dataset and return them for the report."""
    info = {
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "image_shape": [int(v) for v in x_train.shape[1:]],
        "num_classes": int(len(np.unique(y_train))),
        "pixel_min": int(x_train.min()),
        "pixel_max": int(x_train.max()),
    }
    print("\n=== DATASET EXPLORATION ===")
    print(f"x_train shape : {x_train.shape}   y_train shape : {y_train.shape}")
    print(f"x_test shape  : {x_test.shape}   y_test shape  : {y_test.shape}")
    print(f"Training samples: {info['train_samples']} | Testing samples: {info['test_samples']}")
    print(f"Image size: {info['image_shape'][0]}x{info['image_shape'][1]} (grayscale)")
    print(f"Number of classes: {info['num_classes']} -> {CLASS_NAMES}")
    print(f"Pixel range: {info['pixel_min']} to {info['pixel_max']}")
    print("Images per digit (train):", dict(zip(*[v.tolist() for v in np.unique(y_train, return_counts=True)])))
    return info


# ---- A3. Preprocess ----------------------------------------------------------
def preprocess_data(x_train, y_train, x_test, y_test):
    """Normalise pixels to 0-1, add the channel axis, and one-hot encode labels."""
    x_train_cnn = (x_train.astype("float32") / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test_cnn = (x_test.astype("float32") / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes=10)
    y_test_onehot = keras.utils.to_categorical(y_test, num_classes=10)
    return x_train_cnn, y_train_onehot, x_test_cnn, y_test_onehot


# ---- A4. Build, compile, train -------------------------------------------------
def build_cnn():
    """Two Conv+Pool blocks to find patterns, then Dense layers to decide the digit."""
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),   # finds edges / strokes
        layers.MaxPooling2D((2, 2)),                    # shrinks the feature maps
        layers.Conv2D(64, (3, 3), activation="relu"),   # finds shapes (loops, curves)
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),                           # randomly switches off neurons
        layers.Flatten(),                               # 2D maps -> 1D vector
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),         # 10 probabilities (sum = 1)
    ], name="DigitVision_CNN")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, x_train, y_train, epochs=EPOCHS, verbose=2):
    """Train the CNN; 10% of the training data is kept aside for validation."""
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT, verbose=verbose)
    return history.history


# ---- A5. Evaluate ------------------------------------------------------------
def evaluate_model(model, x_test, y_test_onehot, y_test):
    """Test loss/accuracy, predictions, confusion matrix and classification report."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                                       output_dict=True, digits=4),
    }
    return results, y_pred, probabilities


def get_model_summary_lines(model):
    """Capture model.summary() as text so it can be shown in the app and report."""
    lines = []
    model.summary(print_fn=lines.append, line_length=78)
    return lines


# ---- A6. Plots (shared by Part A terminal mode and Part B app) -------------------
def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)


def fig_sample_images(x_raw, y_raw, n=10):
    """First n training images with their actual labels."""
    fig, axes = plt.subplots(2, 5, figsize=(9, 5))
    for ax, image, label in zip(axes.ravel(), x_raw[:n], y_raw[:n]):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {label}", fontsize=10, color=COLORS["text"])
        ax.axis("off")
    fig.suptitle("Sample MNIST images with actual labels", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_model_summary(summary_lines):
    """Draw the model summary text as an image."""
    fig, ax = plt.subplots(figsize=(7.2, 0.19 * len(summary_lines) + 0.3))
    ax.text(0.0, 1.0, "\n".join(summary_lines), family="monospace", fontsize=8,
            va="top", linespacing=1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def fig_training_curve(history, metric):
    """Training vs validation curve for 'accuracy' or 'loss'."""
    epochs_axis = range(1, len(history[metric]) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(epochs_axis, history[metric], marker="o", color=COLORS["primary"], label="Training")
    ax.plot(epochs_axis, history["val_" + metric], marker="s", color=COLORS["accent"], label="Validation")
    ax.set_title(f"Training vs Validation {metric.capitalize()}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return fig


def fig_confusion_matrix(matrix):
    """Heatmap: rows = actual digit, columns = predicted digit."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    ax.set_title("Confusion Matrix (test data)", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_predictions(x_raw, y_true, y_pred, probabilities, show_correct, n=10):
    """Grid of correct (green) or incorrect (red) test predictions."""
    mask = (y_pred == y_true) if show_correct else (y_pred != y_true)
    indices = np.where(mask)[0]
    rng = np.random.default_rng(SEED)
    indices = rng.choice(indices, size=min(n, len(indices)), replace=False)
    color = COLORS["good"] if show_correct else COLORS["bad"]
    fig, axes = plt.subplots(2, 5, figsize=(9, 5.0))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(x_raw[idx], cmap="gray")
        ax.set_title(f"Actual: {y_true[idx]}  Pred: {y_pred[idx]}\n{probabilities[idx].max() * 100:.1f}% sure",
                     fontsize=9, color=color)
    fig.suptitle("Correct predictions" if show_correct else "Incorrect predictions", fontweight="bold")
    fig.tight_layout()
    return fig


def explain_results(metrics):
    """Plain-language explanation built from the real numbers."""
    acc = metrics["test_accuracy"] * 100
    wrong = round((1 - metrics["test_accuracy"]) * metrics["dataset"]["test_samples"])
    final_train = metrics["history"]["accuracy"][-1] * 100
    final_val = metrics["history"]["val_accuracy"][-1] * 100
    report = metrics["classification_report"]
    per_class_f1 = {d: report[d]["f1-score"] for d in CLASS_NAMES}
    best = max(per_class_f1, key=per_class_f1.get)
    worst = min(per_class_f1, key=per_class_f1.get)
    return [
        f"The CNN was tested on {metrics['dataset']['test_samples']} images it had never seen. "
        f"It got {acc:.2f}% of them right and made {wrong} mistakes (test loss {metrics['test_loss']:.4f}).",
        f"After the last epoch, training accuracy was {final_train:.2f}% and validation accuracy was "
        f"{final_val:.2f}%. These two are close, so the model learned general patterns instead of memorising.",
        f"The easiest digit was {best} (F1-score {per_class_f1[best]:.3f}) and the hardest was {worst} "
        f"(F1-score {per_class_f1[worst]:.3f}). Mistakes usually happen between digits that look alike.",
    ]


# ---- A7. Complete workflow + saving the trained model ------------------------------
def save_artifacts(model, metrics):
    os.makedirs(CACHE_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file)


def load_artifacts():
    """Return (model, metrics) from the temp cache, or None if missing/broken."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        return None
    try:
        with open(METRICS_PATH, encoding="utf-8") as file:
            metrics = json.load(file)
        return keras.models.load_model(MODEL_PATH), metrics
    except Exception as error:
        print(f"Cached model could not be loaded ({error}). It will be retrained.")
        return None


def train_and_evaluate(epochs=EPOCHS, verbose=2):
    """Whole Part A pipeline. Used by the terminal mode and by the Streamlit app."""
    keras.utils.set_random_seed(SEED)
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist()
    dataset_info = explore_dataset(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
    x_train, y_train, x_test, y_test_onehot = preprocess_data(x_train_raw, y_train_raw, x_test_raw, y_test_raw)

    model = build_cnn()
    summary_lines = get_model_summary_lines(model)  # taken before training: no optimizer parameters yet
    history = train_model(model, x_train, y_train, epochs=epochs, verbose=verbose)
    results, y_pred, probabilities = evaluate_model(model, x_test, y_test_onehot, y_test_raw)

    metrics = {"dataset": dataset_info, "history": history, "epochs": epochs,
               "batch_size": BATCH_SIZE, "total_params": int(model.count_params()),
               "summary_lines": summary_lines, **results}
    save_artifacts(model, metrics)
    eval_data = {"x_train": x_train_raw, "y_train": y_train_raw, "x_test": x_test_raw,
                 "y_test": y_test_raw, "y_pred": y_pred, "probabilities": probabilities}
    return model, metrics, eval_data


def run_part_a(output_dir, epochs):
    """Terminal mode: run the CNN workflow, print results and save all graphs."""
    os.makedirs(output_dir, exist_ok=True)
    model, metrics, data = train_and_evaluate(epochs=epochs)
    print("\n=== MODEL SUMMARY ===")
    print("\n".join(metrics["summary_lines"]))
    print("\n=== TEST RESULTS ===")
    print(f"Test loss     : {metrics['test_loss']:.4f}")
    print(f"Test accuracy : {metrics['test_accuracy'] * 100:.2f}%")
    report_text = classification_report(data["y_test"], data["y_pred"], target_names=CLASS_NAMES, digits=4)
    print("\n=== CLASSIFICATION REPORT ===\n" + report_text)

    figures = {
        "01_sample_images": fig_sample_images(data["x_train"], data["y_train"]),
        "02_model_summary": fig_model_summary(metrics["summary_lines"]),
        "03_accuracy_curve": fig_training_curve(metrics["history"], "accuracy"),
        "04_loss_curve": fig_training_curve(metrics["history"], "loss"),
        "05_confusion_matrix": fig_confusion_matrix(metrics["confusion_matrix"]),
        "06_correct_predictions": fig_predictions(data["x_test"], data["y_test"], data["y_pred"],
                                                  data["probabilities"], show_correct=True),
        "07_incorrect_predictions": fig_predictions(data["x_test"], data["y_test"], data["y_pred"],
                                                    data["probabilities"], show_correct=False),
    }
    for name, fig in figures.items():
        fig.savefig(os.path.join(output_dir, name + ".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as file:
        file.write(report_text)

    print("=== RESULT EXPLAINED SIMPLY ===")
    for line in explain_results(metrics):
        print("-", line)
    print(f"\nGraphs saved in: {os.path.abspath(output_dir)}")
    print(f"Trained model cached for the app in: {CACHE_DIR}")


# =============================================================================
# PART B - STREAMLIT APPLICATION (DigitVision AI)
# =============================================================================

# ---- B1. Image validation and preprocessing -----------------------------------
def validate_and_load_image(uploaded_file):
    """Check the upload and return a PIL image. Raises ValueError with a friendly message."""
    if not uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("Unsupported file type. Please upload a PNG, JPG or JPEG image.")
    data = uploaded_file.getvalue()
    if len(data) == 0:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
    try:
        Image.open(io.BytesIO(data)).verify()       # detects fake / corrupted images
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        raise ValueError("This file is not a valid image (it may be corrupted or renamed).")
    if image.format not in ("PNG", "JPEG"):
        raise ValueError("The file content is not a PNG or JPEG image.")
    if min(image.size) < 8:
        raise ValueError("The image is too small. Please upload an image of at least 8x8 pixels.")
    return image


def preprocess_uploaded_image(image, invert_mode="auto"):
    """
    Convert any uploaded picture to the MNIST format: 28x28, white digit on black.
    Steps: grayscale -> invert if needed -> crop to digit -> resize to 20x20 -> centre in 28x28.
    """
    if image.mode != "RGB" and image.mode != "L":   # remove transparency safely
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image.convert("RGBA"))
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    needs_invert = border.mean() > 127 if invert_mode == "auto" else invert_mode == "yes"
    if needs_invert:                                # MNIST digits are light on dark
        gray = 255.0 - gray
    if gray.max() - gray.min() < 30:
        raise ValueError("The image looks blank or has very low contrast. Please upload a clear digit.")
    gray = (gray - gray.min()) / (gray.max() - gray.min())

    rows, cols = np.where(gray > 0.3)
    if len(rows) == 0:
        raise ValueError("No digit could be found in the image.")
    digit = gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    scale = 20.0 / max(digit.shape)                 # MNIST digits fit inside 20x20
    new_size = (max(1, round(digit.shape[1] * scale)), max(1, round(digit.shape[0] * scale)))
    digit = np.asarray(Image.fromarray((digit * 255).astype(np.uint8)).resize(new_size, Image.LANCZOS),
                       dtype=np.float32) / 255.0

    # Place the digit so its centre of mass is at the middle of the 28x28 canvas
    total = digit.sum()
    centre_y = (digit.sum(axis=1) * np.arange(digit.shape[0])).sum() / total
    centre_x = (digit.sum(axis=0) * np.arange(digit.shape[1])).sum() / total
    top = int(np.clip(round(IMG_SIZE / 2 - centre_y), 0, IMG_SIZE - digit.shape[0]))
    left = int(np.clip(round(IMG_SIZE / 2 - centre_x), 0, IMG_SIZE - digit.shape[1]))
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return canvas


def predict_digit(model, image_28x28):
    """Return the 10 class probabilities for one 28x28 image."""
    batch = image_28x28.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    probabilities = model.predict(batch, verbose=0)[0]
    if probabilities.shape != (10,) or not np.isfinite(probabilities).all():
        raise RuntimeError("The model returned an unexpected output.")
    return probabilities


# ---- B2. Cached resources ---------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the CNN model (first launch trains it on MNIST - a few minutes)...")
def get_model_bundle():
    """Load the saved CNN, or train it with the Part A pipeline if it does not exist yet."""
    loaded = load_artifacts()
    if loaded is not None:
        return loaded
    model, metrics, _ = train_and_evaluate()
    return model, metrics


@st.cache_resource(show_spinner="Loading MNIST demo images...")
def get_demo_images():
    (_, _), (x_test, y_test) = load_mnist()
    return x_test, y_test


# ---- B3. Styling and small UI helpers ------------------------------------------------
APP_CSS = """

<style>
.stApp {
    background: linear-gradient(135deg, #0B1020 0%, #111A32 55%, #17152F 100%);
    color: #F1F5F9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
[data-testid="stSidebar"] {
    background: #10182B;
    border-right: 1px solid #263653;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0;
}
.hero {
    background: linear-gradient(115deg, #4338CA 0%, #6D28D9 48%, #0891B2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 35px rgba(79,70,229,.25);
}
.hero h1 {
    color: #FFFFFF !important;
    margin: 0;
    font-size: 2.1rem;
}
.hero p {
    color: #E0E7FF !important;
    margin-top: .5rem;
    font-size: 1.05rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #172238;
    border-color: #30415F !important;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(0,0,0,.15);
}
.metric {
    background: #172238;
    border: 1px solid #30415F;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.metric .value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #67E8F9;
}
.metric .label {
    font-size: .8rem;
    color: #A5B4FC;
    text-transform: uppercase;
}
.digit-circle {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    margin: .5rem auto;
    background: linear-gradient(135deg, #7C3AED, #0891B2);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4.6rem;
    font-weight: 800;
    box-shadow: 0 8px 25px rgba(124,58,237,.35);
}
.center { text-align: center; color: #F1F5F9; }
.conf-bar {
    background: #334155;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin: .5rem 0;
}
.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #8B5CF6, #22D3EE);
}
.placeholder {
    text-align: center;
    color: #CBD5E1;
    padding: 2rem 1rem;
}
.stButton > button {
    background: linear-gradient(90deg, #6D28D9, #0891B2);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}
.stButton > button:hover {
    border: 1px solid #67E8F9;
    color: white !important;
}
.stMarkdown, p, label, h1, h2, h3, h4, h5 {
    color: #E2E8F0;
}
[data-testid="stFileUploader"] {
    background: #1E293B;
    border: 1px dashed #64748B;
    border-radius: 12px;
    padding: 10px;
}
[data-baseweb="select"] > div {
    background: #1E293B;
    border-color: #475569;
}
</style>

"""


_open_cards = []  # cards that are currently open (card_start ... card_end)


def card_start(title):
    """Open a white card (a bordered Streamlit container) with a title."""
    container = st.container(border=True)
    container.__enter__()
    _open_cards.append(container)
    st.markdown(f"##### {title}")


def card_end():
    """Close the most recently opened card."""
    _open_cards.pop().__exit__(None, None, None)


def metric_tile(label, value):
    return f'<div class="metric"><div class="value">{value}</div><div class="label">{label}</div></div>'


def show_figure(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def fig_probabilities(probabilities):
    """Bar chart of the 10 class probabilities; the winner is highlighted."""
    winner = int(np.argmax(probabilities))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(CLASS_NAMES, probabilities * 100,
                  color=[COLORS["primary"] if i == winner else "#CBD5E1" for i in range(10)])
    for bar, value in zip(bars, probabilities * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability (%)")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def enlarge(image_28x28, size=196):
    """Scale the 28x28 array up (without blur) so it is visible on screen."""
    return Image.fromarray((image_28x28 * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


def render_prediction(probabilities, processed_image):
    """Predicted digit, confidence, processed input and probability visualisation."""
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit]) * 100
    card_start("Prediction")
    st.markdown(f'<div class="digit-circle">{digit}</div>'
                f'<div class="center"><b>Predicted digit: {digit}</b><br>'
                f'Confidence: <b>{confidence:.2f}%</b></div>'
                f'<div class="conf-bar"><div class="conf-fill" style="width:{confidence:.1f}%"></div></div>',
                unsafe_allow_html=True)
    if confidence < 60:
        st.warning("The model is not very sure. Try a clearer, thicker, centred digit.")
    card_end()

    card_start("Probability of each digit")
    show_figure(fig_probabilities(probabilities))
    with st.expander("See the exact probabilities"):
        st.table({"Digit": CLASS_NAMES, "Probability (%)": [f"{p * 100:.2f}" for p in probabilities]})
    card_end()

    card_start("What the CNN actually saw (28x28 grayscale)")
    st.image(enlarge(processed_image), width=196)
    card_end()


# ---- B4. Pages ------------------------------------------------------------------------
def page_recognize(model):
    left, right = st.columns(2, gap="large")
    probabilities, processed = None, None
    with left:
        card_start("1. Upload a handwritten digit")
        uploaded = st.file_uploader("PNG, JPG or JPEG (max 5 MB)", type=["png", "jpg", "jpeg"],
                                    key=f"uploader_{st.session_state.uploader_key}")
        style = st.selectbox("Image style", ["Auto-detect", "Dark digit on light background",
                                             "Light digit on dark background"])
        invert_mode = {"Auto-detect": "auto", "Dark digit on light background": "yes",
                       "Light digit on dark background": "no"}[style]
        if uploaded is not None:
            try:
                image = validate_and_load_image(uploaded)
                st.image(image, caption=f"Uploaded: {uploaded.name}", width=220)
                processed = preprocess_uploaded_image(image, invert_mode)
                probabilities = predict_digit(model, processed)
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Prediction failed: {error}")
            if st.button("Clear upload / reset"):
                st.session_state.uploader_key += 1
                st.rerun()
        card_end()
    with right:
        if probabilities is not None:
            render_prediction(probabilities, processed)
        else:
            card_start("Prediction")
            st.markdown('<div class="placeholder">Upload a digit image to see the prediction here.</div>',
                        unsafe_allow_html=True)
            card_end()


def page_demo(model):
    x_test, y_test = get_demo_images()
    left, right = st.columns(2, gap="large")
    with left:
        card_start("Try a sample from the MNIST test set")
        digit = st.selectbox("Choose a digit", CLASS_NAMES)
        candidates = np.where(y_test == int(digit))[0]
        if st.button("Show another sample") or "demo_pick" not in st.session_state:
            st.session_state.demo_pick = int(np.random.default_rng().integers(len(candidates)))
        index = int(candidates[st.session_state.demo_pick % len(candidates)])
        sample = x_test[index].astype("float32") / 255.0
        st.image(enlarge(sample), caption=f"Actual label: {y_test[index]}", width=196)
        card_end()
    with right:
        try:
            probabilities = predict_digit(model, sample)
            render_prediction(probabilities, sample)
            if int(np.argmax(probabilities)) == int(y_test[index]):
                st.success("Correct: the prediction matches the actual label.")
            else:
                st.error("Incorrect: the prediction does not match the actual label.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


def page_performance(metrics):
    tiles = [("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%"),
             ("Test loss", f"{metrics['test_loss']:.4f}"),
             ("Parameters", f"{metrics['total_params']:,}"),
             ("Epochs / batch", f"{metrics['epochs']} / {metrics['batch_size']}")]
    for column, (label, value) in zip(st.columns(4), tiles):
        column.markdown(metric_tile(label, value), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        card_start("Accuracy during training")
        show_figure(fig_training_curve(metrics["history"], "accuracy"))
        card_end()
    with right:
        card_start("Loss during training")
        show_figure(fig_training_curve(metrics["history"], "loss"))
        card_end()
    left, right = st.columns(2, gap="large")
    with left:
        card_start("Confusion matrix")
        show_figure(fig_confusion_matrix(metrics["confusion_matrix"]))
        card_end()
    with right:
        card_start("Precision, recall and F1-score per digit")
        report = metrics["classification_report"]
        st.dataframe([{"Digit": d, "Precision": round(report[d]["precision"], 4),
                       "Recall": round(report[d]["recall"], 4), "F1-score": round(report[d]["f1-score"], 4),
                       "Images": int(report[d]["support"])} for d in CLASS_NAMES],
                     hide_index=True)
        card_end()
    card_start("What do these results mean?")
    for line in explain_results(metrics):
        st.markdown("- " + line)
    with st.expander("CNN architecture (model summary)"):
        st.code("\n".join(metrics["summary_lines"]), language="text")
    card_end()


def page_instructions():
    card_start("How to use DigitVision AI")
    st.markdown("""
1. Open **Recognize** from the sidebar.
2. Upload a PNG, JPG or JPEG picture that contains **one** handwritten digit (0-9).
3. Leave *Image style* on **Auto-detect**, or choose it yourself if the result looks wrong.
4. The app converts the picture to 28x28 grayscale (the MNIST format) and the CNN predicts the digit.
5. Read the **predicted digit**, the **confidence**, and the **probability of all 10 digits**.
6. Use **Clear upload / reset** to start again. No image? Try **Demo Samples**.

**Tips for the best result:** write one digit with a thick dark pen on plain paper, keep the digit
large and centred, and avoid shadows or lines in the background.
""")
    card_end()
    card_start("How it works")
    st.markdown("""
Upload -> validate file -> grayscale -> invert if background is light -> crop and centre in 28x28
-> CNN (Conv + Pool + Dense) -> softmax probabilities -> highest probability = predicted digit.
""")
    card_end()


def render_sidebar(metrics):
    with st.sidebar:
        st.markdown(f"## 🔢 {APP_NAME}")
        st.caption("Handwritten Digit Recognition")
        page = st.radio("Navigate", ["Recognize", "Demo Samples", "Model Performance", "Instructions"])
        st.divider()
        st.markdown("**Model status**")
        st.success("CNN ready")
        st.write(f"Test accuracy: **{metrics['test_accuracy'] * 100:.2f}%**")
        st.write(f"Trained on: **{metrics['dataset']['train_samples']:,}** MNIST images")
        st.divider()
        st.caption("Built by Vedant Vijay | Minor Project 7")
    return page


def run_app():
    st.set_page_config(page_title=f"{APP_NAME} - Handwritten Digit Recognition", page_icon="🔢", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    try:
        model, metrics = get_model_bundle()
    except Exception as error:
        st.error(f"The CNN model could not be prepared: {error}")
        st.stop()

    page = render_sidebar(metrics)
    st.markdown(f'<div class="hero"><h1>🔢 {APP_NAME} - Handwritten Digit Recognition</h1>'
                '<p>Upload a handwritten digit and a Convolutional Neural Network trained on MNIST '
                'will recognise it instantly.</p></div>', unsafe_allow_html=True)
    if page == "Recognize":
        page_recognize(model)
    elif page == "Demo Samples":
        page_demo(model)
    elif page == "Model Performance":
        page_performance(metrics)
    else:
        page_instructions()


# =============================================================================
# ENTRY POINT: Streamlit runs the app, plain Python runs Part A
# =============================================================================
if get_script_run_ctx(suppress_warning=True) is not None:
    run_app()
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DigitVision AI - CNN workflow (Part A)")
    parser.add_argument("--train", action="store_true", help="run the CNN workflow (Part A)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of training epochs")
    parser.add_argument("--out", default="part_a_outputs", help="folder for the saved graphs")
    arguments = parser.parse_args()
    if arguments.train:
        run_part_a(arguments.out, arguments.epochs)
    else:
        print("Use --train to run the CNN workflow, or run the app with:\n"
              "    streamlit run VedantVijay_CNN_Streamlit_Project.py")
