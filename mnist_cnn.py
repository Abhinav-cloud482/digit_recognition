import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "mnist_cnn.keras")


def load_data(val_fraction=0.1, seed=42):
    """Load MNIST, normalize, reshape, and carve out a validation split
    from the training data (so the test set is only ever used for final
    evaluation, never for tuning)."""
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    rng = np.random.default_rng(seed)
    n_val = int(len(train_images) * val_fraction)
    idx = rng.permutation(len(train_images))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return (
        train_images[train_idx], train_labels[train_idx],
        train_images[val_idx], train_labels[val_idx],
        test_images, test_labels,
    )


def build_model():
    """A deeper CNN than the original: BatchNorm stabilizes training,
    Dropout fights overfitting, and an extra dense layer adds capacity."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]


def train(model, train_images, train_labels, val_images, val_labels, epochs, batch_size=64):
    # Light augmentation: MNIST digits are already centered, so keep
    # transformations small to avoid distorting digit identity.
    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
    )
    datagen.fit(train_images)

    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=get_callbacks(),
    )
    return history


def plot_training_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves -> {path}")


def evaluate_and_report(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")

    pred_probs = model.predict(test_images, verbose=0)
    preds = np.argmax(pred_probs, axis=1)

    report = classification_report(test_labels, preds, digits=3)
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved classification report -> {report_path}")

    return preds, pred_probs, test_acc


def plot_confusion_matrix(test_labels, preds, path):
    cm = confusion_matrix(test_labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def plot_prediction_grid(images, true_labels, preds, path, n=16, title="Sample Predictions"):
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n and i < len(images):
            ax.imshow(images[i].reshape(28, 28), cmap="gray")
            correct = true_labels[i] == preds[i]
            color = "green" if correct else "red"
            ax.set_title(f"T:{true_labels[i]} P:{preds[i]}", color=color, fontsize=10)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved prediction grid -> {path}")


def plot_misclassified(test_images, test_labels, preds, path, n=16):
    wrong_idx = np.where(preds != test_labels)[0]
    print(f"Misclassified: {len(wrong_idx)} / {len(test_labels)} "
          f"({len(wrong_idx) / len(test_labels) * 100:.2f}%)")
    if len(wrong_idx) == 0:
        print("No misclassified examples to plot.")
        return
    chosen = wrong_idx[:n]
    plot_prediction_grid(
        test_images[chosen], test_labels[chosen], preds[chosen],
        path, n=len(chosen), title="Misclassified Examples",
    )


def predict_single(model, images, labels, index, path=None):
    """Predict one image and optionally save an image + probability bar chart."""
    probs = model.predict(images[index:index + 1], verbose=0)[0]
    pred = int(np.argmax(probs))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].imshow(images[index].reshape(28, 28), cmap="gray")
    axes[0].set_title(f"True: {labels[index]}  Pred: {pred}")
    axes[0].axis("off")

    axes[1].bar(range(10), probs)
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Model confidence")

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        print(f"Saved single-prediction figure -> {path}")
    plt.close(fig)
    return pred, probs


def main():
    parser = argparse.ArgumentParser(description="Enhanced MNIST CNN")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-train", action="store_true",
                         help="Retrain even if a saved model exists")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data()
    print(f"Train: {len(train_images)}  Val: {len(val_images)}  Test: {len(test_images)}")

    if os.path.exists(MODEL_PATH) and not args.force_train:
        print(f"Found existing model at {MODEL_PATH}, loading it (use --force-train to retrain).")
        model = tf.keras.models.load_model(MODEL_PATH)
        history = None
    else:
        model = build_model()
        model.summary()
        history = train(model, train_images, train_labels, val_images, val_labels,
                         epochs=args.epochs, batch_size=args.batch_size)
        # ModelCheckpoint already saved the best weights to MODEL_PATH.
        if history is not None:
            plot_training_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))

    preds, pred_probs, test_acc = evaluate_and_report(model, test_images, test_labels)

    plot_confusion_matrix(test_labels, preds, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_prediction_grid(test_images, test_labels, preds,
                          os.path.join(OUTPUT_DIR, "sample_predictions.png"))
    plot_misclassified(test_images, test_labels, preds,
                        os.path.join(OUTPUT_DIR, "misclassified.png"))
    predict_single(model, test_images, test_labels, index=0,
                    path=os.path.join(OUTPUT_DIR, "single_prediction_example.png"))

    print("\nDone. All outputs are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
