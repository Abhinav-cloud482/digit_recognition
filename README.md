# digit_recognition
Developed a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. Applied image preprocessing, trained the model with the Adam optimizer, and evaluated its performance for accurate digit recognition.


# MNIST Digit Classifier (CNN)

A convolutional neural network for classifying handwritten digits (0–9) from the MNIST dataset, built with TensorFlow/Keras. The pipeline includes data augmentation, training callbacks, and model checkpointing, along with a full evaluation suite covering confusion matrices, classification reports, and prediction visualizations.


## Features

- **CNN architecture** with Conv2D, BatchNormalization, MaxPooling, and Dropout layers
- **Data augmentation** (rotation, shift, zoom) for better generalization
- **Train/validation/test split** — the test set is only touched for final evaluation
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best model auto-saved)
- **Model persistence** — trained model is saved and reloaded automatically on future runs
- **Evaluation outputs**:
  - Training accuracy/loss curves
  - Confusion matrix heatmap
  - Classification report (precision/recall/F1 per digit)
  - Grid of sample predictions (correct vs. incorrect)
  - Grid of misclassified digits for error analysis
  - Single-image prediction with a probability bar chart

## Project structure

```
.
├── mnist_cnn.py         # main script
├── requirements.txt     # dependencies
├── README.md
└── outputs/              # generated after running — models, plots, reports
```


## Usage

Train (or load an existing model) and run full evaluation:

```bash
python mnist_cnn.py
```

Force a fresh training run even if a saved model exists:

```bash
python mnist_cnn.py --force-train
```

Customize training:

```bash
python mnist_cnn.py --epochs 20 --batch-size 128
```

The first run downloads MNIST automatically (requires internet access) and caches it locally via Keras.

## Outputs

After running, check the `outputs/` folder for:

| File | Description |
|---|---|
| `mnist_cnn.keras` | Trained model weights |
| `training_history.png` | Accuracy/loss curves over epochs |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `classification_report.txt` | Precision/recall/F1 per digit |
| `sample_predictions.png` | Grid of test images with predictions |
| `misclassified.png` | Grid of the model's mistakes |
| `single_prediction_example.png` | One prediction with probability bar chart |

## Example results

On a typical run, the model reaches **~99% test accuracy** after early stopping kicks in (usually within 10–15 epochs).

## License

MIT — feel free to use and modify.
