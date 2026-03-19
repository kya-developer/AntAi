# AntAi

A simple TensorFlow-based ant behavior prediction project. The model uses environmental measurements and species names to predict ant behavior categories.

## 📁 Repository structure

- `main.py` - CLI for interactive prediction sessions.
- `train.py` - loads `trainingData.json`, builds and trains the model, then saves `ant_model.keras` and generates `behavior_vocab.json`.
- `model.py` - defines the Keras model architecture and lookup layers for species/behavior.
- `predict.py` - loads the saved model and vocab, exposes `Predict()` for inference.
- `trainingData.json` - labeled dataset used for training.
- `behavior_vocab.json` - saved behavior labels mapping index -> class.
- `ant_model.keras` - serialized trained model artifact (created by `train.py`).

## 🚀 Setup

1. Create a Python virtual environment (recommended):
   - `python -m venv .venv`
   - `source .venv/Scripts/activate` (Windows)

2. Install required packages:
   - `pip install tensorflow numpy pyfiglet`

3. Confirm your `trainingData.json` includes entries like:
   ```json
   {
     "tempurature": 22.0,
     "humidity": 0.50,
     "light": true,
     "time": 13,
     "species": "species_a",
     "behavior": "forage"
   }
   ```

## 🏋️ Training

Run:
```bash
python train.py
```

This performs the following steps:
- loads data from `trainingData.json`
- builds a dataset and string lookup vectors
- creates model from `model.py` using lookup layers
- trains for 75 epochs (configurable in `train.py`)
- saves model to `ant_model.keras` and behavior vocabulary to `behavior_vocab.json`

## 🧪 Prediction (programmatic)

Use `predict.py` directly from other code:
```python
from predict import Predict
label = Predict(tempurature=25.0, humidity=0.85, light=True, time=10, species='species_a')
print('Predicted behavior:', label)
```

## 🖥️ CLI usage (`main.py`)

Run:
```bash
python main.py
```

Commands:
- `help` - show usage.
- `predict` - enter an interactive prompt for values and print prediction.
- `exit` - stop.

Example flow:
```
>>> predict
please answer the following questions:
tempurature: 24.5
humidity: 0.60
light (True/False): True
time (0-23): 14
species: species_a
Predicted behavior: forage
```

## ⚠️ Notes

- `main.py` parses `light` as boolean (`True` / `False`).
- `tempurature` is the spelling used in code and must match in JSON keys.
- `behavior_vocab.json` must exist alongside `ant_model.keras` for prediction.

## 🛠️ Troubleshooting

- If `tf.keras.models.load_model` fails, confirm model file exists and TensorFlow version is compatible.
- If unknown species/behavior values are passed, the StringLookup layer may produce all-zero vectors and lower accuracy.

## ✅ Improvements (optional)

- Add data validation and sanitize CLI entries.
- Save a `requirements.txt` with pinned package versions.
- Add unit tests for `Predict` and model input conversion.
