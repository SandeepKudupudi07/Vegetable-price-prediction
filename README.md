# Vegetable Price Prediction

A BTech third-year machine learning project that estimates vegetable prices from weather, demand, crop conditions, and seasonal factors. A Flask web interface combines predictions from **LSTM** and **Dense neural network** models and displays the estimated price in **INR per kilogram**.

## Features

- Predictions for 12 vegetables: Potato, Tomato, Peas, Pumpkin, Cucumber, Ivy Gourd, Radish, Bitter Gourd, Onion, Garlic, Cabbage, and Cauliflower.
- A form for temperature, rainfall, humidity, demand, freshness, inflation, yield per hectare, season, disaster conditions, and holiday/festival status.
- An equal-weight ensemble of two pretrained models, with the original feature and target scalers included.
- A result page displaying the predicted price and a link to make another prediction.
- The recovered training notebook and matching 61,368-row historical dataset.

## Run locally

Use **Python 3.11**. The supplied models were saved with **Keras 3.4.1**; use the pinned dependencies in `requirements.txt`.

### Windows PowerShell

```powershell
git clone https://github.com/SandeepKudupudi07/Vegetable-price-prediction.git
cd Vegetable-price-prediction
.\run.ps1
```

The launcher creates `.venv`, installs dependencies on its first run, and starts the app at **http://127.0.0.1:5000**. Stop it with `Ctrl+C`.

If the first dependency installation is interrupted, complete it with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1
```

### Manual setup

From the repository root, create and activate a Python 3.11 virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python -m flask --app app --no-debug run --host 127.0.0.1 --port 5000
```

Keep the repository root as the working directory so that the app can find `models/`. The pages use Tailwind CSS from a CDN, so their styling requires an internet connection.

## Example

Choose **Tomato** and enter the following values:

| Input | Value |
| --- | --- |
| Temperature | 28 |
| Rainfall | 100 |
| Humidity | 65 |
| Demand | 75 |
| Freshness | 4 |
| Inflation | 5 |
| Yield per hectare | 200 |
| Season | Summer |
| Disaster | Normal |
| Holiday/festival | No |

With the included models and verified runtime, this example produces approximately **INR 75.76/kg**. This is a demonstration prediction, not a live market quote.

## How prediction works

1. Build the feature row in the order expected by the saved scaler, including categorical dummy columns.
2. Scale the resulting **23 features** with `feature_scaler.pkl`.
3. Send the scaled row to the Dense network and reshape it to `(1, 1, 23)` for the LSTM network.
4. Convert both outputs back to price units using `target_scaler.pkl`.
5. Average the two prices and display the result.

The included application uses **LSTM + Dense**. Earlier experimental versions used ARIMA, but ARIMA is not part of this application's inference pipeline. The LSTM receives a single timestep of tabular features; the web app does not implement a future-date forecasting horizon.

## Project structure

```text
Vegetable-price-prediction/
├── app.py                         # Flask routes, preprocessing, and prediction
├── requirements.txt               # Verified inference dependencies
├── run.ps1                        # Windows launcher
├── models/
│   ├── lstm_vegetable_price_model.h5
│   ├── dense_vegetable_price_model.h5
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
├── templates/
│   ├── index.html                 # Prediction form
│   └── result.html                # Price result
├── data/
│   └── main.csv                   # Matching historical training dataset
└── training/
    └── original-training.ipynb     # Recovered training and experiment notebook
```

## Verification

The recovered project was checked on **27 November 2026** using Python 3.11.0, TensorFlow 2.17.0, Keras 3.4.1, scikit-learn 1.5.2, and Flask 2.3.3.

- Both pretrained models and their scalers load successfully.
- **21/21 ordinary prediction checks pass**, covering all 12 vegetables and the displayed season, disaster, and holiday choices.
- A local HTTP check passes through the home page, prediction redirect, and rendered result page.
- The recovered dataset's reconstructed training feature means match the saved feature scaler exactly.

Reconstructing the notebook's 80/20 random split with `random_state=42` gives these ensemble results on 12,274 holdout rows:

| Metric | Result |
| --- | --- |
| Mean absolute error | INR 5.83/kg |
| R-squared | 0.8206 |

These results describe a historical random split. They are not a chronological backtest or evidence of current market accuracy. The saved models were tested without retraining.

## Training notebook

`training/original-training.ipynb` preserves the recovered experiments. It expects a file named `main.csv` in its working directory; update that path to `data/main.csv` when running from the repository root. Some cells are environment-specific and the notebook also contains Flask experimentation code.

The notebook has not been rerun end to end during recovery. Training and plotting may require additional packages, including Matplotlib. `requirements.txt` covers the verified web application inference environment.

## Known limitations

- This is an academic prototype based on historical data, with no live market or weather feed.
- Server-side validation needs improvement: unknown vegetable values are accepted, while missing or non-numeric inputs can return JSON errors with HTTP status 200.
- The form offers `Cyclone`, but the trained feature encoder has no separate Cyclone column. It currently receives the same disaster-category encoding as the baseline condition.
- The original `python app.py` entry point enables Flask debug mode. Use the supplied launcher or the documented `--no-debug` command for the local demo.

## Author

[Sandeep Kudupudi](https://github.com/SandeepKudupudi07)
