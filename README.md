# NYC Temperature Forecasting Engine

An XGBoost-based nowcasting system for Central Park (KNYC) temperatures, trained on 140,000+ rows of ASOS observation data from the Iowa Environmental Mesonet. Achieves **3.4°F MAE** on held-out test data and runs as a live Discord bot that posts hourly forecasts.

---

## Models

Three separate models are trained in `KNYC_Nowcaster.ipynb`:

| Model | Target | Test MAE |
|---|---|---|
| Daily High | Today's peak temperature | 3.4°F |
| t+3h | Temperature 3 hours from now | — |
| t+6h | Temperature 6 hours from now | — |

All three models use the same feature set and are saved as `.pkl` files loaded directly by the Discord bot.

---

## Features

Features are engineered from hourly KNYC ASOS observations and fall into a few categories:

- **Raw obs** — temperature, dewpoint, relative humidity, MSLP, wind speed
- **Derived meteorological** — dew depression, feel gap, td ratio
- **Pressure tendencies** — 1h, 3h, and 6h MSLP change
- **Temperature tendencies** — 1h, 3h, and 6h temp change
- **Lag variables** — temperature and dewpoint at 1h, 2h, 3h, 6h, 12h, 24h lookback
- **Rolling statistics** — 3h, 6h, 24h rolling mean; 3h rolling std
- **Temporal encoding** — sine/cosine transforms of hour and month to capture diurnal and seasonal cycles

---

## Project Structure

```
NYC-Temp-Forecaster/
├── KNYC_Nowcaster.ipynb        # Data loading, feature engineering, training, evaluation
├── knyc_discord_bot.py         # Live Discord bot — fetches obs and runs inference hourly
├── models/
│   ├── knyc_model_daily_high.pkl
│   ├── knyc_model_t3h.pkl
│   └── knyc_model_t6h.pkl
└── .env                        # Discord credentials (never committed)
```

---

## Setup

**Install dependencies:**
```bash
pip install discord.py python-dotenv requests pandas numpy scikit-learn joblib xgboost
```

**Create a `.env` file:**
```
DISCORD_TOKEN=your_bot_token_here
GUILD_ID=your_server_id_here
```

**Run the bot:**
```bash
python knyc_discord_bot.py
```

The bot posts an immediate nowcast on startup to confirm it's working, then aligns to `:59 UTC` and posts every hour — 8 minutes after KNYC's scheduled `:51` METAR observation drops.

---

## How the Bot Works

Each hour the bot:
1. Fetches the last 30 hours of KNYC `:51` ASOS observations from the Iowa Mesonet API
2. Scrapes the IEM observation history page as a fallback for the latest reading (updates faster than the API)
3. Engineers the same feature set used during training
4. Runs inference on all three models
5. Computes a **reassessed high** — the floor-locked max of the model prediction and the observed high so far today
6. Posts a color-coded Discord embed with current conditions and all three forecasts

---

## Data Source

Historical training data and live observations are pulled from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu) ASOS archive for station `KNYC` (Central Park, New York). The training dataset spans 2010–2023 with a time-aware train/val/test split to prevent data leakage.

To re-pull training data, run the data fetch cell in `KNYC_Nowcaster.ipynb` — no large CSV files are committed to this repo.
