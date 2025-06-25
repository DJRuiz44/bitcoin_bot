# bitcoin_bot

A simple automated trading bot for Bitcoin built on the Alpaca API. It trains a RandomForest model to predict short‑term movements and aggressively buys BTC when confidence is high. The bot only sells when the model expects a dip or when a stop‑loss is triggered.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the following environment variables with your credentials:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - `ALPACA_BASE_URL` (optional, defaults to `https://api.alpaca.markets`)
   - `BOT_EMAIL` and `BOT_EMAIL_PASSWORD` for notifications (optional)
   - `BOT_EMAIL_RECIPIENT` (optional)
3. Run the bot:
   ```bash
   python bitcoin_bot.py
   ```

The model automatically retrains every few trades or once per day.

## Troubleshooting

A common issue when starting the bot is receiving a `403` error from the
Alpaca API. This usually means the API keys are missing or invalid. Ensure that
`ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are correctly set in your environment
before running the bot.
