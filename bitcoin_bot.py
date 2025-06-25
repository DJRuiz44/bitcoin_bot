import os
import time
from datetime import datetime, timedelta
import threading
import platform
import smtplib
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

EMAIL_ADDRESS = os.getenv("BOT_EMAIL")
EMAIL_PASSWORD = os.getenv("BOT_EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("BOT_EMAIL_RECIPIENT", EMAIL_ADDRESS)

SYMBOL = "BTC/USD"
POSITION_SYMBOL = "BTCUSD"
MODEL_PATH = "ai_model.pkl"

FEATURES = ["returns", "sma_3", "sma_6", "sma_15", "stddev", "vol_chg"]

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")


# === Utility Functions ===

def submit_order(amount: float, side: str, crypto: bool = True) -> None:
    if crypto:
        api.submit_order(
            symbol=SYMBOL,
            notional=round(amount, 2),
            side=side,
            type="market",
            time_in_force="gtc",
        )
    else:
        api.submit_order(
            symbol=SYMBOL,
            qty=amount,
            side=side,
            type="market",
            time_in_force="gtc",
        )


def play_sound() -> None:
    if platform.system() == "Windows":
        os.system("start trade_alert.wav")
    elif platform.system() == "Darwin":
        os.system("afplay trade_alert.wav")
    else:
        os.system("aplay trade_alert.wav")


def send_email(subject: str, body: str) -> None:
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECIPIENT
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Email failed: {exc}")


def get_latest_data() -> pd.DataFrame:
    try:
        bars = api.get_crypto_bars(
            SYMBOL,
            tradeapi.TimeFrame.Minute,
            limit=30,
        ).df
        df = bars.reset_index()
        df = df[df["symbol"] == SYMBOL].copy()
        df["returns"] = df["close"].pct_change()
        df["sma_3"] = df["close"].rolling(window=3).mean()
        df["sma_6"] = df["close"].rolling(window=6).mean()
        df["sma_15"] = df["close"].rolling(window=15).mean()
        df["stddev"] = df["close"].rolling(window=15).std()
        df["vol_chg"] = df["volume"].pct_change()
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        return df
    except Exception as exc:
        print(f"get_latest_data failed: {exc}")
        return pd.DataFrame()


def retrain_model() -> None:
    print("Retraining model...")
    bars = api.get_crypto_bars(
        SYMBOL,
        tradeapi.TimeFrame.Minute,
        start=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
    ).df
    df = bars.reset_index()
    df = df[df["symbol"] == SYMBOL].copy()
    df["returns"] = df["close"].pct_change()
    df["sma_3"] = df["close"].rolling(window=3).mean()
    df["sma_6"] = df["close"].rolling(window=6).mean()
    df["sma_15"] = df["close"].rolling(window=15).mean()
    df["stddev"] = df["close"].rolling(window=15).std()
    df["vol_chg"] = df["volume"].pct_change()
    df["future_return"] = df["close"].shift(-10) / df["close"] - 1
    df["signal"] = df["future_return"].apply(
        lambda x: "buy" if x > 0.005 else ("sell" if x < -0.005 else "hold")
    )
    df = df.dropna()

    X = df[FEATURES]
    y = df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


def run_bot() -> None:
    if not os.path.exists(MODEL_PATH):
        retrain_model()
    model = joblib.load(MODEL_PATH)
    trades_since_retrain = 0
    last_retrain = datetime.now().date()

    while True:
        if trades_since_retrain >= 4 or datetime.now().date() != last_retrain:
            retrain_model()
            model = joblib.load(MODEL_PATH)
            trades_since_retrain = 0
            last_retrain = datetime.now().date()

        df = get_latest_data()
        if df.empty:
            time.sleep(60)
            continue

        latest = df.iloc[-1]
        X = pd.DataFrame([latest[FEATURES].values], columns=FEATURES)
        signal = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        try:
            pos = api.get_position(POSITION_SYMBOL)
            qty = float(pos.qty)
            avg_entry = float(pos.avg_entry_price)
        except Exception:
            qty = 0.0
            avg_entry = 0.0

        account = api.get_account()
        cash = float(account.cash)
        price = latest["close"]

        print(
            f"{datetime.now()}: signal={signal} buy_prob={probs[0]:.2%} "
            f"sell_prob={probs[2]:.2%} cash=${cash:.2f} qty={qty}"
        )

        if qty == 0 and signal == "buy" and probs[0] > 0.6 and cash >= 1.0:
            amount = cash * 0.99
            submit_order(amount, "buy", crypto=True)
            trades_since_retrain += 1
            if os.path.exists("trade_alert.wav"):
                threading.Thread(target=play_sound).start()
            send_email("Initial BUY", f"Bought BTC with ${amount:.2f}")
            continue

        if qty > 0:
            if signal == "sell" and probs[2] > 0.6 and price > avg_entry:
                submit_order(qty, "sell", crypto=True)
                trades_since_retrain += 1
                if os.path.exists("trade_alert.wav"):
                    threading.Thread(target=play_sound).start()
                send_email("Sold BTC", f"Sold {qty} BTC at ${price:.2f} expecting lower re-entry")
                continue

            if price < avg_entry * 0.5:
                submit_order(qty, "sell", crypto=True)
                if os.path.exists("trade_alert.wav"):
                    threading.Thread(target=play_sound).start()
                send_email("Stop Loss", f"Sold {qty} BTC at ${price:.2f} to prevent deeper loss")
                continue

        time.sleep(60)


if __name__ == "__main__":
    run_bot()
