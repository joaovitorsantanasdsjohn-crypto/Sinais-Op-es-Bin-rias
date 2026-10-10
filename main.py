
import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import websocket
import pandas as pd

from flask import Flask, jsonify
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

# ==================================================
# CONFIGURAÇÕES
# ==================================================

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOLS = ["frxEURUSD", "frxGBPUSD"]
GRANULARITY = 900
CANDLE_COUNT = 500
THRESHOLD = 0.70
REPORT_INTERVAL = 3600

BRASILIA_TZ = ZoneInfo("America/Sao_Paulo")

FEATURES = [
    "return_1",
    "return_2",
    "return_4",
    "range_pct",
    "body_pct",
    "sma_diff",
    "volatility",
    "rsi",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = Flask(__name__)

monitor_thread = None
monitor_lock = threading.Lock()
stop_event = threading.Event()
last_report = {}


# ==================================================
# HORÁRIO DE BRASÍLIA
# ==================================================

def brasilia_now():
    return datetime.now(timezone.utc).astimezone(BRASILIA_TZ)


def format_brasilia(epoch=None):
    if epoch is None:
        dt = brasilia_now()
    else:
        dt = datetime.fromtimestamp(
            float(epoch), timezone.utc
        ).astimezone(BRASILIA_TZ)

    return dt.strftime("%d/%m/%Y %H:%M:%S")


# ==================================================
# TELEGRAM
# ==================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Configure as variáveis do Telegram.")
        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=15,
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        logging.exception("Erro ao enviar mensagem ao Telegram.")
        return False


# ==================================================
# BUSCA DE CANDLES NA DERIV
# ==================================================

def get_candles(symbol):
    if not DERIV_APP_ID:
        raise RuntimeError("DERIV_APP_ID não configurado.")

    url = (
        "wss://ws.derivws.com/websockets/v3"
        f"?app_id={DERIV_APP_ID}"
    )

    ws = websocket.create_connection(url, timeout=20)

    try:
        ws.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": CANDLE_COUNT,
            "end": "latest",
            "granularity": GRANULARITY,
            "style": "candles",
        }))

        deadline = time.time() + 20

        while time.time() < deadline:
            response = json.loads(ws.recv())

            if "error" in response:
                raise RuntimeError(
                    response["error"].get("message", "Erro da Deriv")
                )

            if "candles" not in response:
                continue

            df = pd.DataFrame(response["candles"])

            for column in ["epoch", "open", "high", "low", "close"]:
                df[column] = pd.to_numeric(
                    df[column], errors="coerce"
                )

            df = (
                df.dropna()
                .sort_values("epoch")
                .drop_duplicates("epoch")
                .reset_index(drop=True)
            )

            # Ignora o candle que ainda pode estar em formação.
            cutoff = int(time.time()) - GRANULARITY
            df = df[df["epoch"] <= cutoff].reset_index(drop=True)

            if len(df) < 100:
                raise ValueError("Candles insuficientes.")

            return df

        raise TimeoutError("Tempo esgotado ao consultar a Deriv.")

    finally:
        ws.close()


# ==================================================
# INDICADORES TÉCNICOS
# ==================================================

def calculate_features(raw):
    df = raw.copy()

    df["return_1"] = df["close"].pct_change()
    df["return_2"] = df["close"].pct_change(2)
    df["return_4"] = df["close"].pct_change(4)

    df["range_pct"] = (
        (df["high"] - df["low"]) / df["close"]
    )

    df["body_pct"] = (
        (df["close"] - df["open"]) / df["open"]
    )

    sma_5 = df["close"].rolling(5).mean()
    sma_10 = df["close"].rolling(10).mean()

    df["sma_diff"] = (sma_5 - sma_10) / df["close"]
    df["volatility"] = df["return_1"].rolling(10).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    return df.replace([float("inf"), float("-inf")], float("nan"))


# ==================================================
# CONJUNTO HISTÓRICO PARA AVALIAÇÃO
# ==================================================

def create_dataset(raw):
    df = calculate_features(raw)

    # Resultado conhecido do candle seguinte.
    df["target"] = (
        df["close"].shift(-1) > df["close"]
    ).astype(int)

    # Remove a última linha: o resultado seguinte ainda é desconhecido.
    df = df.iloc[:-1].copy()
    df = df.dropna(subset=FEATURES + ["target"])

    return df.reset_index(drop=True)


# ==================================================
# MACHINE LEARNING: TESTE HISTÓRICO
# ==================================================

def evaluate_model(raw):
    df = create_dataset(raw)

    if len(df) < 100:
        raise ValueError("Dados insuficientes para avaliação.")

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    if train["target"].nunique() < 2:
        raise ValueError("O treino contém somente uma classe.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    model.fit(train[FEATURES], train["target"])

    predictions = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]

    accuracy = accuracy_score(test["target"], predictions)
    brier = brier_score_loss(test["target"], probabilities)

    # Contagem de ocorrências históricas no conjunto de teste.
    high_count = int((probabilities >= THRESHOLD).sum())
    low_count = int((probabilities <= 1 - THRESHOLD).sum())

    return {
        "accuracy": float(accuracy),
        "brier": float(brier),
        "samples": int(len(test)),
        "high_count": high_count,
        "low_count": low_count,
    }


# ==================================================
# NOTIFICAÇÃO DE AVALIAÇÃO HISTÓRICA
# ==================================================

def monitor_symbol(symbol):
    candles = get_candles(symbol)
    metrics = evaluate_model(candles)

    now = brasilia_now()
    key = f"{symbol}_evaluation"

    if time.time() - last_report.get(key, 0) < REPORT_INTERVAL:
        return metrics

    # Mensagem de avaliação histórica; não é um sinal ao vivo.
    message = (
        f"📊 {symbol} — Monitoramento\n"
        f"🕒 Horário: {now.strftime('%H:%M:%S')} (Brasília)\n"
        f"📅 Data: {now.strftime('%d/%m/%Y')}\n"
        f"📈 Acurácia histórica: {metrics['accuracy']:.2%}\n"
        f"📐 Brier score: {metrics['brier']:.4f}\n"
        f"🔎 Casos históricos com estimativa de alta ≥ 70%: "
        f"{metrics['high_count']}\n"
        f"🔎 Casos históricos com estimativa de baixa ≥ 70%: "
        f"{metrics['low_count']}\n\n"
        "Avaliação histórica; não representa previsão futura."
    )

    if send_telegram(message):
        last_report[key] = time.time()

    return metrics


# ==================================================
# LOOP DE MONITORAMENTO
# ==================================================

def monitor_loop():
    logging.info("Monitoramento iniciado.")

    while not stop_event.is_set():
        for symbol in SYMBOLS:
            if stop_event.is_set():
                break

            try:
                monitor_symbol(symbol)
            except Exception:
                logging.exception("Erro no ativo %s", symbol)

        stop_event.wait(GRANULARITY)


def start_monitor_once():
    global monitor_thread

    with monitor_lock:
        if monitor_thread is None or not monitor_thread.is_alive():
            monitor_thread = threading.Thread(
                target=monitor_loop,
                name="deriv-monitor",
                daemon=True,
            )
            monitor_thread.start()


# ==================================================
# ROTAS DO RENDER
# ==================================================

@app.route("/")
def home():
    return "Bot de monitoramento Deriv ativo.", 200


@app.route("/health")
def health():
    running = (
        monitor_thread is not None
        and monitor_thread.is_alive()
    )

    return jsonify({
        "status": "ok",
        "horario_brasilia": format_brasilia(),
        "monitor_running": running,
    }), 200


# Inicializa o monitor ao importar o módulo pelo Gunicorn.
start_monitor_once()
