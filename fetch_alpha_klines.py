import sqlite3
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt

DB_FILE = "binance_data.db"
KLINES_URL = "https://www.binance.com/bapi/defi/v1/public/alpha-trade/klines"

INTERVAL = "15m"
LIMIT = 1000

def interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60 * 1000
    if unit == "h":
        return amount * 60 * 60 * 1000
    if unit == "d":
        return amount * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")

def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except:
        return None

def fetch_all_klines_incremental(conn, symbol, interval=INTERVAL, limit=LIMIT):
    alpha_token_id = symbol[:-4]  # e.g. ALPHA_391USDT -> ALPHA_391
    quote_asset = symbol[-4:]     # e.g. USDT
    last_ts = get_last_timestamp(conn, alpha_token_id, quote_asset, interval)

    if last_ts:
        print(f"🔁 Continuing from {datetime.utcfromtimestamp(last_ts/1000)}")
        start_time = last_ts + interval_to_milliseconds(interval)
    else:
        # Start from Jan 1, 2025 (UTC)
        start_time = int(datetime(2025, 1, 1).timestamp() * 1000)
        print(f"No previous data — starting from {datetime.utcfromtimestamp(start_time/1000)}")

    total = 0
    while True:
        data = fetch_klines(symbol, interval, limit=limit, start_time=start_time)
        if not data:
            print("No more new data.")
            break

        store_klines(conn, alpha_token_id, quote_asset, interval, data)
        total += len(data)
        last_open_time = int(data[-1][0])
        update_fetch_state(conn, alpha_token_id, quote_asset, interval, last_open_time)

        print(f"Inserted {len(data)} rows. Latest open_time = {last_open_time}")
        if len(data) < limit:
            print("Reached latest available data.")
            break

        start_time = last_open_time + interval_to_milliseconds(interval)
        time.sleep(0.5)

    print(f"Done. Total inserted/fetched: {total}")

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT, start_time=None):
    print(f"Fetching klines for {symbol} ...")
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = int(start_time)
    r = requests.get(KLINES_URL, params=params)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text}")
        return []
    obj = r.json()
    if not obj.get("success", False):
        print(f"API error: {obj}")
        return []
    return obj.get("data", [])

# --- DB Operations -----------------------------------------------------------

def get_last_timestamp(conn, alpha_token_id, quote_asset, interval):
    cur = conn.cursor()
    cur.execute("""
        SELECT last_open_time FROM klines_fetch_state
        WHERE alpha_token_id=? AND quote_asset=? AND interval=?;
    """, (alpha_token_id, quote_asset, interval))
    row = cur.fetchone()
    return row[0] if row else None

def update_fetch_state(conn, alpha_token_id, quote_asset, interval, last_open_time):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO klines_fetch_state (alpha_token_id, quote_asset, interval, last_open_time)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(alpha_token_id, quote_asset, interval)
        DO UPDATE SET
            last_open_time=excluded.last_open_time,
            updated_at=CURRENT_TIMESTAMP;
    """, (alpha_token_id, quote_asset, interval, last_open_time))
    conn.commit()


def store_klines(conn, alpha_token_id, quote_asset, interval, data):
    """Store klines into alpha_klines table with alpha_token_id and quote_asset split."""
    # Split the full symbol
    # Example: "ALPHA_391USDT" → alpha_token_id="ALPHA_391", quote_asset="USDT"

    sql = """
    INSERT OR IGNORE INTO alpha_klines (
        alpha_token_id, quote_asset, interval, open_time,
        open_price, high_price, low_price, close_price, volume,
        close_time, quote_asset_volume, number_of_trades,
        taker_buy_base_volume, taker_buy_quote_volume, ignored
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = conn.cursor()
    rows = []
    for k in data:
        rows.append((
            alpha_token_id, quote_asset, interval,
            int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
            float(k[5]), int(k[6]), float(k[7]), int(k[8]),
            float(k[9]), float(k[10]), k[11] if len(k) > 11 else None
        ))
    cur.executemany(sql, rows)
    conn.commit()
    print(f"Inserted {len(rows)} klines for {alpha_token_id} ({alpha_token_id}/{quote_asset}).")


def split_symbol(symbol: str):
    """
    Split symbol like 'ALPHA_391USDT' → ('ALPHA_391', 'USDT')
    Handles symbols where alpha_token_id includes an underscore.
    """
    if not symbol.startswith("ALPHA_"):
        raise ValueError(f"Unexpected symbol format: {symbol}")
    # Find where letters stop after the numeric part
    # Example: ALPHA_391USDT -> alpha_token_id='ALPHA_391', quote_asset='USDT'
    parts = symbol.split("USDT")
    if len(parts) == 2 and parts[1] == "":
        return parts[0], "USDT"
    # Fallback: split by last 4 capital letters (common quote assets: USDT, BNB, ETH)
    for quote in ["USDT", "BNB", "BTC", "ETH", "FDUSD"]:
        if symbol.endswith(quote):
            return symbol[:-len(quote)], quote
    return symbol, None



def plot_klines_from_db(db_path, symbol, interval="15m"):
    """Fetch klines for a token from SQLite and plot close_price vs open_time."""

    alpha_token_id = symbol[:-4]  # e.g. ALPHA_391
    quote_asset = symbol[-4:]     # e.g. USDT

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT open_time, close_price
        FROM alpha_klines
        WHERE alpha_token_id = ? AND quote_asset = ? AND interval = ?
        ORDER BY open_time ASC;
    """, (alpha_token_id, quote_asset, interval))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"No data found for {symbol} ({interval}).")
        return

    # Separate columns
    open_times = [
        datetime.utcfromtimestamp(r[0] / 1000).strftime("%M-%H-%d-%m-%Y")
        for r in rows
    ]    
    close_prices = [r[1] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(open_times, close_prices, label=f"{symbol} ({interval})", linewidth=1.2)
    plt.xlabel("Date (dd-mm-yyyy)")
    plt.ylabel("Close Price (USDT)")
    plt.title(f"{symbol} Close Price History ({interval})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()

def fetch_kline_data(conn, symbol):
    data = fetch_all_klines_incremental(conn, symbol)
    if not data:
        print("No data returned.")
        return
    
    conn.close()

QUOTE_ASSET = "USDT"

#updates all token with new kline data
def fetch_all_tokens_and_update(conn):
    cur = conn.cursor()
    cur.execute("SELECT alpha_id FROM alpha_tokens;")
    tokens = [row[0] for row in cur.fetchall()]

    print(f"Found {len(tokens)} tokens in database.")
    for i, alpha_id in enumerate(tokens, start=1):
        symbol = f"{alpha_id}{QUOTE_ASSET}"
        print(f"\n[{i}/{len(tokens)}] Processing {symbol} ...")
        try:
            fetch_all_klines_incremental(conn, symbol)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            time.sleep(2)

def main():
    conn = sqlite3.connect(DB_FILE)

    #fetch single token
    #fetch_kline_data(conn, "ALPHA_391USDT")
    #plot_klines_from_db(DB_FILE, "ALPHA_391USDT")

    #fetch all
    #fetch_all_tokens_and_update(conn)
    


if __name__ == "__main__":
    main()
