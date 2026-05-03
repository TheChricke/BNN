import sqlite3

DB_FILE = "binance_data.db"

CREATE_TOKENS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alpha_tokens (
    alpha_id TEXT PRIMARY KEY,
    token_id TEXT,
    symbol TEXT,
    name TEXT,
    chain_id TEXT,
    chain_name TEXT,
    contract_address TEXT,
    decimals INTEGER,
    price REAL,
    percent_change_24h REAL,
    volume_24h REAL,
    market_cap REAL,
    fdv REAL,
    liquidity REAL,
    holders INTEGER,
    total_supply REAL,
    circulating_supply REAL,
    listing_time INTEGER,
    icon_url TEXT,
    chain_icon_url TEXT
);
"""

CREATE_KLINES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alpha_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Split symbol parts
    alpha_token_id TEXT NOT NULL,      -- e.g. "ALPHA_391"
    quote_asset TEXT NOT NULL,         -- e.g. "USDT"

    -- Candle metadata
    interval TEXT NOT NULL,            -- e.g. "15m"
    open_time INTEGER NOT NULL,        -- ms timestamp (epoch)
    open_price REAL NOT NULL,
    high_price REAL NOT NULL,
    low_price REAL NOT NULL,
    close_price REAL NOT NULL,
    volume REAL NOT NULL,
    close_time INTEGER NOT NULL,
    quote_asset_volume REAL,
    number_of_trades INTEGER,
    taker_buy_base_volume REAL,
    taker_buy_quote_volume REAL,
    ignored TEXT,

    -- Constraints
    FOREIGN KEY (alpha_token_id) REFERENCES alpha_tokens(alpha_id),

    -- Prevent duplicates
    UNIQUE(alpha_token_id, quote_asset, interval, open_time)
);
"""

CREATE_FETCH_STATE_SQL= """
CREATE TABLE IF NOT EXISTS klines_fetch_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha_token_id TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    interval TEXT NOT NULL,
    last_open_time INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (alpha_token_id) REFERENCES alpha_tokens(alpha_id),
    UNIQUE(alpha_token_id, quote_asset, interval)
);
"""

CREATE_INDICATORS_SQL= """
CREATE TABLE IF NOT EXISTS alpha_indicators (
    alpha_token_id TEXT,
    open_time INTEGER,
    macd REAL,
    macd_signal REAL,
    macd_hist REAL,
    macd_z REAL,
    PRIMARY KEY (alpha_token_id, open_time),
    FOREIGN KEY (alpha_token_id) REFERENCES alpha_klines(alpha_token_id)
);
"""

CREATE_MODEL_TRAINING_LOSS_SQL= """
CREATE TABLE IF NOT EXISTS model_training_loss (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ma1 INTEGER,
    ma2 INTEGER,
    ma3 INTEGER,
    vol_window INTEGER,
    look_forward INTEGER,
    lr REAL,
    kl_weight REAL,
    epochs INTEGER,
    layer_size INTEGER,
    final_loss REAL
)
"""

def create_database():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Create table
    #cur.execute(CREATE_TOKENS_TABLE_SQL)
    #conn.commit()

    #cur.execute(CREATE_KLINES_TABLE_SQL)
    #conn.commit()

    cur.execute(CREATE_MODEL_TRAINING_LOSS_SQL)
    conn.commit()

    conn.close()

if __name__ == "__main__":
    create_database()
