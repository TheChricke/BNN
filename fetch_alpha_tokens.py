import sqlite3
import requests

# Configuration
DB_FILE = "binance_data.db"
TABLE_NAME = "alpha_tokens"
ALPHA_TOKEN_LIST_URL = "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"

def fetch_alpha_tokens():
    resp = requests.get(ALPHA_TOKEN_LIST_URL)
    resp.raise_for_status()
    obj = resp.json()
    # The docs show the structure as { code, message, data: [ {...}, ... ] } :contentReference[oaicite:1]{index=1}
    data = obj.get("data")
    if data is None:
        raise RuntimeError("No data field in response: " + str(obj))
    return data

def store_tokens(tokens, conn):
    cur = conn.cursor()
    sql = """
    INSERT OR REPLACE INTO alpha_tokens (
        alpha_id, token_id, symbol, name, chain_id, chain_name,
        contract_address, decimals, price, percent_change_24h, volume_24h,
        market_cap, fdv, liquidity, holders, total_supply, circulating_supply,
        listing_time, icon_url, chain_icon_url
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    data = []
    for t in tokens:
        data.append((
            t.get("alphaId"),
            t.get("tokenId"),
            t.get("symbol"),
            t.get("name"),
            t.get("chainId"),
            t.get("chainName"),
            t.get("contractAddress"),
            t.get("decimals"),
            _to_float(t.get("price")),
            _to_float(t.get("percentChange24h")),
            _to_float(t.get("volume24h")),
            _to_float(t.get("marketCap")),
            _to_float(t.get("fdv")),
            _to_float(t.get("liquidity")),
            _to_int(t.get("holders")),
            _to_float(t.get("totalSupply")),
            _to_float(t.get("circulatingSupply")),
            t.get("listingTime"),
            t.get("iconUrl"),
            t.get("chainIconUrl"),
        ))
    cur.executemany(sql, data)
    conn.commit()
    print(f"✅ Inserted or updated {len(tokens)} tokens.")

def main():
    # Open (or create) DB
    conn = sqlite3.connect(DB_FILE)
    try:
        tokens = fetch_alpha_tokens()
        print(f"Fetched {len(tokens)} tokens")

        store_tokens(tokens, conn)
        print("Inserted tokens into database")

    finally:
        conn.close()

def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except:
        return None
    
def _to_int(v):
    try:
        return int(v) if v not in (None, "") else None
    except:
        return None

if __name__ == "__main__":
    main()
