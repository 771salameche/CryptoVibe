import re

CRYPTOCURRENCIES = {
    "Bitcoin": ["Bitcoin", "bitcoin", "BTC", "btc", "\u20bf"],
    "Ethereum": ["Ethereum", "ethereum", "ETH", "eth", "\u039e"], # U+039E Greek Capital Letter Xi, ETH is the more common identifier
    "Tether": ["Tether", "tether", "USDT", "usdt"],
    "XRP": ["XRP", "xrp"],
    "BNB": ["BNB", "bnb"],
    "USDC": ["USDC", "usdc"],
    "Solana": ["Solana", "solana", "SOL", "sol"],
    "TRON": ["TRON", "tron", "TRX", "trx"],
    "Dogecoin": ["Dogecoin", "dogecoin", "DOGE", "doge", "\u00d0"],
    "Cardano": ["Cardano", "cardano", "ADA", "ada", "\u20b3"],
    "Hyperliquid": ["Hyperliquid", "hyperliquid", "HYPE", "hype"],
    "Bitcoin Cash": ["Bitcoin Cash", "bitcoin cash", "BCH", "bch"],
    "Chainlink": ["Chainlink", "chainlink", "LINK", "link"],
    "UNUS SED LEO": ["UNUS SED LEO", "LEO", "leo"],
    "Stellar": ["Stellar", "stellar", "XLM", "xlm"],
    "Monero": ["Monero", "monero", "XMR", "xmr"],
    "Zcash": ["Zcash", "zcash", "ZEC", "zec"],
    "Ethena USDe": ["Ethena USDe", "USDe"],
    "Litecoin": ["Litecoin", "litecoin", "LTC", "ltc", "\u0141"],
    "Avalanche": ["Avalanche", "avalanche", "AVAX", "avax"],
    "Hedera": ["Hedera", "hedera", "HBAR", "hbar"],
    "Sui": ["Sui", "sui", "SUI"],
    "Dai": ["Dai", "dai", "DAI"],
    "Shiba Inu": ["Shiba Inu", "shiba inu", "SHIB", "shib"],
    "World Liberty Financial": ["World Liberty Financial", "WLFI"],
    "Cronos": ["Cronos", "cronos", "CRO", "cro"],
    "Toncoin": ["Toncoin", "toncoin", "TON", "ton"],
    "PayPal USD": ["PayPal USD", "PYUSD"],
    "Uniswap": ["Uniswap", "uniswap", "UNI", "uni"],
    "Polkadot": ["Polkadot", "polkadot", "DOT", "dot"],
    "Mantle": ["Mantle", "mantle", "MNT", "mnt"],
    "Bittensor": ["Bittensor", "bittensor", "TAO", "tao"],
    "Canton": ["Canton", "canton", "CC", "cc"],
    "Aave": ["Aave", "aave", "AAVE"],
    "World Liberty Financial USD": ["World Liberty Financial USD", "USD1"],
    "Bitget Token": ["Bitget Token", "BGB"],
    "Aster": ["Aster", "aster", "ASTER"],
    "NEAR Protocol": ["NEAR Protocol", "NEAR", "near"],
    "OKB": ["OKB", "okb"],
    "Internet Computer": ["Internet Computer", "ICP", "icp"],
    "Ethereum Classic": ["Ethereum Classic", "ethereum classic", "ETC", "etc"],
    "Ethena": ["Ethena", "ethena", "ENA", "ena"],
    "Pi": ["Pi", "pi", "PI"],
    "Pepe": ["Pepe", "pepe", "PEPE"],
    "Ondo": ["Ondo", "ondo", "ONDO"],
    "Tether Gold": ["Tether Gold", "XAUt"],
    "Kaspa": ["Kaspa", "kaspa", "KAS", "kas"],
    "Worldcoin": ["Worldcoin", "worldcoin", "WLD", "wld"],
    "Aptos": ["Aptos", "aptos", "APT", "apt"],
    "PAX Gold": ["PAX Gold", "PAXG"]
}

# 2. MAJOR EXCHANGES (at least 10)
EXCHANGES = {
    "Binance": ["Binance", "binance", "BNB"],
    "OKX": ["OKX", "okx"],
    "Bybit": ["Bybit", "bybit"],
    "Coinbase": ["Coinbase", "coinbase", "COIN"],
    "KuCoin": ["KuCoin", "kucoin", "KCS"],
    "Gate.io": ["Gate.io", "Gate", "gate"],
    "Crypto.com": ["Crypto.com", "crypto.com", "CRO"],
    "MEXC": ["MEXC", "mexc"],
    "Kraken": ["Kraken", "kraken"],
    "Bitget": ["Bitget", "bitget", "BGB"]
}

# 3. KEY INFLUENCERS (at least 7)
INFLUENCERS = {
    "Vitalik Buterin": ["Vitalik Buterin", "Vitalik", "Buterin", "@VitalikButerin"],
    "Changpeng Zhao": ["Changpeng Zhao", "CZ", "cz_binance", "@cz_binance"],
    "Elon Musk": ["Elon Musk", "Elon", "Musk", "@elonmusk"],
    "Brian Armstrong": ["Brian Armstrong", "Brian", "Armstrong", "@brian_armstrong"],
    "Michael Saylor": ["Michael Saylor", "Saylor", "@saylor"],
    "Anthony Pompliano": ["Anthony Pompliano", "Pomp", "@APompliano"],
    "Andreas Antonopoulos": ["Andreas Antonopoulos", "Andreas", "@aantonop"],
    "Jack Dorsey": ["Jack Dorsey", "Jack", "@jack"]
}


# 4. COMMON CRYPTO TERMS
CRYPTO_TERMS = {
    "positive": [
        "moon", "mooning", "bullish", "bull run", "pump", "hodl", "hold on for dear life",
        "lambo", "diamond hands", "\ud83d\udc8e\ud83d\ude4c", "buy the dip", "btfd", "to the moon"
    ],
    "negative": [
        "dump", "bearish", "bear market", "crash", "scam", "rug pull", "rekt",
        "paper hands", "fud", "fear, uncertainty, and doubt"
    ],
    "events": [
        "halving", "fork", "hard fork", "soft fork", "airdrop", "listing",
        "mainnet", "mainnet launch", "ico", "initial coin offering"
    ]
}

# --- Helper logic to create reverse maps and regex patterns ---

def _create_reverse_map(entity_dict):
    """Creates a reverse map from a variation to its canonical name."""
    reverse_map = {}
    for canonical_name, variations in entity_dict.items():
        for variation in variations:
            reverse_map[variation.lower()] = canonical_name
    return reverse_map

def _create_entity_regex(entity_dict):
    """Creates a compiled regex pattern for all entities in a dictionary."""
    all_variations = [v for variations in entity_dict.values() for v in variations]
    # Sort by length descending to match longer names first (e.g., "Bitcoin Cash" before "Bitcoin")
    all_variations.sort(key=len, reverse=True)
    # Escape special regex characters in variations
    safe_variations = [re.escape(var) for var in all_variations]
    pattern = r"\b(" + "|".join(safe_variations) + r")\b"
    return re.compile(pattern, re.IGNORECASE)

# Create reverse maps for quick lookups
_CRYPTO_REVERSE_MAP = _create_reverse_map(CRYPTOCURRENCIES)
_EXCHANGE_REVERSE_MAP = _create_reverse_map(EXCHANGES)
_INFLUENCER_REVERSE_MAP = _create_reverse_map(INFLUENCERS)

# 6. REGEX PATTERNS
# Pre-compiled regex for each entity type for fast, case-insensitive, whole-word matching.
CRYPTO_REGEX = _create_entity_regex(CRYPTOCURRENCIES)
EXCHANGE_REGEX = _create_entity_regex(EXCHANGES)
INFLUENCER_REGEX = _create_entity_regex(INFLUENCERS)


# 5. HELPER FUNCTIONS
def extract_cryptos(text: str) -> list[str]:
    """
    Extracts all unique cryptocurrency canonical names found in a given text.

    Args:
        text: The string to search for cryptocurrency names.

    Returns:
        A list of unique canonical cryptocurrency names (e.g., ["Bitcoin", "Ethereum"])
    """
    found_matches = CRYPTO_REGEX.findall(text)
    # Use a set for uniqueness and map matches to their canonical names
    canonical_names = {_CRYPTO_REVERSE_MAP[match.lower()] for match in found_matches}
    return sorted(list(canonical_names))

def extract_exchanges(text: str) -> list[str]:
    """
    Extracts all unique exchange canonical names found in a given text.

    Args:
        text: The string to search for exchange names.

    Returns:
        A list of unique canonical exchange names (e.g., ["Binance", "Coinbase"])
    """
    found_matches = EXCHANGE_REGEX.findall(text)
    canonical_names = {_EXCHANGE_REVERSE_MAP[match.lower()] for match in found_matches}
    return sorted(list(canonical_names))

def extract_influencers(text: str) -> list[str]:
    """
    Extracts all unique influencer canonical names found in a given text.

    Args:
        text: The string to search for influencer names.

    Returns:
        A list of unique canonical influencer names (e.g., ["Elon Musk", "Vitalik Buterin"])
    """
    found_matches = INFLUENCER_REGEX.findall(text)
    canonical_names = {_INFLUENCER_REVERSE_MAP[match.lower()] for match in found_matches}
    return sorted(list(canonical_names))


# --- Test Examples ---
if __name__ == "__main__":
    test_text = """
    Big news today! @elonmusk tweeted about Dogecoin again, causing a huge pump.
    I bought some DOGE on Coinbase, but I'm thinking of moving my BTC and ETH
    holdings from Kraken to Binance for better rates. Vitalik Buterin remains silent.
    Is Bitcoin Cash (BCH) still relevant? Asking for a friend. Pomp thinks so.
    Don't get rekt by the next rug pull! Always check the exchange, like gate.io.
    """

    print("-" * 10 + " Testing NER Extraction Functions " + "-" * 10 + "\n")
    print(f"Test Text:\n{test_text}\n")

    # Test crypto extraction
    found_cryptos = extract_cryptos(test_text)
    print(f"Found Cryptocurrencies: {found_cryptos}")
    assert sorted(found_cryptos) == sorted(["Bitcoin", "Ethereum", "Dogecoin", "Bitcoin Cash"])

    # Test exchange extraction
    found_exchanges = extract_exchanges(test_text)
    print(f"Found Exchanges: {found_exchanges}")
    assert sorted(found_exchanges) == sorted(["Coinbase", "Kraken", "Binance", "Gate.io"])

    # Test influencer extraction
    found_influencers = extract_influencers(test_text)
    print(f"Found Influencers: {found_influencers}")
    assert sorted(found_influencers) == sorted(["Elon Musk", "Vitalik Buterin", "Anthony Pompliano"])
    
    # Test edge cases
    edge_case_text = "He read about cardano on a reddit thread, not ada itself."
    print(f"\nEdge Case Text:\n'{edge_case_text}'")
    found_cryptos_edge = extract_cryptos(edge_case_text)
    print(f"Found Cryptocurrencies: {found_cryptos_edge}")
    assert found_cryptos_edge == ["Cardano"]

    print("\n" + "-" * 10 + " All tests passed successfully! " + "-" * 10)

    print("\n" + "-" * 10 + " Dictionary and Term Examples " + "-" * 10)
    print(f"\nTop 5 Cryptos: {list(CRYPTOCURRENCIES.keys())[:5]}")
    print(f"Top 5 Exchanges: {list(EXCHANGES.keys())[:5]}")
    print(f"Positive Terms: {CRYPTO_TERMS['positive'][:5]}")
    print(f"Negative Terms: {CRYPTO_TERMS['negative'][:5]}")
    print(f"Event Terms: {CRYPTO_TERMS['events'][:5]}")
