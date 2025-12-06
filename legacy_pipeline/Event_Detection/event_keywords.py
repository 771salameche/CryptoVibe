"""
A comprehensive event keyword taxonomy for detecting crypto-related events
in social media posts.
"""
import re

# Structured dictionary format for event categories.
# Each category includes a name, keywords, sentiment, priority (higher is more important),
# and a color code for visualization.
EVENT_TAXONOMY = {
    # --- Negative Events (Highest Priority) ---
    'HACK': {
        'name': 'Hack / Exploit',
        'keywords': [
            'hacked', 'exploit', 'stolen', 'security breach', 'vulnerability',
            'rug pull', '51% attack', 'double spend', 'funds compromised',
            'private key leak', r'drained of \$\d+', 'flash loan attack'
        ],
        'sentiment': 'negative',
        'priority': 10,
        'color': '#FF0000'  # Red
    },
    'CRASH': {
        'name': 'Market Crash',
        'keywords': [
            'crash', 'dump', 'plunge', 'liquidation', 'wipeout', 'nosedive',
            'flash crash', 'major sell-off', 'blood bath', 'plummeted', 'rekt'
        ],
        'sentiment': 'negative',
        'priority': 9,
        'color': '#FF4500'  # OrangeRed
    },
    'REGULATION': {
        'name': 'Regulatory Action',
        'keywords': [
            'sec', 'cftc', 'lawsuit', 'banned', 'illegal', 'regulation',
            'investigation', 'crackdown', 'subpoena', 'regulatory action',
            'enforcement', 'unregistered securities', 'delisted by order'
        ],
        'sentiment': 'negative',
        'priority': 8,
        'color': '#FFD700'  # Gold
    },
    'OUTAGE': {
        'name': 'Network Outage',
        'keywords': [
            'down', 'offline', 'outage', 'network issue', 'maintenance',
            'transactions? suspended', 'node issues', 'syncing issues',
            'network halted', 'congestion'
        ],
        'sentiment': 'negative',
        'priority': 7,
        'color': '#FFA500'  # Orange
    },
    # --- Positive Events ---
    'LISTING': {
        'name': 'Exchange Listing',
        'keywords': [
            'listed on', 'now available on', 'added to', 'coming to',
            r'\blist(ed|ing|s)?\b on', 'new listing', 'trading pair',
            'launches on', 'starts trading'
        ],
        'sentiment': 'positive',
        'priority': 6,
        'color': '#00FF00'  # Lime
    },
    'PARTNERSHIP': {
        'name': 'Partnership / Collaboration',
        'keywords': [
            'partnership', 'collaboration', 'teams up with', 'in collaboration with',
            'strategic alliance', 'partnered with', 'joins forces', 'joint venture'
        ],
        'sentiment': 'positive',
        'priority': 5,
        'color': '#32CD32'  # LimeGreen
    },
    'UPGRADE': {
        'name': 'Tech Upgrade / Launch',
        'keywords': [
            'mainnet', 'testnet', 'hard fork', 'soft fork', 'protocol upgrade',
            'launches', 'upgrade complete', r'v\d(\.\d+)?', 'version \d',
            'sharding', 'roadmap update'
        ],
        'sentiment': 'positive',
        'priority': 4,
        'color': '#008000'  # Green
    },
    'ADOPTION': {
        'name': 'Adoption',
        'keywords': [
            'accepts', 'integrated by', 'supports', 'added support for',
            'powered by', 'built on', 'adopts', 'now accepting'
        ],
        'sentiment': 'positive',
        'priority': 3,
        'color': '#2E8B57'  # SeaGreen
    },
    # --- Neutral Events (Lowest Priority) ---
    'CONFERENCE': {
        'name': 'Conference / Event',
        'keywords': [
            'conference', 'summit', 'webinar', 'meetup', 'speaking at',
            'panelist at', 'keynote', 'hackathon'
        ],
        'sentiment': 'neutral',
        'priority': 2,
        'color': '#1E90FF'  # DodgerBlue
    },
    'ANNOUNCEMENT': {
        'name': 'General Announcement',
        'keywords': [
            'announces', 'reveals', 'upcoming', 'plans to', 'releases',
            'whitepaper', 'ama', 'ask me anything'
        ],
        'sentiment': 'neutral',
        'priority': 1,
        'color': '#808080'  # Gray
    },
}

def classify_event_type(text: str) -> tuple[str, str] or tuple[None, None]:
    """
    Classifies a text based on the highest-priority event keyword found.

    Args:
        text: The social media post or text to classify.

    Returns:
        A tuple containing the event name and its sentiment (e.g., ('Hack / Exploit', 'negative')),
        or (None, None) if no event keywords are found.
    """
    found_events = []
    text_lower = text.lower()

    for category_code, details in EVENT_TAXONOMY.items():
        for keyword in details['keywords']:
            # Check if the keyword is a regex pattern or a plain string
            if re.search(r'[\(\)\|\?\+\*\\]', keyword):  # Simple regex char check
                if re.search(keyword, text_lower):
                    found_events.append(details)
                    break  # Move to the next category once one keyword matches
            elif keyword in text_lower:
                found_events.append(details)
                break  # Move to the next category

    if not found_events:
        return None, None

    # Sort found events by priority (highest first) and return the top one
    highest_priority_event = max(found_events, key=lambda x: x['priority'])
    return highest_priority_event['name'], highest_priority_event['sentiment']


# --- Validation Examples ---
if __name__ == "__main__":
    sample_posts = [
        "BREAKING: Major exchange has been hacked for $50M in ETH. All transactions suspended.",
        "Huge news! Our token will be listed on Coinbase Pro next week!",
        "The much-awaited mainnet v2.0 upgrade is finally live. Expect faster transactions.",
        "Our team is thrilled to announce a strategic partnership with Microsoft to build on Azure.",
        "The SEC has filed a lawsuit against Ripple Labs, calling XRP an unregistered security.",
        "Don't miss our CEO speaking at the Future of Crypto summit in Miami!",
        "The network is down for emergency maintenance after a vulnerability was found.",
        "Wow, the market just dumped hard. My portfolio is completely rekt.",
        "We are proud to announce that our online store now accepts Dogecoin for payments.",
        "The team just released the new roadmap for 2025. Lots of exciting things upcoming.",
        "A flash loan attack on the new DeFi protocol resulted in a huge loss of funds.",
        "Trading for the new SHIB/USDT pair is now available on Binance.",
        "Our lead dev will be doing an AMA on Reddit tomorrow to answer all your questions.",
        "The Bitcoin network just experienced a flash crash, liquidating over $1B in longs.",
        "We are investigating a network outage. Further details will be shared soon."
    ]

    print("--- Event Keyword Classification Validation ---")
    for i, post in enumerate(sample_posts):
        event_name, sentiment = classify_event_type(post)
        print(f"\nPost {i+1}: \"{post}\" ")
        if event_name:
            print(f"  -> Detected Event: '{event_name}' (Sentiment: {sentiment})")
        else:
            print("  -> No specific event detected.")
    print("\n--- Validation Complete ---")
