import re


COLOR_MAP = {
    "navy blue": "Navy Blue",
    "navy": "Navy Blue",
    "blue": "Blue",
    "black": "Black",
    "white": "White",
    "purple": "Purple",
    "red": "Red",
    "green": "Green",
    "yellow": "Yellow",
    "pink": "Pink",
    "brown": "Brown",
    "grey": "Grey",
    "gray": "Grey",
}

GENDER_MAP = {
    "men": "Men",
    "man": "Men",
    "male": "Men",
    "women": "Women",
    "woman": "Women",
    "female": "Women",
}

ARTICLE_TYPE_MAP = {
    "shirt": "Shirts",
    "shirts": "Shirts",
    "tshirt": "Tshirts",
    "t-shirts": "Tshirts",
    "t-shirt": "Tshirts",
    "tee": "Tshirts",
    "tees": "Tshirts",
    "sock": "Socks",
    "socks": "Socks",
    "shoe": "Shoes",
    "shoes": "Shoes",
    "dress": "Dresses",
    "dresses": "Dresses",
    "top": "Tops",
    "tops": "Tops",
    "jean": "Jeans",
    "jeans": "Jeans",
    "legging": "Leggings",
    "leggings": "Leggings",
    "skirt": "Skirts",
    "skirts": "Skirts",
    "kurta": "Kurtas",
    "kurtas": "Kurtas",
    "sandal": "Sandals",
    "sandals": "Sandals",
    "flip flop": "Flip Flops",
    "flip flops": "Flip Flops",
    "bag": "Handbags",
    "handbag": "Handbags",
    "handbags": "Handbags",
}

USAGE_MAP = {
    "casual": "Casual",
    "formal": "Formal",
    "sport": "Sports",
    "sports": "Sports",
}

STOPWORDS = {
    "for", "with", "in", "on", "at", "of", "a", "an", "the", "i", "want", "need"
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_first_match(query: str, mapping: dict):
    for key in sorted(mapping.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, query):
            return mapping[key]
    return None


def build_search_query(q: str) -> str:
    tokens = [t for t in q.split() if t not in STOPWORDS]
    return " ".join(tokens)


def parse_query(query: str) -> dict:
    q = normalize_text(query)

    color = find_first_match(q, COLOR_MAP)
    gender = find_first_match(q, GENDER_MAP)
    article_type = find_first_match(q, ARTICLE_TYPE_MAP)
    usage = find_first_match(q, USAGE_MAP)

    search_query = build_search_query(q)

    return {
        "original_query": query,
        "search_query": search_query,
        "strict_filters": {
            "gender": gender,
            "usage": usage,
            "article_type": article_type,
            "color": color,
        },
        "soft_filters": {
            "gender": gender,
            "usage": usage,
        }
    }


if __name__ == "__main__":
    query = "navy blue shirt for men"
    result = parse_query(query)
    print(result)