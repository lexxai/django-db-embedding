import hashlib


def hash_query(query_text: str) -> str:
    """Return SHA256 hex digest of query to protect privacy."""
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()
