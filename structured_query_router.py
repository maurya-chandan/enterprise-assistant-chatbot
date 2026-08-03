"""
Deterministic query router for exact-key lookups and aggregate/filter
questions that vector-similarity RAG cannot answer reliably.

Why this exists: the semantic retriever in customer_rag_orchestration_layer.py
is good at fuzzy, meaning-based questions. It is structurally NOT good at:

  - Exact ID lookups ("PROD0139", "CUST00045", "ORD100002") - nearest-
    neighbor search over short, near-identical templated sentences can
    retrieve the WRONG record, and a small local LLM (llama3.2:1b) will
    narrate whatever it's handed rather than reliably saying "not found."
  - Counts/aggregates/filters ("how many orders", "which products are low
    on stock") - the retriever only ever returns `k` documents, so the LLM
    structurally cannot see all 5,000 orders or all 300 products to count
    or filter across them.
  - Pronoun follow-ups ("price for that product") - there is no
    conversation memory in the RAG chain at all; each request is a fresh,
    independent semantic search.

This module answers those cases directly against the source CSVs -
deterministic, no LLM involved, so no hallucination risk - and returns None
when a question matches neither pattern, signaling the caller to fall back
to the RAG chain. RBAC is NOT enforced in here; the caller
(customer_api_gateway.py) must only call this after confirming the caller's
role is allowed to see customer_service data at all, exactly as it already
does before invoking the RAG chain.
"""

import os
import re
import csv
import threading

DATA_DIR = "./data"
LOW_STOCK_THRESHOLD = 50

_data_lock = threading.Lock()
_customers = None
_products = None
_orders = None
_tracking = None
_first_name_index = None   # {lowercased first name: [customer_id, ...]}
_last_name_index = None    # {lowercased last name: [customer_id, ...]}
_full_name_index = None    # {"lowercased first last": [customer_id, ...]}


def _load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _ensure_loaded():
    global _customers, _products, _orders, _tracking
    global _first_name_index, _last_name_index, _full_name_index
    if _customers is not None:
        return
    with _data_lock:
        if _customers is None:
            _customers = {c["customer_id"]: c for c in _load_csv("customers.csv")}
            _products = {p["product_id"]: p for p in _load_csv("products.csv")}
            _orders = {o["order_id"]: o for o in _load_csv("orders.csv")}
            _tracking = {t["order_id"]: t for t in _load_csv("tracking.csv")}

            _first_name_index = {}
            _last_name_index = {}
            _full_name_index = {}
            for cid, c in _customers.items():
                first = c["first_name"].lower()
                last = c["last_name"].lower()
                _first_name_index.setdefault(first, []).append(cid)
                _last_name_index.setdefault(last, []).append(cid)
                _full_name_index.setdefault(f"{first} {last}", []).append(cid)


ID_PATTERNS = {
    "customer": re.compile(r'\bCUST0*(\d+)\b', re.IGNORECASE),
    "order": re.compile(r'\bORD0*(\d+)\b', re.IGNORECASE),
    "product": re.compile(r'\bPROD0*(\d+)\b', re.IGNORECASE),
}
# Canonical zero-padded width for IDs where a user might drop leading zeros
# (e.g. "PROD7" instead of "PROD0007"). Order IDs don't use zero-padding
# from 1 - they're a flat 100000+n offset - so there's no fallback for them.
ID_WIDTHS = {"customer": ("CUST", 5), "product": ("PROD", 4)}

# Binds the hint word directly to the determiner ("that customer", "this
# order") rather than scanning the whole sentence for any entity-type word.
# That distinction matters: "order number of THAT CUSTOMER" contains both
# "order" and "customer" - if we scanned independently we'd lock onto
# whichever word happens to appear first ("order") and wrongly conclude the
# reference doesn't match a last-mentioned customer, when the phrase is
# unambiguously about "that customer."
REFERENTIAL_WITH_NOUN = re.compile(
    r'\b(?:that|this|the same)\s+(customer|account|client|product|item|sku|order|shipment|package|tracking)\b',
    re.IGNORECASE,
)
BARE_REFERENTIAL = re.compile(r'\b(it|its)\b', re.IGNORECASE)
NOUN_TO_ENTITY_TYPE = {
    "customer": "customer", "account": "customer", "client": "customer",
    "product": "product", "item": "product", "sku": "product",
    "order": "order", "shipment": "order", "package": "order", "tracking": "order",
}

COUNT_TERMS = re.compile(r'\b(how many|count|number of|total)\b', re.IGNORECASE)
LOW_STOCK_TERMS = re.compile(r'\b(low on stock|low stock|running low|restock|out of stock)\b', re.IGNORECASE)
WANTS_ORDERS = re.compile(r'\borders?\b', re.IGNORECASE)
STATUS_WORDS = ["delivered", "shipped", "out for delivery", "processing", "delayed", "cancelled", "returned"]

NAME_TOKEN_RE = re.compile(r"[A-Za-z]+")
MAX_NAME_MATCHES_SHOWN = 30
# A two-word "first last" match (e.g. "Noah Taylor") is specific enough on
# its own - the odds of that exact pair appearing by coincidence in an
# unrelated question are low. A single bare name token is riskier (several
# last names in this dataset - White, Brown, Clark, Lee - are also common
# English words), so single-token matches only count as a name lookup when
# the question also reads like it's asking about a person.
CUSTOMER_NAME_INTENT_HINT = re.compile(
    r'\b(customer|client|account holder|details of|info(?:rmation)?\s+(?:on|for|about)|'
    r'who is|contact\s+(?:info|information)|profile of)\b',
    re.IGNORECASE,
)


def extract_explicit_id(text):
    """Returns (entity_type, canonical_id) for the first recognizable ID in
    `text`. Tries an exact match first, then falls back to re-padding the
    numeric part for customer/product IDs. Returns (None, None) if nothing
    recognizable is found."""
    _ensure_loaded()
    lookups = {"customer": _customers, "order": _orders, "product": _products}

    for entity_type, pattern in ID_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(0).upper()
        number = match.group(1)
        lookup = lookups[entity_type]

        if raw in lookup:
            return entity_type, raw
        if entity_type in ID_WIDTHS:
            prefix, width = ID_WIDTHS[entity_type]
            padded = f"{prefix}{number.zfill(width)}"
            if padded in lookup:
                return entity_type, padded
    return None, None


def resolve_reference(text, last_mentioned):
    """Resolves referential phrasing ('that customer', 'that product', bare
    'it'/'its') against the caller's last-mentioned entity. Returns
    (entity_type, id) or (None, None) if it can't be resolved."""
    if not last_mentioned:
        return None, None

    noun_match = REFERENTIAL_WITH_NOUN.search(text)
    if noun_match:
        hinted_type = NOUN_TO_ENTITY_TYPE[noun_match.group(1).lower()]
        if last_mentioned.get("type") == hinted_type:
            return hinted_type, last_mentioned["id"]
        return None, None  # e.g. "that order" while we were actually discussing a product

    if BARE_REFERENTIAL.search(text):
        return last_mentioned.get("type"), last_mentioned.get("id")

    return None, None


def _format_customer(customer_id):
    c = _customers[customer_id]
    return (
        f"Customer {customer_id}: {c['first_name']} {c['last_name']}. "
        f"Email: {c['email']}. Phone: {c['phone']}. "
        f"Address: {c['address']}, {c['city']}, {c['state']} {c['zip']}. "
        f"Member since {c['join_date']}. Loyalty tier: {c['loyalty_tier']}."
    )


def _format_product(product_id):
    p = _products[product_id]
    return (
        f"Product {product_id}: {p['product_name']} ({p['category']}). "
        f"Current price: ${p['unit_price']}. "
        f"Stock on hand: {p['stock_quantity']} units at {p['warehouse']}."
    )


def _format_order(order_id):
    o = _orders[order_id]
    customer = _customers.get(o["customer_id"])
    customer_desc = (
        f"{customer['first_name']} {customer['last_name']} ({o['customer_id']})"
        if customer else o["customer_id"]
    )
    items = []
    for item in o["items"].split("|"):
        if not item:
            continue
        pid, qty, price = item.split(":")
        pname = _products[pid]["product_name"] if pid in _products else pid
        items.append(f"{qty}x {pname} ({pid}) at ${price} each")

    tracking = _tracking.get(order_id)
    tracking_desc = (
        f"Shipped via {tracking['carrier']} (#{tracking['tracking_number']}), "
        f"status: {tracking['current_status']}, last seen {tracking['last_location']}, "
        f"estimated delivery {tracking['estimated_delivery']}."
        if tracking else "No shipment/tracking record yet."
    )
    return (
        f"Order {order_id} placed by {customer_desc} on {o['order_date']}. "
        f"Items: {'; '.join(items)}. Total: ${o['order_total']}. "
        f"Payment: {o['payment_method']}. Status: {o['order_status']}. {tracking_desc}"
    )


def _answer_orders_for_customer(customer_id):
    customer = _customers.get(customer_id)
    name = f"{customer['first_name']} {customer['last_name']}" if customer else customer_id
    matching = [o for o in _orders.values() if o["customer_id"] == customer_id]

    if not matching:
        return f"No orders found for customer {customer_id} ({name}).", []

    matching.sort(key=lambda o: o["order_date"], reverse=True)
    lines = [
        f"{o['order_id']} - {o['order_date']} - {o['order_status']} - ${o['order_total']}"
        for o in matching
    ]
    answer = f"Customer {customer_id} ({name}) has {len(matching)} order(s):\n" + "\n".join(lines)
    return answer, [o["order_id"] for o in matching]


def _answer_count(text):
    lowered = text.lower()
    for entity_name, lookup in (("order", _orders), ("customer", _customers), ("product", _products)):
        if entity_name in lowered:
            matched_status = next((s for s in STATUS_WORDS if s in lowered), None)
            if entity_name == "order" and matched_status:
                count = sum(1 for o in _orders.values() if o["order_status"].lower() == matched_status)
                return f"There are {count} orders with status '{matched_status.title()}'."
            return f"There are {len(lookup)} {entity_name}s on record."
    return f"There are {len(_orders)} orders, {len(_customers)} customers, and {len(_products)} products on record."


def _answer_low_stock(threshold=LOW_STOCK_THRESHOLD):
    low = sorted(
        (p for p in _products.values() if int(p["stock_quantity"]) < threshold),
        key=lambda p: int(p["stock_quantity"]),
    )
    if not low:
        return f"No products are below the {threshold}-unit low-stock threshold.", []
    lines = [f"{p['product_id']}: {p['product_name']} - {p['stock_quantity']} units left" for p in low]
    answer = f"{len(low)} product(s) below the {threshold}-unit low-stock threshold:\n" + "\n".join(lines)
    return answer, [p["product_id"] for p in low]


def find_name_matches(text):
    """
    Looks for a customer name in `text` against the ACTUAL names in the
    dataset (exact matching, not semantic similarity). Tries the more
    specific two-word "first last" match first; only falls back to a
    single bare name token if the sentence also reads like a person lookup
    (see CUSTOMER_NAME_INTENT_HINT). Returns a list of matching customer_ids
    (possibly empty, possibly more than one if several customers share that
    name - which is common: this dataset has up to 43 customers sharing a
    single first name).
    """
    tokens = [t.lower() for t in NAME_TOKEN_RE.findall(text)]

    for i in range(len(tokens) - 1):
        full = f"{tokens[i]} {tokens[i + 1]}"
        if full in _full_name_index:
            return _full_name_index[full]

    if not CUSTOMER_NAME_INTENT_HINT.search(text):
        return []

    candidates = set()
    for t in tokens:
        candidates.update(_first_name_index.get(t, []))
        candidates.update(_last_name_index.get(t, []))
    return list(candidates)


def _format_ambiguous_name_response(customer_ids):
    ids_sorted = sorted(customer_ids)
    shown = ids_sorted[:MAX_NAME_MATCHES_SHOWN]
    lines = [f"{cid}: {_customers[cid]['first_name']} {_customers[cid]['last_name']}" for cid in shown]
    header = f"Found {len(ids_sorted)} customers matching that name - please specify which one by customer ID:"
    if len(ids_sorted) > MAX_NAME_MATCHES_SHOWN:
        header += f" (showing {MAX_NAME_MATCHES_SHOWN} of {len(ids_sorted)})"
    return {
        "answer": header + "\n" + "\n".join(lines),
        "sources": shown,
        "mentioned": None,
    }


def guess_entity_type(entity_id: str):
    """Best-effort entity type from an ID string, used to record what the
    RAG fallback path was talking about (e.g. from its returned sources)."""
    if not entity_id:
        return None
    upper = entity_id.upper()
    if upper.startswith("CUST"):
        return "customer"
    if upper.startswith("ORD"):
        return "order"
    if upper.startswith("PROD"):
        return "product"
    return None


def route_query(query_text, last_mentioned=None):
    """
    Attempts to answer `query_text` deterministically.
    Returns {"answer": str, "sources": [...], "mentioned": {"type","id"} | None}
    on a match, or None if this isn't a structured-lookup/aggregate question
    - the caller should fall back to the RAG chain in that case.
    """
    _ensure_loaded()
    text = query_text.strip()

    entity_type, entity_id = extract_explicit_id(text)
    if not entity_id:
        entity_type, entity_id = resolve_reference(text, last_mentioned)

    # No ID, no resolvable reference - see if a real customer NAME is in the
    # text. Matched against the actual dataset, not embedding similarity, so
    # it can tell the difference between "no such person" and "there are 27
    # people with that name," instead of guessing or falsely saying neither.
    ambiguous_name_response = None
    if not entity_id:
        name_matches = find_name_matches(text)
        if len(name_matches) == 1:
            entity_type, entity_id = "customer", name_matches[0]
        elif len(name_matches) > 1:
            ambiguous_name_response = _format_ambiguous_name_response(name_matches)

    # A query can NAME a customer while actually asking about a DIFFERENT
    # entity (their orders) - "order number of customer CUST00494" and
    # "order number of that customer" both contain a customer ID/reference
    # AND the word "order", and both mean "list this customer's orders," not
    # "show me the customer's own profile again" and not "how many orders
    # exist in total." This has to be checked before the plain single-entity
    # lookup below, since a customer ID being present would otherwise win by
    # default and silently ignore the "order" intent.
    if entity_type == "customer" and entity_id and WANTS_ORDERS.search(text):
        answer, order_ids = _answer_orders_for_customer(entity_id)
        return {
            "answer": answer,
            "sources": order_ids,
            "mentioned": {"type": "customer", "id": entity_id},
        }

    if entity_id:
        formatter = {"customer": _format_customer, "product": _format_product, "order": _format_order}[entity_type]
        return {
            "answer": formatter(entity_id),
            "sources": [entity_id],
            "mentioned": {"type": entity_type, "id": entity_id},
        }

    # Only reached if the name matched MULTIPLE customers (a single match
    # was already handled above via entity_id) - ask which one, rather than
    # guessing or claiming no record exists.
    if ambiguous_name_response is not None:
        return ambiguous_name_response

    if LOW_STOCK_TERMS.search(text):
        answer, product_ids = _answer_low_stock()
        return {"answer": answer, "sources": product_ids, "mentioned": None}

    if COUNT_TERMS.search(text):
        return {"answer": _answer_count(text), "sources": [], "mentioned": None}

    return None