"""
Converts the structured CSVs (customers, products, orders, tracking) into
short natural-language documents and embeds them into a Chroma vector store,
the same way data_ingestion_pipeline.py did for PDFs in the original project.

Key difference from the PDF version: there is no per-source classification
lookup here, because ALL of this data belongs to a single sensitivity tier.
Every document this script produces is tagged classification="customer_service"
- that single tag is what api_gateway_rbac_controller.py's retriever filter
checks. Only roles whose ROLE_PERMISSIONS include "customer_service" (admin
and customer_service_agent) will ever get chunks back; every other role's
retriever filter excludes everything, so the LLM sees no context and reports
it doesn't have the information. That's the actual access-control boundary,
not the API layer.

One row -> one short document, not one row -> one chunk-split document. These
summaries are a couple of sentences long, so splitting them further would
just fragment a single order/customer/product record across multiple chunks
for no benefit - unlike the PDF pipeline, which chunks because individual
pages are much longer than the retriever's useful context window.
"""

import os
import csv
import hashlib

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "./data"
CHROMA_DB_DIRECTORY = "./chroma_db_customer_service"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLASSIFICATION = "customer_service"  # single tier - see module docstring


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Run generate_synthetic_data.py first.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_customer_documents(customers):
    docs = []
    for c in customers:
        text = (
            f"Customer {c['customer_id']}: {c['first_name']} {c['last_name']}. "
            f"Email: {c['email']}. Phone: {c['phone']}. "
            f"Address: {c['address']}, {c['city']}, {c['state']} {c['zip']}. "
            f"Member since {c['join_date']}. Loyalty tier: {c['loyalty_tier']}."
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "classification": CLASSIFICATION,
                "doc_type": "customer",
                "customer_id": c["customer_id"],
                "source": "customers.csv",
            },
        ))
    return docs


def build_product_documents(products):
    docs = []
    for p in products:
        text = (
            f"Product {p['product_id']}: {p['product_name']} ({p['category']}). "
            f"Current price: ${p['unit_price']}. "
            f"Stock on hand: {p['stock_quantity']} units at {p['warehouse']}."
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "classification": CLASSIFICATION,
                "doc_type": "product",
                "product_id": p["product_id"],
                "source": "products.csv",
            },
        ))
    return docs


def _describe_items(items_field, product_lookup):
    """Turns the encoded 'PROD0012:2:19.99|PROD0045:1:9.99' items field into
    a readable list, resolving product IDs to names where possible."""
    parts = []
    for item in items_field.split("|"):
        if not item:
            continue
        product_id, qty, unit_price = item.split(":")
        product = product_lookup.get(product_id)
        name = product["product_name"] if product else product_id
        parts.append(f"{qty}x {name} ({product_id}) at ${unit_price} each")
    return "; ".join(parts) if parts else "no items on record"


def build_order_documents(orders, customer_lookup, product_lookup, tracking_lookup):
    docs = []
    for o in orders:
        customer = customer_lookup.get(o["customer_id"])
        customer_desc = (
            f"{customer['first_name']} {customer['last_name']} ({o['customer_id']}, "
            f"email {customer['email']}, phone {customer['phone']})"
            if customer else f"unknown customer {o['customer_id']}"
        )
        items_desc = _describe_items(o["items"], product_lookup)

        tracking = tracking_lookup.get(o["order_id"])
        if tracking:
            tracking_desc = (
                f"Shipped via {tracking['carrier']} (tracking #{tracking['tracking_number']}) "
                f"on {tracking['shipped_date']}. Current status: {tracking['current_status']}, "
                f"last seen at {tracking['last_location']}. "
                f"Estimated delivery: {tracking['estimated_delivery']}."
            )
        else:
            tracking_desc = "No shipment/tracking record yet."

        text = (
            f"Order {o['order_id']} placed by {customer_desc} on {o['order_date']}. "
            f"Items: {items_desc}. Order total: ${o['order_total']}. "
            f"Payment method: {o['payment_method']}. Order status: {o['order_status']}. "
            f"{tracking_desc}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "classification": CLASSIFICATION,
                "doc_type": "order",
                "order_id": o["order_id"],
                "customer_id": o["customer_id"],
                "source": "orders.csv",
            },
        ))
    return docs


def _stable_doc_id(doc: Document) -> str:
    """Deterministic ID from doc_type + the record's natural key, so re-running
    ingestion upserts existing vectors instead of duplicating them."""
    doc_type = doc.metadata.get("doc_type", "")
    natural_key = (
        doc.metadata.get("order_id")
        or doc.metadata.get("customer_id")
        or doc.metadata.get("product_id")
        or ""
    )
    raw = f"{doc_type}|{natural_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_vector_store(documents, db_dir: str):
    print(f"Initializing HuggingFace Embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    ids = [_stable_doc_id(doc) for doc in documents]

    print(f"Embedding {len(documents)} documents into ChromaDB (this can take a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir,
        ids=ids,
    )
    print(f"Vector store successfully updated and persisted at: {db_dir}")
    return vectorstore


def main():
    customers = load_csv("customers.csv")
    products = load_csv("products.csv")
    orders = load_csv("orders.csv")
    tracking = load_csv("tracking.csv")

    if not (customers and products and orders):
        print("Missing source data. Run generate_synthetic_data.py first, then re-run this script.")
        return

    customer_lookup = {c["customer_id"]: c for c in customers}
    product_lookup = {p["product_id"]: p for p in products}
    tracking_lookup = {t["order_id"]: t for t in tracking}

    print("Building documents...")
    documents = []
    documents.extend(build_customer_documents(customers))
    documents.extend(build_product_documents(products))
    documents.extend(build_order_documents(orders, customer_lookup, product_lookup, tracking_lookup))
    print(
        f"Built {len(documents)} documents total "
        f"({len(customers)} customer + {len(products)} product + {len(orders)} order docs), "
        f"all tagged classification='{CLASSIFICATION}'."
    )

    build_vector_store(documents, CHROMA_DB_DIRECTORY)
    print("Customer Service Data Ingestion Pipeline completed successfully.")


if __name__ == "__main__":
    main()
