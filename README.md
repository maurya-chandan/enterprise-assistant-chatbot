# Customer Service Assistant (RBAC + RAG)

Same architecture as the earlier HR-policy project - a Streamlit chat UI talks
to a FastAPI gateway that enforces role-based access before invoking a local
RAG chain - rebuilt around synthetic customer/order/product/tracking data.

**The access rule for this project: only `admin` and `customer_service_agent`
can retrieve anything. Every other role gets zero results, enforced both by
an early 403 and, independently, by the vector-search filter itself.**

## Files

| File | Purpose |
|---|---|
| `generate_synthetic_data.py` | Creates `./data/customers.csv`, `products.csv`, `orders.csv`, `tracking.csv` |
| `customer_data_ingestion_pipeline.py` | Converts those CSVs into text documents (all tagged `classification=customer_service`) and embeds them into Chroma |
| `customer_rag_orchestration_layer.py` | Builds the retrieval chain, scoped to whatever classifications the caller is allowed |
| `structured_query_router.py` | Answers exact-ID lookups, counts, and low-stock questions directly against the CSVs - no LLM, no hallucination risk. See "Why the router exists" below |
| `customer_api_gateway.py` | FastAPI gateway: verifies role, enforces RBAC, tries the router first, falls back to the RAG chain, masks PII in the question |
| `customer_frontend_chat_ui.py` | Streamlit chat UI with a role selector (`admin` / `customer_service_agent` / `guest`) |

## Why the router exists

Semantic (vector-embedding) search is great for fuzzy, meaning-based
questions, but it's the wrong tool for a few question shapes that come up
constantly with this kind of data - and using it anyway doesn't just give
imprecise answers, it gives *confidently wrong* ones:

- **Exact ID lookups** ("what's the phone number for CUST00045") - the
  retriever only returns the `k` nearest-by-embedding documents, and short,
  near-identical templated sentences can rank the *wrong* customer/order/
  product as "close enough." A small local LLM (`llama3.2:1b`) will then
  narrate whatever it's handed - including a different customer's real
  phone number - rather than reliably saying "not found."
- **Counts and aggregates** ("how many orders are there", "what's low on
  stock") - the retriever caps out at `k` documents, so the LLM structurally
  cannot see all 5,000 orders or all 300 products to count or filter across
  them. It will confidently answer based on whatever handful it was handed.
- **Pronoun follow-ups** ("price for that product") - the RAG chain has no
  conversation memory; every request is an independent semantic search, so
  "that product" has nothing to resolve against on its own.
- **Name lookups** ("details of Noah") - a first name alone is often
  ambiguous (this dataset has up to 43 customers sharing one first name).
  Semantic search doesn't know that - it just returns whichever handful of
  documents rank closest and the LLM narrates them, sometimes claiming "no
  record" while listing unrelated people. The router checks the real name
  index and either resolves unambiguously or honestly says how many people
  match and lists them, rather than guessing.

`structured_query_router.py` handles all three by answering directly from
the CSVs - deterministic, exact, and with a small per-`user_id` memory of
"the last entity mentioned" so simple follow-ups resolve correctly. It
returns `None` for anything that doesn't match one of these patterns, and
the gateway falls back to the semantic RAG chain for those - which remains
the right tool for genuinely open-ended questions like "summarize this
customer's order history."

Sample generated data is already included under `./data/` so you can run
ingestion immediately - `generate_synthetic_data.py` is there so you can
regenerate it (e.g. change `NUM_ORDERS` to scale up or down) whenever you want.

## Setup

`requirements.txt`  "Run it" below.

pip install -r requirements.txt


fresh environment:

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

You'll also need Ollama running locally with the same model as before:
```bash
ollama pull llama3.2:1b
ollama serve
```

## Run it

**1. Generate the data** (skip if you're keeping the included sample data):
```bash
python generate_synthetic_data.py
```

**2. Ingest it into the vector store:**
```bash
python customer_data_ingestion_pipeline.py
```
This downloads the embedding model on first run and writes to
`./chroma_db_customer_service/`.

**3. Start the gateway** (a different port from the HR-policy project, `8001`,
so you can run both projects side by side if you want):
```bash
uvicorn customer_api_gateway:app --host 127.0.0.1 --port 8001 --reload
```
Wait for `Uvicorn running on http://127.0.0.1:8001` before moving on.

**4. Start the chat UI**, in a separate terminal:
```bash
streamlit run customer_frontend_chat_ui.py
```

## Try it

With role set to `admin` or `customer_service_agent`, ask things like:
- "How many orders are there?" - answered by the router, exact count
- "What's the phone number for customer CUST00045?" - exact record, no guessing
- "What products are low on stock?" - real `stock_quantity` filter, not a guess
- "PROD0139 product stock" then, as a follow-up, "price for that product" -
  the router remembers PROD0139 was just mentioned and resolves "that
  product" correctly
- "Full details of customer CUST00494" then "order number of that customer" -
  correctly lists CUST00494's actual orders, not the global order count and
  not the customer's profile again
- "Details of Noah" - since many customers share that first name, this
  correctly lists all of them with their IDs instead of guessing or saying
  "no record"
- "Details of Noah Taylor" - a full name is specific enough to resolve
  directly to one customer
- "Summarize customer CUST00045's order history" - open-ended enough that
  this still goes through the semantic RAG chain, which is the right tool
  for it

Check the "🔍 View Gateway & RAG Details" expander under any answer - it now
shows an **Answer Path** (`structured_lookup` vs `semantic_rag`) so you can
see which one handled a given question.

Then switch the sidebar role to `guest` and ask the same question again -
you should get a 403 ("Access Denied") instead of an answer. That's the RBAC
boundary working as intended.

## Notes / things to adapt before any real use

- The mock IdP (HMAC-signed role header) is a local dev convenience, not
  real authentication - swap it for actual SSO/OIDC before this touches real
  data, same caveat as the original project.
- Because this dataset is customer PII by nature (emails, phones, addresses),
  the assistant will legitimately surface that PII to `admin` and
  `customer_service_agent` - that's expected for a support tool, not a leak.
  The PII scrubber in the gateway only masks PII an agent might type into
  their own question, not what comes back in answers.
- `order_id`, `customer_id`, and `product_id` are used as natural keys for
  stable vector IDs, so re-running ingestion after regenerating data upserts
  cleanly instead of duplicating records.
- The "last mentioned entity" follow-up memory in `customer_api_gateway.py`
  is a plain in-process dict keyed by `user_id` - it resets on restart and
  won't sync if you ever run multiple gateway workers/processes. Fine for
  this local demo; a real deployment would move it to shared storage (Redis,
  a DB row) keyed by session, not just `user_id`.
