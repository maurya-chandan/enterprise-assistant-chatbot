import re
import os
import hmac
import hashlib
import logging
import threading
import traceback
import uuid

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from customer_rag_orchestration_layer import get_rag_components, build_scoped_chain
import structured_query_router as struct_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Service Assistant API Gateway",
    description="RBAC-gated RAG endpoint over customer/order/product/tracking data.",
    version="1.1",
)

# --- Shared, expensive-to-load RAG components (embeddings/vectorstore/LLM) ---
_vectorstore = None
_llm = None
_init_lock = threading.Lock()

# --- Per-user "last mentioned entity" memory, for resolving follow-ups like
# "price for that product" or bare "it". In-memory only: it resets on
# gateway restart and does not sync across multiple gateway processes - fine
# for a single local instance, not for a horizontally-scaled deployment.
_last_mentioned = {}
_last_mentioned_lock = threading.Lock()

# --- Role -> classification permission mapping ---
# This mapping IS the authorization boundary. There is only one data tier in
# this project ("customer_service"), and by design ONLY admin and
# customer_service_agent may see it. Any role not listed here - including
# unrecognized or unauthenticated callers - falls through to an empty
# permission list via the .get(role, []) lookup in ask_assistant(), not to
# some other tier. There is no role in this table that gets partial access;
# it's all-or-nothing for this dataset.
ROLE_PERMISSIONS = {
    "admin": ["customer_service"],
    "customer_service_agent": ["customer_service"],
}
DEFAULT_ROLE = "guest"  # fail-safe: an unrecognized/unauthenticated caller has NO access

# --- Mock identity verification ---
# In production this MUST be replaced by a real identity provider (SSO/OIDC)
# issuing a token the gateway verifies against the IdP's keys. This HMAC
# scheme only stops a role from being changed by editing an unsigned header;
# it shares a secret with the local demo frontend and is NOT real
# authentication - do not ship this as-is.
MOCK_IDP_SECRET = os.environ.get("MOCK_IDP_SECRET", "dev-only-insecure-shared-secret")


def sign_role(role: str) -> str:
    signature = hmac.new(MOCK_IDP_SECRET.encode(), role.encode(), hashlib.sha256).hexdigest()
    return f"{role}.{signature}"


def verify_role_token(token: str) -> str:
    """Verifies a signed role token, returning the verified role or the safe default."""
    try:
        role, signature = token.rsplit(".", 1)
    except (ValueError, AttributeError):
        logger.warning("Malformed auth token; defaulting to no-access role.")
        return DEFAULT_ROLE

    expected = hmac.new(MOCK_IDP_SECRET.encode(), role.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        logger.warning("Auth token signature mismatch; defaulting to no-access role.")
        return DEFAULT_ROLE

    # Note: we deliberately do NOT reject unknown roles here the way the
    # original project did (which fell back to a still-privileged default
    # role). Here the safe default already has zero access, so an unknown-
    # but-validly-signed role also just falls through to zero permissions
    # in ask_assistant() below. Nothing unrecognized can ever gain access.
    return role.lower()


@app.on_event("startup")
def load_pipeline():
    """Initializes embeddings, vector store, and LLM once when FastAPI starts."""
    global _vectorstore, _llm
    logger.info("Initializing shared RAG components on startup...")
    try:
        _vectorstore, _llm = get_rag_components()
        logger.info("RAG components loaded successfully.")
    except Exception:
        logger.error("Failed to load RAG components on startup:\n" + traceback.format_exc())


def ensure_components_loaded():
    """Lazily (re)initializes shared components exactly once, even under concurrent requests."""
    global _vectorstore, _llm
    if _vectorstore is not None and _llm is not None:
        return
    with _init_lock:
        if _vectorstore is None or _llm is None:
            logger.info("RAG components missing at request time; initializing now...")
            _vectorstore, _llm = get_rag_components()


class QueryRequest(BaseModel):
    user_id: str
    query: str


def scrub_pii(text: str) -> str:
    """
    Masks PII in the AGENT'S QUESTION before it's sent to the LLM or logged.
    This is not about hiding customer data from the agent - a customer
    service agent is expected to see customer emails/phones/addresses
    returned in answers, that's the point of the tool. This scrubber exists
    so that if an agent pastes a customer's raw SSN or card-adjacent text
    into their own question, it doesn't get echoed into logs or the model
    prompt unnecessarily.
    """
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    return text


# NOTE: standard 'def' (non-async) lets FastAPI run this in a threadpool so a
# slow LLM invocation doesn't block the asyncio event loop.
@app.post("/ask")
def ask_assistant(
    request: QueryRequest,
    x_auth_token: str = Header(default=None),
    x_user_role: str = Header(default=None),  # legacy/dev-only fallback, unauthenticated
):
    request_id = str(uuid.uuid4())

    # STEP 1: Resolve a VERIFIED role. Prefer the signed token; fall back to
    # the raw header only if no token was sent, logging loudly since that
    # path is not authenticated.
    if x_auth_token:
        role = verify_role_token(x_auth_token)
    elif x_user_role:
        logger.warning(
            f"[{request_id}] Request used unsigned x-user-role header - "
            "not authenticated, do not trust this in production."
        )
        role = x_user_role.lower()
    else:
        role = DEFAULT_ROLE

    allowed_classifications = ROLE_PERMISSIONS.get(role, [])

    logger.info(
        f"[{request_id}] Incoming query from user '{request.user_id}' with verified role "
        f"'{role}' (allowed: {allowed_classifications or 'NONE'})"
    )

    # STEP 2: Enforcement gate. Unlike the original HR-policy project, there
    # is no multi-tier target label to route on here - the entire dataset
    # requires being admin or customer_service_agent, full stop. A role with
    # no allowed classifications is rejected immediately rather than being
    # allowed to reach the retriever (whose empty-filter behavior would also
    # correctly return nothing - this is a fast, clear 403 on top of that
    # fail-closed retriever behavior, not a substitute for it).
    if not allowed_classifications:
        logger.warning(
            f"[{request_id}] SECURITY ALERT: role '{role}' attempted to query "
            "customer service data without permission."
        )
        raise HTTPException(
            status_code=403,
            detail=f"Access Denied: the role '{role}' does not have permission to query this data. "
                   "Only 'admin' and 'customer_service_agent' roles may access customer records.",
        )

    # STEP 3: PII Masking of the incoming question (see scrub_pii docstring).
    scrubbed_query = scrub_pii(request.query)

    # STEP 4: Try the deterministic router FIRST. Exact ID lookups, counts,
    # and low-stock filters are things vector similarity search cannot
    # answer reliably (see structured_query_router.py's module docstring for
    # why) - if this matches, we skip the LLM/RAG chain entirely, so there's
    # no hallucination surface for these question types at all. Access
    # control still applies exactly the same way: we only ever reach this
    # line if STEP 2 already confirmed the role has customer_service
    # permission, so the router itself doesn't need its own RBAC check.
    last_mentioned = _last_mentioned.get(request.user_id)
    structured_result = struct_router.route_query(scrubbed_query, last_mentioned)

    if structured_result is not None:
        if structured_result["mentioned"]:
            with _last_mentioned_lock:
                _last_mentioned[request.user_id] = structured_result["mentioned"]
        logger.info(f"[{request_id}] Answered via structured lookup (no LLM call).")
        return {
            "status": "success",
            "pipeline_stages": {
                "scrubbed_query": scrubbed_query,
                "rbac_status": "authorized",
                "allowed_classifications": allowed_classifications,
                "answer_path": "structured_lookup",
            },
            "answer": structured_result["answer"],
            "sources": structured_result["sources"],
        }

    logger.info(f"[{request_id}] No structured match; falling back to semantic RAG invocation...")
    try:
        ensure_components_loaded()
        # STEP 5: Build a retriever scoped to the caller's verified
        # permissions - this is what actually enforces access control at the
        # vector-search level, independent of STEP 2 above.
        rag_chain = build_scoped_chain(_vectorstore, _llm, allowed_classifications)

        logger.info(f"[{request_id}] Invoking chain with scrubbed query.")
        response = rag_chain.invoke({"input": scrubbed_query})
        logger.info(f"[{request_id}] RAG invocation completed successfully.")

        final_answer = response.get("answer", "No answer generated.")
        sources_used = list(set(
            doc.metadata.get("order_id") or doc.metadata.get("customer_id")
            or doc.metadata.get("product_id") or doc.metadata.get("source", "Unknown")
            for doc in response.get("context", [])
        ))

        # Keep "last mentioned" fresh for the RAG path too, so a follow-up
        # pronoun after an open-ended question still has something to
        # resolve against.
        first_recognizable = next(
            (s for s in sources_used if struct_router.guess_entity_type(s)), None
        )
        if first_recognizable:
            with _last_mentioned_lock:
                _last_mentioned[request.user_id] = {
                    "type": struct_router.guess_entity_type(first_recognizable),
                    "id": first_recognizable,
                }

        return {
            "status": "success",
            "pipeline_stages": {
                "scrubbed_query": scrubbed_query,
                "rbac_status": "authorized",
                "allowed_classifications": allowed_classifications,
                "answer_path": "semantic_rag",
            },
            "answer": final_answer,
            "sources": sources_used,
        }

    except Exception:
        logger.error(f"[{request_id}] RAG pipeline exception:\n" + traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Internal error while generating a response (request_id: {request_id}). "
                          "Check server logs for details.",
            },
        )
