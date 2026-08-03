import streamlit as st
import requests
import hmac
import hashlib
import os

st.set_page_config(page_title="Customer Service Assistant", page_icon="🎧", layout="wide")
st.title("🎧 Customer Service Assistant")
st.markdown("Ask about orders, customers, products, stock, or shipment tracking. Restricted to `admin` and `customer_service_agent` roles.")

# NOTE: this sidebar is a MOCK identity provider for local development only.
# A real deployment must replace this with a proper SSO/OIDC login flow -
# anyone who can read this app's source can also read MOCK_IDP_SECRET, so
# this is a convenience check, not real authentication.
MOCK_IDP_SECRET = os.environ.get("MOCK_IDP_SECRET", "dev-only-insecure-shared-secret")

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8001/ask")


def sign_role(role: str) -> str:
    signature = hmac.new(MOCK_IDP_SECRET.encode(), role.encode(), hashlib.sha256).hexdigest()
    return f"{role}.{signature}"


st.sidebar.header("Identity Provider (Mock)")
user_id = st.sidebar.text_input("User ID", "agent_204")
# "guest" is included on purpose - pick it to see the RBAC denial path work.
user_role = st.sidebar.selectbox("Role", ["admin", "customer_service_agent", "guest"])
st.sidebar.caption(f"Gateway: {GATEWAY_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("e.g. What's the status of order ORD100002?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    headers = {"x-auth-token": sign_role(user_role)}
    payload = {"user_id": user_id, "query": prompt}

    try:
        with st.spinner("Gateway checking permissions and invoking local LLM..."):
            response = requests.post(GATEWAY_URL, json=payload, headers=headers, timeout=180)

        try:
            data = response.json()
        except ValueError:
            with st.chat_message("assistant"):
                st.error(
                    f"⛔ Gateway returned a non-JSON response (HTTP {response.status_code}). "
                    "Check that the API Gateway is running correctly."
                )
            st.stop()

        if response.status_code == 200:
            answer = data.get("answer")
            stages = data.get("pipeline_stages", {})
            sources = data.get("sources", [])

            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("🔍 View Gateway & RAG Details"):
                    st.write(f"**Scrubbed Query:** `{stages.get('scrubbed_query')}`")
                    st.write(f"**RBAC Check:** ✅ {stages.get('rbac_status')}")
                    st.write(f"**Allowed Classifications:** {stages.get('allowed_classifications')}")
                    answer_path = stages.get("answer_path", "unknown")
                    path_label = (
                        "🎯 Structured lookup (exact match, no LLM involved)"
                        if answer_path == "structured_lookup"
                        else "🧠 Semantic RAG (embedding search + LLM)"
                    )
                    st.write(f"**Answer Path:** {path_label}")
                    if sources:
                        st.write(f"**Records Cited:** {', '.join(sources)}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        else:
            error_detail = data.get("detail", "Unknown Error")
            with st.chat_message("assistant"):
                st.error(f"⛔ Gateway Blocked Request: {error_detail}")
            st.session_state.messages.append({"role": "assistant", "content": f"⛔ Blocked: {error_detail}"})

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API Gateway. Ensure it's running at {GATEWAY_URL}.")
    except requests.exceptions.ReadTimeout:
        st.error("Gateway request timed out while waiting for local LLM generation.")
