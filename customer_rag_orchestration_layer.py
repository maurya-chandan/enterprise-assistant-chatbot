import os
# --- PREVENT WINDOWS C++ SEGMENTATION FAULTS & THREAD COLLISIONS ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db_customer_service"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_rag_components():
    """
    Loads the expensive, shared pieces of the RAG stack ONCE: embeddings,
    the persisted vector store, and the LLM client. None of these know "who
    is asking" - that's applied per-request in build_scoped_chain() below -
    so a single cached instance is safely reused across requests from
    callers with different permissions.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    return vectorstore, llm


def build_scoped_chain(vectorstore, llm, allowed_classifications):
    """
    Builds a retrieval chain whose retriever is restricted, AT THE VECTOR
    SEARCH LEVEL, to chunks tagged with one of `allowed_classifications`.

    In this project there is only one real tier - "customer_service" - so in
    practice allowed_classifications is either ["customer_service"] (admin
    and customer_service_agent) or [] (everyone else). An empty list is not
    a special case here: the $in filter with an empty list simply matches
    nothing, so the retriever returns zero documents and the LLM correctly
    reports it has no information, rather than falling back to unfiltered
    search. Documents ingested without a matching classification tag are
    likewise excluded - fail closed, not fail open.
    """
    allowed_classifications = allowed_classifications or []

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"classification": {"$in": allowed_classifications}},
        }
    )

    system_prompt = (
        "You are an internal Customer Service Assistant. Use the following "
        "retrieved customer, order, product, and shipment records to answer "
        "the agent's question accurately, citing order IDs, customer IDs, or "
        "tracking numbers where relevant.\n"
        "If the answer is not contained within the context, say you do not "
        "have that record rather than guessing. Do not hallucinate order "
        "details, prices, or tracking information.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
