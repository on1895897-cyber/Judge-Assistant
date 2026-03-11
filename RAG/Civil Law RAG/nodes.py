"""
nodes.py

This file contains the core LangGraph nodes for the Judge Assistant RAG system.
Each node handles a specific responsibility in the query → retrieval → answer workflow.

Nodes included:
- Preprocessor: rewrites and classifies queries
- Off-Topic: handles out-of-scope questions
- Textual: retrieves specific law articles by number/range
- Retriever: performs semantic retrieval of relevant documents
- Rule Grader: checks retrieval quality and decides refinement
- Refine: improves queries if retrieval failed
- LLM Grader: evaluates retrieved docs using the LLM
- Generate Answer: produces final analytical or textual answer
- Cannot Answer: fallback when no answer can be generated
"""

from typing import TypedDict, Optional, List
import re
import json
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from config import LLM_MODEL, EMBEDDING_MODEL, DB_DIR
from langsmith import traceable
from prompts import (
    PREPROCESSOR_PROMPT, 
    UNIFIED_REFINE_PROMPT,
    LLM_GRADER_PROMPT,
    ANALYTICAL_PROMPT
)
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

llm = ChatGroq(model=LLM_MODEL)

database = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)
# ------------------------
# State definition
# ------------------------
class State(TypedDict):
    last_query: Optional[str]
    last_results: List[Document]
    last_answer: Optional[str]
    current_book: Optional[str]
    current_part: Optional[str]
    current_chapter: Optional[str]
    current_article: Optional[int]
    filter_type: str
    k: int
    books_in_scope: List[str]
    query_history: List[str]
    retrieval_history: List[List[Document]]
    answer_history: List[str]
    db_initialized: bool
    db: Optional[Chroma]
    split_config: dict
    rewritten_question: Optional[str]
    classification: Optional[str]  # analytical | textual | off_topic
    retrieval_confidence: Optional[float]
    retry_count: int
    max_retries: int 
    refined_query: Optional[str]
    grade: Optional[str]  # pass | refine | fail
    llm_pass: Optional[bool]
    failure_reason: Optional[str]
    proceedToGenerate: Optional[bool]
    retrieval_attempts: int
    final_answer: str | None

# ------------------------
# Node implementations
# ------------------------

def strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences that LLMs sometimes wrap around JSON."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def fast_filters(query: str) -> str | None:
    """Fast rule-based filtering for off-topic queries."""
    query = query.strip()
    if len(query) < 5 or not re.search(r"[\u0600-\u06FF]", query):
        return "off_topic"
    return None


# Preprocessor Node
@traceable(name="Preprocessor Node")
def preprocessor_node(state: State) -> State:
    """
    Preprocesses the user's query by rewriting and classifying it.
    """
    query = state.get("last_query", "")
    if not query:
        state["classification"] = "off_topic"
        state["rewritten_question"] = None
        return state

    # 1. Rule-based filters
    fast_result = fast_filters(query)
    if fast_result == "off_topic":
        state["rewritten_question"] = None
        state["classification"] = "off_topic"
        return state

    # 2. LLM call
    prompt = PREPROCESSOR_PROMPT.format(question=query)
    response = llm.invoke(prompt)
    content = response.content.strip()

    # 3. Parse JSON (strip markdown code fences that LLMs may add)
    content = strip_code_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        state["rewritten_question"] = query
        state["classification"] = "analytical"
        return state

    # 4. Normalize labels
    classification_map = {
        "تحليلي": "analytical",
        "نصّي": "textual",
        "خارج السياق": "off_topic"
    }

    state["rewritten_question"] = data.get("rewritten_question", query)
    state["classification"] = classification_map.get(
        data.get("classification"),
        "analytical"
    )

    # 5. Update history
    state["query_history"].append(query)

    return state


# Off-Topic Node
@traceable(name="Off-Topic Node")
def off_topic_node(state: State) -> State:
    state["final_answer"] = (
        "يبدو أن السؤال المطروح خارج نطاق اختصاص هذا النظام. "
        "يرجى طرح سؤال متعلق بالقانون المدني أو الأحكام القانونية ذات الصلة."
    )
    return state


# Textual Node
@traceable(name="Textual Node")
def textual_node(state: dict) -> dict:
    """
    Handles textual/legal questions that can be answered by
    directly retrieving the relevant civil law articles.

    Assumes the preprocessor has normalized:
    - Numbers in words → digits
    - Intervals like 'بين X و Y' → digits
    """
    db = database
    query = state.get("rewritten_question") or state["last_query"]

    # Check for interval query: "بين X و Y"
    range_match = re.search(r"بين\s*(\d+)\s*و\s*(\d+)", query)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        docs: List[Document] = []
        for num in range(start, end + 1):
            docs += db.similarity_search(
                f"المادة {num}",
                k=1,
                filter={
                    "$and": [
                        {"type": {"$eq": "article"}},
                        {"index": {"$eq": num}}
                    ]
                }
            )
        state["current_article"] = f"{start}-{end}"
        state["last_results"] = docs
        state["final_answer"] = "\n\n".join([doc.page_content for doc in docs]) if docs else \
            "عذرًا، لم أتمكن من العثور على نص المواد المطلوبة."
        return state

    # Check for exact article number
    article_match = re.search(r"المادة\s*(\d+)", query)
    if article_match:
        article_num = int(article_match.group(1))
        state["current_article"] = article_num
        docs: List[Document] = db.similarity_search(
            f"المادة {article_num}",
            k=1,
            filter={
                "$and": [
                    {"type": {"$eq": "article"}},
                    {"index": {"$eq": article_num}}
                ]
            }
        )
        state["last_results"] = docs
        state["final_answer"] = "\n\n".join([doc.page_content for doc in docs]) if docs else \
            "عذرًا، لم أتمكن من العثور على نص المادة المطلوب."
        return state

    # Fallback: semantic search (no number or vague query)
    docs: List[Document] = db.similarity_search(
        query,
        k=3,
        filter={"type": {"$eq": "article"}}
    )
    state["last_results"] = docs
    state["final_answer"] = "\n\n".join([doc.page_content for doc in docs]) if docs else \
        "عذرًا، لم أتمكن من العثور على نص المادة المطلوب."

    return state


# Retriever Node
@traceable(name="Retriever Node")
def retrieve_node(state: dict, k: int = 5) -> dict:
    """
    Retrieve relevant articles from the Egyptian Civil Law corpus
    based on the rewritten question. Stores results in state for grading.

    Inputs:
        - state: current RAG state
        - db: Chroma vector store
        - k: number of articles to retrieve
    """
    db = database  # use the module-level Chroma instance (Bug 2 fix)
    query = state.get("rewritten_question") or state.get("last_query")

    # 1. Retrieve top-k relevant articles with relevance scores (Bug 3 fix)
    results_with_scores = db.similarity_search_with_relevance_scores(
        query, k=k, filter={"type": "article"}
    )

    if not results_with_scores:
        state["last_results"] = []
        state["retrieval_confidence"] = 0.0
        return state

    # 2. Unpack documents and actual similarity scores
    last_results = [doc for doc, _ in results_with_scores]
    similarities = [score for _, score in results_with_scores]
    avg_confidence = sum(similarities) / len(similarities)

    # 3. Update state
    state["last_results"] = last_results
    state["retrieval_confidence"] = round(avg_confidence, 2)  # 0-1 scale

    return state


# Rule Grader Node
@traceable(name="Rule Grader Node")
def rule_grader_node(state: dict, min_docs: int = 1, min_confidence: float = 0.4) -> dict:
    """
    Grades the quality of retrieved documents and decides
    whether refinement is needed.

    Outputs:
        - state["grade"]: "pass" | "refine" | "fail"
    """
    if state["retry_count"] >= state["max_retries"]:
        state["grade"] = "fail"
        state["failure_reason"] = "تم تجاوز الحد الأقصى لمحاولات تحسين الاستعلام دون العثور على نتائج كافية."
        return state

    docs = state.get("last_results", [])
    confidence = state.get("retrieval_confidence", 0.0)
    query = state.get("rewritten_question") or state.get("last_query")

    # Case 1: No documents at all
    if not docs:
        state["grade"] = "fail"
        state["failure_reason"] = "لم يتم العثور على مواد قانونية."
        return state

    # Case 2: Too few documents
    if len(docs) < min_docs:
        state["grade"] = "refine"
        return state

    # Case 3: Low confidence score
    if confidence < min_confidence:
        state["grade"] = "refine"
        return state

    # If everything looks good
    state["grade"] = "pass"
    return state


# Refine Node
@traceable(name="Refine Node")
def refine_node(state: dict) -> dict:
    state["retry_count"] += 1

    query = state.get("refined_query") or state.get("rewritten_question") or state["last_query"]
    reason = state.get("failure_reason")

    reason_block = f"سبب فشل البحث السابق:\n{reason}" if reason else ""

    prompt = UNIFIED_REFINE_PROMPT.format(
        query=query,
        reason_block=reason_block
    )

    response = llm.invoke(prompt)

    try:
        data = json.loads(strip_code_fences(response.content))
        state["refined_query"] = data["refined_query"]
    except Exception:
        state["refined_query"] = query

    return state


# LLM Grader Node
@traceable(name="LLM Grader Node")
def llm_grader_node(state: dict) -> dict:
    query = (
        state.get("refined_query")
        or state.get("rewritten_question")
        or state["last_query"]
    )
    docs = state.get("last_results", [])

    docs_text = "\n\n".join(
        f"المادة {d.metadata.get('article_number')}:\n{d.page_content}"
        for d in docs
    )

    prompt = LLM_GRADER_PROMPT.format(
        query=query,
        docs=docs_text
    )

    response = llm.invoke(prompt)

    try:
        result = json.loads(strip_code_fences(response.content))
        state["llm_pass"] = result["pass"]
        state["failure_reason"] = result.get("reason", "")
    except Exception:
        state["llm_pass"] = False
        state["failure_reason"] = "فشل تحليل المستندات بسبب خطأ في تنسيق الرد."

    return state


# Generate Answer Node
@traceable(name="Generate Answer Node")
def generate_answer_node(state: dict) -> dict:
    query = state.get("refined_query") or state.get("rewritten_question") or state["last_query"]
    docs = state.get("last_results", [])

    if not docs:
        state["final_answer"] = "لم يتم العثور على مواد قانونية ذات صلة للإجابة على السؤال."
        return state

    context_text = "\n\n".join(
        f"(المادة {d.metadata.get('article_number', 'غير معروفة')})\n{d.page_content}"
        for d in docs
    )

    prompt = ANALYTICAL_PROMPT.format(
        context_text=context_text,
        query=query
    )

    response = llm.invoke(prompt)

    state["final_answer"] = response.content
    return state


# Cannot Answer Node
@traceable(name="Cannot Answer Node")
def cannot_answer_node(state: dict) -> dict:
    reason = state.get("failure_reason", "تعذر العثور على مواد قانونية مناسبة.")

    state["final_answer"] = f"""
    تعذر تقديم إجابة قانونية دقيقة على السؤال المطروح.

    السبب:
    {reason}

    يرجى إعادة صياغة السؤال أو توضيح الوقائع بشكل أدق.
    """
    return state