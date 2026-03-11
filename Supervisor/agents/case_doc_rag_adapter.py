"""
case_doc_rag_adapter.py

Adapter for the Case Document RAG agent (RAG/Case Doc RAG/rag_docs.py).

Wraps the compiled ``app`` graph and returns an AgentResult with the
answer extracted from case documents.
"""

import logging
import os
import sys
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


def _get_shared_vectorstore():
    """Return the shared Chroma vector store used by the FileIngestor.

    This ensures the Case Doc RAG reads from the *same* store that
    documents were indexed into, avoiding the empty-store problem that
    occurs when ``rag_docs.py`` creates its own in-memory Chroma
    instance at import time.
    """
    from Supervisor.nodes.classify_and_store_document import _get_ingestor
    ingestor = _get_ingestor()
    return ingestor.vectorstore


class CaseDocRAGAdapter(AgentAdapter):
    """Thin wrapper around the Case Doc RAG LangGraph workflow."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Query the Case Doc RAG about specific case documents.

        Parameters
        ----------
        query:
            The (rewritten) judge query about case documents.
        context:
            Should contain ``case_id``.  May also contain
            ``conversation_history`` for multi-turn context.
        """
        try:
            # Add Case Doc RAG directory to path
            rag_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "RAG", "Case Doc RAG",
            )
            rag_dir = os.path.normpath(rag_dir)
            if rag_dir not in sys.path:
                sys.path.insert(0, rag_dir)

            from dotenv import load_dotenv
            load_dotenv()

            import rag_docs
            from rag_docs import app

            # Inject the shared vector store so rag_docs queries the
            # same Chroma instance that documents were indexed into.
            try:
                shared_vs = _get_shared_vectorstore()
                rag_docs.set_vectorstore(shared_vs)
                logger.info("Injected shared vectorstore into rag_docs")
            except Exception as exc:
                logger.warning(
                    "Could not inject shared vectorstore: %s. "
                    "rag_docs will use its own Chroma instance.",
                    exc,
                )

            case_id = context.get("case_id", "")

            # Build message list from conversation history
            messages = []
            for turn in context.get("conversation_history", []):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))

            # The current query as a HumanMessage
            human_msg = HumanMessage(content=query)
            messages.append(human_msg)

            initial_state = {
                "query": human_msg,
                "messages": messages,
                "case_id": case_id,
                "doc_type": None,
                "retrieved_docs": [],
                "context": "",
                "refined_query": "",
                "safety_notes": [],
                "answer": "",
                "onTopic": True,
                "proceedToGenerate": False,
                "rephraseCount": 0,
                "doc_selection_mode": "",
                "selected_doc_id": None,
            }

            result = app.invoke(initial_state)

            # Extract the answer from the last AI message
            answer = result.get("answer", "")
            if not answer:
                result_messages = result.get("messages", [])
                for msg in reversed(result_messages):
                    if hasattr(msg, "content") and hasattr(msg, "type"):
                        if msg.type == "ai":
                            answer = msg.content
                            break

            # Extract sources from retrieved docs
            sources = []
            for doc in result.get("retrieved_docs", []):
                if isinstance(doc, dict):
                    title = doc.get("title", doc.get("doc_id", ""))
                    if title:
                        sources.append(str(title))

            return AgentResult(
                response=answer,
                sources=sources,
                raw_output={
                    "answer": answer,
                    "doc_selection_mode": result.get("doc_selection_mode"),
                    "selected_doc_id": result.get("selected_doc_id"),
                },
            )

        except Exception as exc:
            error_msg = f"Case Doc RAG adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
