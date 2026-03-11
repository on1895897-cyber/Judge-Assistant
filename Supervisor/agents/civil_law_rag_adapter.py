"""
civil_law_rag_adapter.py

Adapter for the Civil Law RAG agent (RAG/Civil Law RAG/graph.py).

Wraps the compiled ``app`` graph and returns an AgentResult with
the final answer about Egyptian civil law provisions.
"""

import logging
import os
import sys
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


class CivilLawRAGAdapter(AgentAdapter):
    """Thin wrapper around the Civil Law RAG LangGraph workflow."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Query the Civil Law RAG for relevant legal provisions.

        Parameters
        ----------
        query:
            The (rewritten) judge query about civil law articles or
            legal provisions.
        context:
            Optional. May contain prior conversation or case context.
        """
        try:
            # Add Civil Law RAG directory to path
            rag_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "RAG", "Civil Law RAG",
            )
            rag_dir = os.path.normpath(rag_dir)
            if rag_dir not in sys.path:
                sys.path.insert(0, rag_dir)

            from dotenv import load_dotenv
            load_dotenv()

            from graph import app
            from nodes import database
            from config import default_state_template

            # Build initial state from the template
            state = dict(default_state_template)
            state["last_query"] = query
            # Bug 2 fix: default_state_template has db=None which causes
            # retrieve_node to crash. Inject the module-level database.
            if state.get("db") is None:
                state["db"] = database

            result = app.invoke(state)

            final_answer = result.get("final_answer", "")
            last_results = result.get("last_results", [])

            # Extract source references from retrieved documents
            sources = []
            for doc in last_results:
                if hasattr(doc, "metadata"):
                    meta = doc.metadata
                    ref = meta.get("article", meta.get("source", ""))
                    if ref:
                        sources.append(str(ref))

            return AgentResult(
                response=final_answer or "",
                sources=sources,
                raw_output={
                    "final_answer": final_answer,
                    "classification": result.get("classification"),
                    "retrieval_confidence": result.get("retrieval_confidence"),
                },
            )

        except Exception as exc:
            error_msg = f"Civil Law RAG adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
