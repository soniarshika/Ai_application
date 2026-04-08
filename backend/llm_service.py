"""
OpenAI API integration for Q&A, structured extraction, and RAG query enhancement.
"""

import json
import logging
import os
import re
from typing import List, Dict

from openai import OpenAI

log = logging.getLogger("logistics.llm")


class LLMService:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-5")

    def _chat(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def answer_question(self, question: str, chunks: List[Dict], scores) -> str:
        """Answer a question grounded strictly in the provided chunks."""
        context_blocks = []
        for i, chunk in enumerate(chunks[:5], start=1):
            sim = float(scores[i - 1]) if i - 1 < len(scores) else 0.0
            context_blocks.append(
                f"[CHUNK {i} — Page {chunk['page_number']}, "
                f"Type: {chunk['chunk_type']}, Relevance: {sim:.2f}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        system = (
            "You are a logistics document analyst. "
            "Your ONLY job is to answer questions using the context excerpts provided. "
            "Rules:\n"
            "1. Answer ONLY from the provided context. Never use outside knowledge.\n"
            "2. If the answer is not present in the context, respond with exactly: "
            "'The answer is not available in the provided document.'\n"
            "3. Quote exact figures, names, and dates as they appear in the document.\n"
            "4. Be concise and direct. Do not speculate or infer beyond the text."
        )

        user = (
            f"Context excerpts (ordered by relevance):\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based strictly on the context above:"
        )

        return self._chat(system, user)

    # ------------------------------------------------------------------
    # HyDE — Hypothetical Document Embeddings
    # ------------------------------------------------------------------

    def generate_hyde(self, question: str) -> str:
        """
        Generate a hypothetical logistics document passage that would answer
        the question. This passage is then embedded and used as the query vector —
        it embeds much closer to the actual document text than the question alone.

        Example:
          Q: "What is the carrier rate?"
          HyDE: "The carrier rate is $1,250.00 per load, payable net 30 days."
          → This embeds close to "Rate: $1,250.00" in the KV block.
        """
        system = (
            "You are a logistics document writer. "
            "Given a question, write a short realistic passage (1-2 sentences) "
            "that looks like it was extracted from a logistics document "
            "(rate confirmation, bill of lading, or shipment order) and directly "
            "answers the question. Use logistics terminology. "
            "Write ONLY the passage — no preamble, no explanation."
        )
        user = f"Question: {question}\n\nWrite the logistics document passage:"
        try:
            result = self._chat(system, user, max_tokens=150)
            log.debug(f"  HyDE passage: {result[:100]}")
            return result
        except Exception as e:
            log.warning(f"  HyDE generation failed: {e} — using original query")
            return question

    # ------------------------------------------------------------------
    # Multi-Query Expansion
    # ------------------------------------------------------------------

    def expand_query(self, question: str) -> List[str]:
        """
        Generate two alternative phrasings of the question using logistics
        terminology variations.
        """
        system = (
            "You are a logistics search assistant. "
            "Given a question about a logistics document, return a JSON object with key "
            "'queries' containing an array of exactly 2 alternative phrasings. "
            "Use different logistics terminology — vary between industry terms, "
            "abbreviations, and plain language. "
            "Example input: 'Who is the carrier?'\n"
            "Example output: {\"queries\": [\"freight company name\", \"trucking company hauling shipment\"]}"
        )
        user = f"Question: {question}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_completion_tokens=150,
            )
            raw = json.loads(response.choices[0].message.content)
            queries = raw.get("queries", [])
            if isinstance(queries, list):
                result = [str(q) for q in queries[:2] if q]
                log.debug(f"  Expanded queries: {result}")
                return result
        except Exception as e:
            log.warning(f"  Query expansion failed: {e} — skipping")
        return []

    # ------------------------------------------------------------------
    # Structured Extraction
    # ------------------------------------------------------------------

    def extract_fields(self, text: str) -> dict:
        """Extract structured logistics fields using JSON mode."""

        schema_to_extract = json.dumps(
            {
                "shipment_id": None,
                "shipper": {"name": None, "address": None},
                "consignee": {"name": None, "address": None},
                "pickup_datetime": None,
                "delivery_datetime": None,
                "equipment_type": None,
                "mode": None,
                "rate": None,
                "currency": None,
                "weight": None,
                "carrier_name": None,
            },
            indent=2,
        )

        system = (
            "You are a logistics data extraction engine. "
            "Extract structured fields from document text and return ONLY valid JSON. "
            "No markdown code fences, no commentary, no extra keys. "
            "If a field is not found, use null. "
            "Dates must be ISO 8601 format (YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD). "
            "Currency must be ISO 4217 (e.g. USD, EUR, CAD). "
            "Rate and weight must be numbers only (no units)."
        )

        user = (
            f"Extract logistics fields from the document below. "
            f"Return this exact JSON schema with found values filled in:\n\n"
            f"{schema_to_extract}\n\n"
            f"Document text:\n{text}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
