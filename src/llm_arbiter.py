import json
import os
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMArbiter:
    """
    Groq LLM-based classifier using direct HTTP calls (no groq SDK).

    - Reads GROQ_API_KEY from env / .env
    - Sends a chat.completions request to llama-3.1-8b-instant
    - Enforces:
        * Context Wins
        * Honesty ([UNMAPPED] for no/invalid fit)
        * Hierarchy (only allowed sub-genres)
    """

    def __init__(self, allowed_labels: List[str]):
        self.allowed_labels = sorted(set(allowed_labels))
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment or .env file")

        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model_name = "llama-3.1-8b-instant"
        self.token_log = []
        # Add this SUMMARY method to LLMArbiter
    def print_usage_summary(self):
        if not self.token_log:
            print("No usage data yet.")
            return
        
        total_in = sum(log['input_tokens'] for log in self.token_log)
        total_out = sum(log['output_tokens'] for log in self.token_log)
        total_all = sum(log['total_tokens'] for log in self.token_log)
        total_cost = sum(log['cost'] for log in self.token_log)
        
        avg_in, avg_out, avg_total = total_in/len(self.token_log), total_out/len(self.token_log), total_all/len(self.token_log)
        
        print("\n" + "="*60)
        print("📊 TOKEN USAGE SUMMARY (from your pipeline)")
        print("="*60)
        print(f"Stories processed: {len(self.token_log)}")
        print(f"Total tokens: {total_in:,} in + {total_out:,} out = {total_all:,} total")
        print(f"Avg per story: {avg_in:.0f} in + {avg_out:.0f} out = {avg_total:.0f} total")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"1M stories cost: ${(avg_total * 1e6 / 1e6 * 59):.0f}/month")  # Scaled
        print(f"Free tier limit: ~7K calls/month → {len(self.token_log)*1000:.0f} calls needed")
        print("="*60)

    # ---------- Prompt design ----------

    def _build_prompt(self, context_text: str) -> str:
        labels_str = ", ".join(self.allowed_labels)

        return f"""
You are an expert FICTION taxonomy classifier.

You must map each story to EXACTLY ONE of the following sub-genres,
or return [UNMAPPED] if there is no honest fit.

ALLOWED SUB-GENRES:
{labels_str}

RULES:
1. Context Wins: Use the STORY BLURB as the primary signal. User tags are noisy hints and may be misleading.
2. Honesty: If the story is instructional, non-fiction, or does not clearly match any sub-genre, output [UNMAPPED].
3. Hierarchy: You are only allowed to choose from the sub-genres listed above, or [UNMAPPED].
4. Output control: Respond strictly in JSON format.

STORY CONTEXT:
{context_text}

Respond with JSON ONLY in this exact format:
{{
  "category": "<one sub-genre from the list above or [UNMAPPED]>",
  "reasoning": "1–3 sentences explaining your choice, quoting key phrases from the blurb."
}}
"""

    # ---------- Public API ----------

    def classify(self, context_text: str) -> Dict[str, str]:
        """
        Call Groq LLM and return a validated result:
          { "category": str, "reasoning": str }
        """
        prompt = self._build_prompt(context_text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            # Cost-aware settings
            "temperature": 0.1,
            "max_tokens": 220,
            "n": 1,
        }

        resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            # Network / API failure → fall back to honest [UNMAPPED]
            return {
                "category": "[UNMAPPED]",
                "reasoning": f"Groq API error {resp.status_code}: {resp.text[:200]}",
            }

        data = resp.json()
        try:
            raw_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {
                "category": "[UNMAPPED]",
                "reasoning": "Unexpected response format from Groq API; marking as [UNMAPPED].",
            }

        parsed = self._safe_parse_json(raw_text)

        category = parsed.get("category", "[UNMAPPED]")
        reasoning = (parsed.get("reasoning") or "").strip()

        # ---------- Output control & hallucination guard ----------
        if category not in self.allowed_labels and category != "[UNMAPPED]":
            category = "[UNMAPPED]"
            extra = (
                " Model proposed a label outside the taxonomy; coerced to [UNMAPPED] "
                "to respect the Hierarchy rule."
            )
            reasoning = (reasoning + extra) if reasoning else extra

        if not reasoning:
            reasoning = "Model did not provide a reasoning string; defaulted to a generic explanation."
        usage_data = data.get("usage")  # 👈 data = resp.json()
        if usage_data:  # Check exists before accessing
            self.token_log.append({
                'input_tokens': usage_data['prompt_tokens'],
                'output_tokens': usage_data['completion_tokens'],
                'total_tokens': usage_data['total_tokens'],
                'cost': (usage_data['prompt_tokens'] * 0.00005 + 
                        usage_data['completion_tokens'] * 0.00008)
            })
            print(f"📊 Tokens: {usage_data['prompt_tokens']}in + {usage_data['completion_tokens']}out = {usage_data['total_tokens']}total | 💰 ${self.token_log[-1]['cost']:.6f}")
        else:
            print("⚠️ No usage data in response")  # Rare edge case [web:52]
        return {"category": category, "reasoning": reasoning}

    # ---------- Helpers ----------

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """
        Extract and parse the first JSON object in the LLM output.
        Fails safe to [UNMAPPED] if parsing fails.
        """
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON object found in LLM response")
            snippet = text[start:end]
            return json.loads(snippet)
        except Exception:
            return {
                "category": "[UNMAPPED]",
                "reasoning": "Failed to parse JSON from model output; marking as [UNMAPPED] per Honesty rule."
            }
