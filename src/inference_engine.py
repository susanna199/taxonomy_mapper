# src/inference_engine.py
from typing import Dict, Any

from .taxonomy_loader import TaxonomyLoader
from .preprocessor import Preprocessor
from .llm_arbiter import LLMArbiter


class InferenceEngine:
    """
    High-level orchestrator for the Adaptive Taxonomy Mapper.

    Flow for each story:
      1. Honesty fast-path: detect obvious non-fiction / instructional blurbs → [UNMAPPED]
      2. Build a rich context string for the LLM (Context Wins)
      3. Call LLM classifier constrained to allowed sub-genres (Hierarchy)
      4. Return category + reasoning + status
    """

    def __init__(self, taxonomy_path: str):
        # Load taxonomy and allowed sub-genres
        self.taxonomy_loader = TaxonomyLoader(taxonomy_path)
        allowed_labels = self.taxonomy_loader.get_leaf_labels()

        # Core components
        self.preprocessor = Preprocessor()
        self.llm = LLMArbiter(allowed_labels=allowed_labels)

    def map_story(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for a single story.

        Input:
          {
            "id": int,
            "user_tags": [...],
            "blurb": "..."
          }

        Output:
          {
            "category": str,
            "reasoning": str,
            "status": "MAPPED" | "UNMAPPED"
          }
        """
        blurb = story.get("blurb", "")

        # 1. Honesty Rule: clear instructional / non-fiction → [UNMAPPED]
        if self.preprocessor.is_obviously_nonfiction(blurb):
            return {
                "category": "[UNMAPPED]",
                "reasoning": "Blurb appears to be instructional/non-fiction, so no honest match in the fiction taxonomy.",
                "status": "UNMAPPED",
            }

        # 2. Build LLM context (Context Wins)
        context_text = self.preprocessor.build_context(story)

        # 3. LLM classification with strict output control
        result = self.llm.classify(context_text)

        status = "MAPPED" if result["category"] != "[UNMAPPED]" else "UNMAPPED"

        return {
            "category": result["category"],
            "reasoning": result["reasoning"],
            "status": status,
        }
