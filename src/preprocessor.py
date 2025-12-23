# src/preprocessor.py
from typing import Dict, Any, List
import re


class Preprocessor:
    """
    Preprocesses raw user input into:
      - a context string tailored for the LLM
      - a lightweight non-fiction / instructional flag (Honesty rule)
    """

    # Simple patterns for clearly non-fiction / instructional content
    NONFICTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"\bhow to\b", re.IGNORECASE),
        re.compile(r"\bstep[- ]by[- ]step\b", re.IGNORECASE),
        re.compile(r"\bingredients?\b", re.IGNORECASE),
        re.compile(r"\brecipe\b", re.IGNORECASE),
        re.compile(r"\bprep time\b", re.IGNORECASE),
        re.compile(r"\bbake at\b", re.IGNORECASE),
        re.compile(r"\bmix\b", re.IGNORECASE),
        re.compile(r"\badd\b", re.IGNORECASE),
        re.compile(r"\bdegrees\b", re.IGNORECASE),
    ]

    def is_obviously_nonfiction(self, blurb: str) -> bool:
        """
        Honesty rule fast-path:
        If the blurb clearly looks like a recipe/instructional text,
        mark it as non-fiction so we can safely output [UNMAPPED].
        """
        text = blurb.strip()
        for pattern in self.NONFICTION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def build_context(self, story: Dict[str, Any]) -> str:
        """
        Build the exact context string that will be passed to the LLM.

        Emphasizes:
          - This is a fiction story
          - The blurb is the primary signal (Context Wins)
          - Tags are secondary, possibly noisy
        """
        blurb = story.get("blurb", "").strip()
        tags_list = story.get("user_tags", [])
        tags_text = ", ".join(tags_list) if tags_list else "None"

        context = (
            "You are classifying a FICTION story into a precise sub-genre.\n"
            "Base your decision mainly on the STORY BLURB. "
            "User tags are noisy hints and must not override the blurb.\n\n"
            f"STORY BLURB:\n{blurb}\n\n"
            f"USER TAGS (noisy hints): {tags_text}\n"
        )

        return context
