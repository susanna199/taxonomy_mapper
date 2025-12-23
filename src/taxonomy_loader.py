# src/taxonomy_loader.py
import json
from typing import Dict, Any, List


class TaxonomyLoader:
    """
    Loads the internal taxonomy and exposes:
      - the raw nested structure
      - a flat list of allowed leaf categories (sub-genres)
    """

    def __init__(self, taxonomy_path: str):
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy()
        self.leaf_labels = self._extract_leaf_labels()

    def _load_taxonomy(self) -> Dict[str, Any]:
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_leaf_labels(self) -> List[str]:
        """
        From:
          {
            "Fiction": {
              "Romance": ["Slow-burn", "Enemies-to-Lovers", ...],
              ...
            }
          }
        produce:
          ["Slow-burn", "Enemies-to-Lovers", ..., "Slasher"]
        """
        leaves: List[str] = []
        for top_level, subtrees in self.taxonomy.items():
            if isinstance(subtrees, dict):
                for _, subgenres in subtrees.items():
                    if isinstance(subgenres, list):
                        leaves.extend(subgenres)
        return leaves

    def get_taxonomy(self) -> Dict[str, Any]:
        return self.taxonomy

    def get_leaf_labels(self) -> List[str]:
        return self.leaf_labels
