#!/usr/bin/env python3
"""
Adaptive Taxonomy Mapper - Entry Point

- Loads taxonomy.json and test_cases.json
- Runs each case through the InferenceEngine
- Prints per-case results and saves a JSON reasoning log
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from src.inference_engine import InferenceEngine


TAXONOMY_PATH = "taxonomy.json"
TEST_CASES_PATH = "test_cases.json"
OUTPUT_PATH = Path("outputs/results.json")


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print("🔥 Adaptive Taxonomy Mapper")
    print("==========================\n")

    # 1. Initialize inference engine
    print("Initializing Inference Engine with taxonomy.json ...")
    engine = InferenceEngine(TAXONOMY_PATH)
    print("✅ Engine ready.\n")
    

    # 2. Load test cases
    test_cases = load_test_cases(TEST_CASES_PATH)
    print(f"Loaded {len(test_cases)} test cases from {TEST_CASES_PATH}.\n")

    results: List[Dict[str, Any]] = []

    # 3. Process each case
    for case in test_cases:
        case_id = case.get("id")
        print(f"--- Case {case_id} ---")
        print(f"User tags: {case.get('user_tags')}")
        print(f"Blurb: {case.get('blurb')}\n")

        result = engine.map_story(case)

        print(f"Mapped category: {result['category']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Status: {result['status']}\n")

        results.append(
            {
                "id": case_id,
                "user_tags": case.get("user_tags"),
                "blurb": case.get("blurb"),
                "category": result["category"],
                "reasoning": result["reasoning"],
                "status": result["status"],
            }
        )

    # 4. Save reasoning log
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 5. Simple summary
    mapped = sum(1 for r in results if r["status"] == "MAPPED")
    unmapped = len(results) - mapped
    print("==========================")
    print(f"Summary: MAPPED={mapped}, UNMAPPED={unmapped}")
    print(f"Detailed results saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
    
