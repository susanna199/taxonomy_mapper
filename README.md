## Taxonomy Mapper

Task is to classify user submitted content into one of 10 sub-genres if the description fits.

#### Project Directory Structure

taxonomy_mapper/

├── main.py                 # Entry point + orchestration

├── src/

    ├── llm_arbiter.py         # Stage 2: LLM classification + validation
    
    ├── preprocessor.py        # Stage 1: Honesty filter
    
    ├── inference_engine.py    # Stage 3: Pipeline orchestration
    
    ├── taxonomy_loader.json   # Loads sub-genres hierarchy

├── test_cases.json        # 10 golden test cases

├── taxonomy.json          # 10 sub-genres under 'Fiction'

├── outputs/

│   └── results.json       # Deliverable: Reasoning log

└── README.md              # This file


### Processing Pipeline (3-Stage)

#### Stage 1: Preprocessor (Honesty Filter)

The purpose of this stage is to detect if the text fits non-fiction so that it can be immediately mapped as 'Unmapped' without having to make API calls unnecessarily. 
It employs RegEx patterns to rule out sentences that are instruction-based or recipe-like instructions (since the test-cases had them).
For scalability purposes, this can be enhanced further.

#### Stage 2: LLMArbiter 

Model - llama-3.1-8b-instruct was used via Groq API

Parameters:
1. temperature: 0.1 (So that the responses are deterministic)
2. max_tokens: 220 (For the purpose of cost control)

This stage consists of a constrained prompt with instructions to classify the story into exactly one of the 10 sub-genres. 
This is done by emphasising on the context from the blurb, (over the tags given by the user), honesty so that when the model is unsure, it will map it as 'Unmapped' and hierarchy to assess which sub-genres of each parent genre would be more suitable (when scaled) instead of merely classifying it into one of many sub-genres.

#### Stage 3: InferenceEngine

This stage returns the json object with the reasoning log and the detailed result.
Ensuring that if the model happened to generate a new category, it is filtered to 'Unmapped' to avoid hallucination.

### Results Summary

- MAPPED: 8/10 (80%)
- UNMAPPED: 2/10 (20% non-fiction)  


