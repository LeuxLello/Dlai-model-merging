# AI-use statement (working record)

This record is deliberately specific and will be condensed for the official report. OpenAI Codex has had a substantial role in formulating and refining the research question, designing the experiment sequence, implementing the core Python modules and Kaggle notebooks, debugging, organizing outputs, analysing preliminary results, locating literature, and drafting documentation. The student directed the project through iterative requests, configured and executed the Kaggle runs, supplied the resulting artifacts, discussed the interpretation and decided to continue or stop each experimental branch.

The final scientific responsibility cannot be delegated: before submission, the student will personally read the cited sections, inspect the code and outputs, verify every reported value and claim, rewrite the report in their own informed voice, and be able to explain the complete pipeline. Items not yet personally verified must not be described as independently checked. This wording follows the course template's requirement for an honest, sufficiently specific disclosure; extensive AI use is permitted, while submitting material the student does not understand is not.

## Running log

- 2026-08-19: Codex helped analyze the project guidelines, formulate the research question, define the initial experiment design, and scaffold the repository.
- 2026-08-19 to 2026-08-24: Codex implemented the training, merging, evaluation, diagnostic and ablation code; created notebooks 01-08; organized user-executed Kaggle outputs; proposed and analysed confirmatory, improvement and error-analysis phases; and maintained repository documentation. The student ran the notebooks on Kaggle, returned the outputs and made the project-level decisions through the conversation.
- 2026-08-24: Codex audited the literature-to-code mapping, identified that the current TIES trim is tensor-wise rather than the official global flattening, and drafted a study guide and corrective roadmap. This issue remains to be resolved experimentally.
- 2026-08-24: Codex implemented global TIES, separated data-subset and training seeds, added unit tests, and created the final corrective notebook 09. The student will execute the notebook on Kaggle and inspect the resulting comparison.
