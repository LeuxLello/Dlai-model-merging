# Scientific references

The repository does not duplicate paper PDFs. The links below point to authoritative sources, and
`references.bib` contains the report citations.

## Model merging

### Task Arithmetic

- Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023.
- Paper: https://arxiv.org/pdf/2212.04089
- Relevant sections: Section 2 for task vectors and scaling; Section 4 for composition; Table 3 for
  GLUE experiments.
- Local implementation: `task_vectors()` and `task_arithmetic()` in
  `src/dlai_merge/merging.py`.

### TIES-Merging

- Yadav et al., *TIES-Merging: Resolving Interference When Merging Models*, NeurIPS 2023.
- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf
- Reference code: https://github.com/prateeky2806/ties-merging
- Relevant sections: Figure 1, Section 4, Algorithm 1, and Section 7.3.
- Local implementation: `ties_merge()` applies tensor-wise trimming;
  `global_ties_merge()` implements the flatten-first global scope used by the reference code.

### Model Soups

- Wortsman et al., *Model soups: averaging weights of multiple fine-tuned models improves accuracy
  without increasing inference time*, ICML 2022.
- Paper: https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf
- Relevant section: Section 2 and the uniform-soup formula. This is background for weight averaging;
  unlike this project, Model Soups combines variants fine-tuned on the same task.

## Base model

- Devlin et al., *BERT*, NAACL 2019: https://aclanthology.org/N19-1423/
- Turc et al., *Well-Read Students Learn Better*, 2019: https://arxiv.org/abs/1908.08962
- Model card: https://huggingface.co/prajjwal1/bert-mini

The model card documents the four-layer, 256-hidden-size BERT-mini checkpoint used by the
experiments and configured in `configs/base.yaml`.

## Datasets

### GLUE: SST-2, MRPC, RTE, and CoLA

- Wang et al., *GLUE*, 2018: https://aclanthology.org/W18-5446/
- Dataset: https://huggingface.co/datasets/nyu-mll/glue

### SuperGLUE: BoolQ

- Wang et al., *SuperGLUE*, 2019: https://arxiv.org/abs/1905.00537
- Dataset: https://huggingface.co/datasets/aps/super_glue

### IMDb

- Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011:
  https://aclanthology.org/P11-1015/
- Original dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- Hugging Face dataset: https://huggingface.co/datasets/stanfordnlp/imdb

## Project-specific components

Normalized retention, BERT-mini parameter scopes, equal-norm controls, early-layer attenuation,
scope-specific density schedules, error-transition categories, directional diagnostics, and
projection-balanced weighting are project experiments or diagnostics rather than reproductions of a
single published method.
