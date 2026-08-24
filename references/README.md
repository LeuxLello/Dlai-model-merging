# Fonti scientifiche e ordine di lettura

Non sono incluse copie dei PDF: i link puntano alle fonti ufficiali, evitano duplicati pesanti e rendono verificabile la provenienza. `references.bib` contiene le citazioni per il report.

## Fondamentali

### Task Arithmetic

- Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023.
- Paper: https://arxiv.org/pdf/2212.04089
- Leggere: Sezione 2, pp. 2-3, per `tau_t = theta_t - theta_0`, scaling e composizione; Sezione 4, pp. 4-5, per la somma; Tabella 3, p. 5, per gli esperimenti GLUE.
- Collegamento al codice: `task_vectors()` e `task_arithmetic()` in `src/dlai_merge/merging.py`.
- Nota rilevante: il paper esclude dalle operazioni condivise le nuove teste introdotte dal fine-tuning; il progetto conserva infatti una testa per task.

### TIES-Merging

- Yadav et al., *TIES-Merging: Resolving Interference When Merging Models*, NeurIPS 2023.
- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf
- Codice ufficiale: https://github.com/prateeky2806/ties-merging
- Leggere: Figura 1 e introduzione; Sezione 4, pp. 4-5; Algoritmo 1, p. 4; Sezione 7.3, p. 10.
- Implementazione minimale ufficiale: `src/ties_minimal.ipynb` nel repository degli autori. Le righe “Flattening out Checkpoints” mostrano che il top-k opera sul task vector globale appiattito.
- Collegamento al codice locale: `ties_merge()` in `src/dlai_merge/merging.py` implementa la stessa logica trim/elect/disjoint-mean, ma il trim è per tensore. Va citata come variante tensor-wise fino al confronto correttivo globale.

### Model Soups (background sul weight averaging)

- Wortsman et al., *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*, ICML 2022.
- Paper: https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf
- Leggere: Sezione 2 e formula della uniform soup a p. 3.
- Uso corretto: background sulla media dei pesi. Non è la fonte principale del nostro Mean, perché Model Soups media varianti dello stesso task, mentre noi fondiamo specialisti di task diversi.

## Modello e dati

### BERT e BERT-mini

- Devlin et al., *BERT*, NAACL 2019: https://aclanthology.org/N19-1423/
- Leggere: Sezione 3 per architettura e fine-tuning; Sezione 4.1 per GLUE.
- Turc et al., *Well-Read Students Learn Better*, 2019: https://arxiv.org/abs/1908.08962
- Model card usata: https://huggingface.co/prajjwal1/bert-mini
- La card documenta `L=4, H=256` e la conversione dai checkpoint Google. È il modello effettivamente indicato da `configs/base.yaml`.

### GLUE: SST-2, MRPC e RTE

- Wang et al., *GLUE*, 2018: https://aclanthology.org/W18-5446/
- Dataset caricato dal codice: https://huggingface.co/datasets/nyu-mll/glue
- Leggere la descrizione dei task e la tabella riepilogativa. Il progetto usa le configurazioni `sst2`, `mrpc`, `rte`.

### IMDb

- Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011: https://aclanthology.org/P11-1015/
- Pagina originale del dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- Dataset caricato dal codice: https://huggingface.co/datasets/stanfordnlp/imdb

## Cosa non viene da un paper specifico

Sono scelte/estensioni del progetto: retention normalizzata, raggruppamento layer BERT-mini, sostituzione per scope, controllo equal-norm, attenuazione dei primi layer, schedule di densità per scope e categorie di transizione degli errori. Devono essere presentate come nostre ablation o strumenti diagnostici, non come riproduzioni della letteratura.
