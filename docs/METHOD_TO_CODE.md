# Tracciabilità: teoria, codice, notebook, evidenza

| Elemento | Provenienza | Codice | Notebook/risultato | Stato |
|---|---|---|---|---|
| `tau_t = theta_t - theta_0` | Task Arithmetic, Sec. 2 | `task_vectors` | 01-08 | replica diretta |
| Mean degli aggiornamenti | weight averaging / baseline | `mean_merge` | 03-08 | baseline |
| `theta_0 + lambda sum(tau_t)` | Task Arithmetic, Sec. 2 e 4 | `task_arithmetic` | 03-06 | replica diretta |
| Trim/elect/disjoint mean | TIES, Sec. 4 | `ties_merge` | 03-08 | logica diretta, trim tensor-wise |
| Top-k globale TIES | TIES codice ufficiale | `global_ties_merge` | notebook 09 | run finale completato |
| Norma L2, coseno, accordo segno | geometria standard + motivazione TIES | `diagnostics.py` | 03-05 | diagnostica del progetto |
| Retention score | normalizzazione del progetto | notebook/evaluation | 03-08 | metrica del progetto |
| Scope embeddings/early/late | architettura BERT-mini | `bert_mini_scopes` | 05-07 | design del progetto |
| Equal-norm | controllo di una spiegazione alternativa | `equal_norm_mean_merge` | 05 | ablation del progetto |
| Early-layer attenuation | ipotesi derivata dall'ablation 05 | `scale_merged_update_by_scope` | 06 | estensione, non migliorativa |
| Scope-density TIES | ipotesi derivata da TIES + ablation 05 | `ties_merge_by_scope` | 07 | estensione, non migliorativa |
| Error transitions | analisi qualitativa | notebook 08 | 08 | spiegazione post-hoc |

## Percorso dell'input

`src/dlai_merge/data.py` contiene il registro dei quattro task. `load_dataset(...)` scarica i dataset pubblici dalla Hugging Face Hub nella cache temporanea di Kaggle. I dataset non sono versionati nel repository. Il tokenizer converte una o due stringhe in token BERT; la colonna `label` rimane il bersaglio supervisionato.

## Percorso dei pesi

1. `training.py` carica `prajjwal1/bert-mini` con una testa binaria.
2. Il fine-tuning aggiorna encoder e testa per 400 optimizer step.
3. `extract_encoder_state` salva l'encoder; `extract_head_state` salva la testa separatamente.
4. `merging.py` opera solo sugli encoder compatibili.
5. `TaskEvaluator` carica un encoder fuso e la testa del task valutato.
6. I punteggi del fuso vengono divisi per quelli dello specialista per ottenere la retention.

## Nomi scientificamente corretti nel report

- Scrivere **tensor-wise TIES variant** per i risultati attuali.
- Scrivere **development seed 42** e **held-out seeds 7/123** per notebook 06-07.
- Scrivere **three-seed confirmation** e non “statistical proof”.
- Scrivere **association** tra coseno e retention, non causalità.
- Scrivere che i tentativi adattivi hanno selezionato il controllo invariato.
