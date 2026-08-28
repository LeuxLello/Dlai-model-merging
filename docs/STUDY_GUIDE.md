# Guida di studio: capire il progetto dall'inizio

Questa guida separa esplicitamente tre livelli: teoria pubblicata, scelte sperimentali del progetto e risultati osservati. Non serve imparare ogni dettaglio di PyTorch: bisogna saper raccontare la domanda, le formule, il protocollo e i limiti.

## 1. La domanda in una frase

Partiamo da un unico BERT-mini pre-addestrato e lo specializziamo separatamente su quattro compiti linguistici. Poi proviamo a combinare a coppie i pesi degli encoder senza nuovo addestramento e misuriamo quanta competenza dei due specialisti rimane. La domanda centrale è:

> La compatibilità geometrica tra gli aggiornamenti appresi da due specialisti predice quanto bene essi possono essere fusi?

Il progetto non sta creando quattro database e non genera dati sintetici. I testi provengono da dataset pubblici scaricati a runtime da Hugging Face. Ogni specialista riceve testi ed etichette e impara un compito binario.

## 2. Gli specialisti, in parole semplici

| Specialista | Dataset | Input reale | Domanda |
|---|---|---|---|
| SST-2 | GLUE/SST-2 | una frase tratta da recensioni | il giudizio è positivo o negativo? |
| IMDb | IMDb Large Movie Review | una recensione cinematografica | la recensione è positiva o negativa? |
| MRPC | GLUE/MRPC | due frasi | esprimono sostanzialmente la stessa cosa? |
| RTE | GLUE/RTE | premessa e ipotesi | la premessa implica l'ipotesi? |
| CoLA | GLUE/CoLA | una frase | è linguisticamente accettabile? |
| BoolQ | SuperGLUE/BoolQ | domanda e passaggio | la risposta è sì o no? |

SST-2 e IMDb sono compiti affini; MRPC e RTE richiedono relazioni tra due frasi. Questa diversità permette di confrontare fusioni intuitivamente compatibili e fusioni più difficili.

## 3. Da un modello base a un task vector

Indichiamo con `theta_0` i pesi dell'encoder BERT-mini iniziale e con `theta_t` i pesi dopo il fine-tuning sul compito `t`. Il task vector è:

```text
tau_t = theta_t - theta_0
```

È la “direzione” nello spazio dei parametri prodotta dall'apprendimento del compito. Nel codice è `task_vectors()` in `src/dlai_merge/merging.py`.

Importante: fondiamo soltanto l'encoder condiviso. Ogni compito mantiene la propria testa di classificazione. Quindi non otteniamo un singolo classificatore che indovina autonomamente quale domanda rispondere: otteniamo un encoder fuso che viene valutato con la testa del compito noto.

## 4. I tre metodi di fusione

### Mean

```text
theta_merge = theta_0 + (1/T) * sum_t tau_t
```

Poiché tutti gli specialisti partono dallo stesso `theta_0`, equivale alla media dei loro encoder. È il baseline più semplice.

### Task Arithmetic

```text
theta_merge = theta_0 + lambda * sum_t tau_t
```

`lambda` regola l'intensità dell'aggiornamento combinato. La formula deriva direttamente da Ilharco et al. (2023).

### TIES: Trim, Elect Sign, Merge

TIES prova a ridurre l'interferenza tra task vector:

1. **Trim:** conserva gli aggiornamenti di magnitudine maggiore.
2. **Elect sign:** per ogni coordinata sceglie il segno dominante.
3. **Disjoint merge:** media solo gli aggiornamenti concordi con quel segno.

Nota di audit: l'implementazione usata nei notebook 01-08 applica il top-k separatamente a ogni tensore. L'implementazione ufficiale TIES appiattisce invece l'intero task vector e applica globalmente il top-k. Il notebook 09 ha completato il confronto correttivo; i risultati precedenti restano correttamente denominati **tensor-wise TIES**.

## 5. Cosa misuriamo

Per ogni coppia di compiti calcoliamo prima diagnostiche sui task vector:

```text
cosine(tau_a, tau_b) = <tau_a, tau_b> / (||tau_a|| ||tau_b||)
```

Un coseno alto significa che i due fine-tuning spingono molti parametri in direzioni simili. Misuriamo anche norme e accordo di segno.

Dopo la fusione valutiamo entrambi i compiti e normalizziamo rispetto allo specialista:

```text
retention = score_merged / score_specialist
```

Retention `1.0` significa che il modello fuso conserva il 100% del punteggio dello specialista; `0.93` significa circa il 93%. Non significa accuratezza del 93%.

## 6. Cosa hanno fatto i notebook

| Notebook | Ruolo | Tipo di evidenza |
|---|---|---|
| 01 | test minimo di ambiente, training e funzioni di merge | controllo tecnico |
| 02 | addestra i quattro specialisti con seed 42 e budget uguale | pilot |
| 03 | esplora 6 coppie, metodi, scale e densità | esplorativo/selezione |
| 04 | congela le configurazioni e ripete con seed 7, 42, 123 | confermativo principale |
| 05 | localizza l'interferenza per gruppi di layer e controlla le norme | ablation esplicativa |
| 06 | prova attenuazione dei primi layer, scelta su 42 e test su 7/123 | tentativo di miglioramento; controllo vince |
| 07 | prova densità TIES diverse per scope, stesso split sviluppo/test | tentativo di miglioramento; controllo vince |
| 08 | osserva esempi reali persi, preservati o recuperati | analisi qualitativa |
| 09 | confronta tensor-wise e global TIES con subset fisso | conferma correttiva finale |
| 10 | aggiunge CoLA/BoolQ, confronta 400/1200 step e diagnostiche direzionali | estensione di generalizzazione |
| 11 | prova pesi projection-balanced su specialisti a 1200 step | test held-out finale; non migliora TIES |

## 7. Cosa abbiamo scoperto finora

- Il coseno tra task vector è positivamente associato alla retention in tutti e tre i seed, ma ogni correlazione usa soltanto sei coppie: è evidenza coerente, non una legge generale.
- Tra i tre metodi congelati, tensor-wise TIES ha la retention media più alta e stabile, circa `0.936`; Task Arithmetic circa `0.928`; Mean circa `0.927`.
- L'interferenza non è spiegata soltanto dalla norma degli aggiornamenti ed è distribuita tra i layer.
- Le due estensioni proposte non hanno superato i controlli su seed tenuti separati. È un risultato negativo valido: evita una falsa dichiarazione di miglioramento.
- L'analisi degli esempi mostra che coppie affini di sentiment sono più stabili, mentre RTE è più fragile e asimmetrico.
- Con subset identici tra seed, il coseno resta positivamente associato alla retention. Global TIES non migliora la media della variante tensor-wise: `0.9307` contro `0.9380`; il delta medio è `-0.00723` con intervallo bootstrap `[-0.01489, 0.00068]`. Global TIES evita però il caso peggiore più severo di tensor-wise TIES.
- Nell'estensione, CoLA non impara a 400 step: le 45 coppie-seed complete devono essere accompagnate dal controllo senza CoLA su 30 unità. In questo controllo tensor-wise TIES ha la perdita media più contenuta (`-0.0355`).
- A 1200 step tutti e sei gli specialisti seed-42 migliorano, ma la degradazione media della fusione aumenta per tutti i metodi. Specializzazione più forte non implica fusione più facile.
- Nel notebook 10 le diagnostiche direzionali erano esplorative; il notebook 11 ne replica la direzione sui seed held-out, senza trasformarle in un metodo di fusione efficace.
- Su seed 7 e 123 a 1200 step, la relazione direzionale si replica ma usarla per assegnare un solo peso a ciascun task non migliora la fusione: projection-balanced perde `0.00894` rispetto a tensor-wise TIES e l'intervallo bootstrap esclude zero.

## 8. Percorso di lettura minimo

1. **Ilharco et al., Task Arithmetic:** Sezione 2 (task vector e formule) e Sezione 4 (somma dei task vector). Guardare anche la tabella GLUE con SST-2, MRPC e RTE.
2. **Yadav et al., TIES-Merging:** Figura 1, Sezione 4 e Algoritmo 1; poi Sezione 7.3 per l'ablation dei componenti.
3. **Wang et al., GLUE:** descrizione dei compiti e motivazione del benchmark.
4. **Turc et al., compact BERT:** tabella/configurazioni dei modelli piccoli; serve per capire perché BERT-mini è una scelta computazionalmente realistica.
5. **Maas et al., IMDb:** dataset e protocollo delle recensioni.

I link ufficiali e le sezioni precise sono in `references/README.md`. Dopo ogni lettura prova a spiegare a voce: input, output, formula, assunzione e limite. Se uno di questi cinque elementi manca, quella parte non è ancora pronta per l'eventuale orale.

## 9. Limiti da saper dichiarare

- Un solo modello base limita ancora la generalizzazione, anche dopo l'estensione a sei task.
- Tre seed sono meglio di uno, ma non costituiscono un campione ampio.
- Le correlazioni iniziali hanno sei coppie per seed; l'estensione ne ha quindici, ma cinque coinvolgono uno specialista CoLA non appreso a 400 step.
- L'esperimento correttivo e l'estensione separano il subset seed dal training seed.
- Il confronto a 1200 step usa soltanto seed 42 e richiede conferma held-out.
- Servono teste specifiche per compito.
- Tensor-wise e global TIES sono entrambe implementate e confrontate; nessuna domina ogni criterio.

Saper dichiarare questi limiti rafforza il progetto: dimostra che le conclusioni sono proporzionate all'evidenza.
