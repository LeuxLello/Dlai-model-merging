# Audit del progetto e roadmap verso la consegna

## Valutazione onesta

Il progetto ha un nucleo scientifico più solido di una semplice demo: domanda falsificabile, baseline congelate, più seed, ablation, risultati negativi, analisi degli errori e un'estensione a sei task. Il bundle più recente contiene 45 coppie-seed e un controllo sul budget di training. È direttamente ammesso dalle guideline sotto “model merging” e “language models”. Non è possibile garantire un voto, ma il materiale può sostenere un buon progetto se il report resta preciso e lo studente sa spiegare codice, formule e limiti.

## Conformità alle guideline

| Requisito | Stato |
|---|---|
| Tema pertinente | soddisfatto: model merging / language models |
| Repository accessibile | soddisfatto |
| Codice riproducibile | quasi completo; notebook Kaggle e test presenti |
| Report nel template fisso | da scrivere |
| Massimo 2 pagine per uno studente | da rispettare; bibliografia può seguire |
| Dichiarazione AI specifica | bozza aggiornata, da finalizzare con il report |
| Comprensione e verificabilità | in corso tramite guida e checklist |

## Limiti scientifici correnti

1. **CoLA non appreso a 400 step:** Matthews correlation è zero in tutti i seed; usare sempre il controllo senza CoLA per le conclusioni sul merging.
2. **Budget lungo single-seed:** il miglioramento degli specialisti e il peggioramento della fusione a 1200 step sono osservati solo sul seed 42.
3. **Un solo modello base:** tutti i risultati riguardano `prajjwal1/bert-mini`.
4. **Diagnostiche direzionali esplorative:** 90 direzioni sono più informative delle sei coppie iniziali, ma non esiste ancora una validazione su coppie held-out.
5. **Teste specifiche:** il modello fuso richiede l'identità del task e non è un sistema universale end-to-end.

## Stato degli esperimenti chiusi

Global TIES, subset fisso, estensione dei task, budget lungo e diagnostiche direzionali sono completi.
Nel controllo senza CoLA, tensor-wise TIES conserva il miglior score change medio (`-0.0355`), mentre
global TIES evita il peggior caso più severo. A 1200 step tutti gli specialisti migliorano, ma ogni
metodo di fusione peggiora mediamente rispetto alla condizione a 400 step.

## Eventuale ultimo esperimento

Se si prosegue, fare un solo tentativo motivato: una regola direzionale di scaling applicata agli
specialisti a 1200 step. Seed 42 può selezionare una scelta globale da una griglia minima con il
metodo congelato come controllo; seed 7 e 123 devono restare held-out. Non aggiungere nello stesso
run DARE, RegMean e molte varianti conflict-aware, perché renderebbero il risultato post-hoc e
difficile da attribuire.

## Interfaccia ludica

Non è una “merdata”, ma è opzionale. Può essere una buona demo finale se visualizza il fenomeno scientifico, per esempio facendo scegliere due specialisti e mostrando esempi preservati/persi e conflitti di segno. Non deve diventare una seconda ricerca né occupare spazio prezioso nelle due pagine. Priorità:

1. report e figure;
2. eventuale unico test direzionale held-out;
3. prova orale/spiegazione;
4. demo ludica solo se resta tempo.

## Checklist personale dello studente

- [ ] So spiegare ciascun dataset con un esempio inventato.
- [ ] So derivare task vector, Mean e Task Arithmetic su tre numeri.
- [ ] So spiegare trim, elect e disjoint merge senza guardare il codice.
- [ ] So dire perché le teste non vengono fuse.
- [ ] So distinguere accuracy, F1 e retention.
- [ ] So spiegare perché seed 42 è sviluppo nei notebook 06-07.
- [ ] So raccontare almeno un risultato negativo e perché è utile.
- [ ] So elencare i limiti senza minimizzarli.
- [ ] Ho letto personalmente le sezioni indicate in `references/README.md`.
- [ ] Posso descrivere con precisione quale lavoro è stato svolto con Codex.
