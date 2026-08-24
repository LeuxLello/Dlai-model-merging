# Audit del progetto e roadmap verso la consegna

## Valutazione onesta

Il progetto ha già un nucleo scientifico più solido di una semplice demo: domanda falsificabile, tre baseline, sei coppie di task, tre seed, configurazioni congelate, ablation, risultati negativi e analisi degli errori. È direttamente ammesso dalle guideline sotto “model merging” e “language models”. Non è possibile garantire un voto, ma il materiale può sostenere un buon progetto se il report resta preciso e lo studente sa spiegare codice, formule e limiti.

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

## Rischi scientifici da risolvere

1. **TIES non letterale:** il trim corrente è tensor-wise. Implementare global TIES e confrontarlo con la variante usata.
2. **Campionamento confuso col seed:** i subset di SST-2/IMDb cambiano col seed. Nel prossimo controllo usare un `subset_seed` fisso, distinto dal training seed, oppure dichiarare esplicitamente che la variabilità include entrambi.
3. **Specialisti economici:** 400 step rendono l'esperimento eseguibile ma le accuratezze assolute non sono SOTA. La domanda riguarda la retention; va detto chiaramente.
4. **Poche unità per correlazione:** sei coppie per seed. Mostrare tutti i punti e non sovrainterpretare p-value o causalità.
5. **Teste specifiche:** il modello fuso richiede l'identità del task. Non chiamarlo sistema universale end-to-end.

## Prossima fase consigliata

Un solo esperimento correttivo e mirato, prima del report:

1. implementare global TIES conforme al codice ufficiale;
2. aggiungere test che confrontino flattening, densità e ricostruzione dello state dict;
3. eseguire notebook 09 con subset fisso e training seed 7/42/123;
4. confrontare Mean, Task Arithmetic, tensor-wise TIES e global TIES sulle stesse 18 unità pair-seed;
5. congelare la conclusione; nessuna ulteriore ricerca di iperparametri dopo aver visto il risultato.

Questo esperimento è più utile al voto di nuove varianti speculative, perché chiude una discrepanza rispetto alla fonte primaria e rende la metodologia difendibile.

L'implementazione e il notebook sono ora presenti; manca soltanto il run Kaggle e l'archiviazione del bundle prodotto.

## Interfaccia ludica

Non è una “merdata”, ma è opzionale. Può essere una buona demo finale se visualizza il fenomeno scientifico, per esempio facendo scegliere due specialisti e mostrando esempi preservati/persi e conflitti di segno. Non deve diventare una seconda ricerca né occupare spazio prezioso nelle due pagine. Priorità:

1. correttezza global TIES;
2. report e figure;
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
