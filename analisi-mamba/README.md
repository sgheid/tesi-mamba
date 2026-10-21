# Analisi dei modi di Mamba

App locale (si apre nel browser, gira tutta sul PC) per l'analisi spettrale di modelli Mamba-1:
poli continui, passo Δ e poli discreti ā, strato per strato.

## Installazione (una volta sola)

Si mette la cartella `analisi-mamba` dentro `tesi-mamba` e si riusa l'ambiente `.venv` che c'è già.

```
cd tesi-mamba
.venv\Scripts\activate
cd analisi-mamba
pip install -r requirements.txt
```

PyTorch con CUDA è già installato. Su un PC nuovo va installato prima, con l'indice CUDA giusto
(numero da `nvidia-smi`), per esempio:
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Avvio

Doppio clic su `avvia.bat`, oppure dal terminale con l'ambiente attivo:
```
streamlit run app.py
```
Il browser si apre su `http://localhost:8501`. Per chiudere: `Ctrl+C` nel terminale.

## Uso

La procedura è a passi; la barra laterale riassume lo stato e permette di tornare indietro.

1. **Modello.** All'apertura si sceglie il modello dalla cartella (predefinita: la cache di Hugging Face,
   `C:\Users\<nome>\.cache\huggingface\hub`) e il dispositivo, poi *Carica e verifica*. Il programma lo
   carica in FP32 senza rete e fa sei controlli: architettura, tokenizer, pesi, struttura interna, valori
   finiti, prova di lettura.
2. **Testo.** Testo di prova (visibile per intero, fisso per risultati ripetibili), testo incollato o file
   `.txt`. *Esegui tutte le fasi* esegue ogni fase in tutti i valori del parametro **Confronto**.
3. **Risultati.** Un menu sceglie la fase e il confronto: riquadri di sintesi, grafici con la guida alla lettura,
   tabella per strato, download in Excel. Niente viene scritto su disco finché non si preme *Salva* (una fase o
   tutte) nel riquadro in fondo: i file vanno in `risultati/`.

## Fasi: grandezza × confronto

Le fasi sono organizzate per **grandezza**; ognuna ha il parametro **Confronto** con tre valori.

| Fase | Grandezza | Solo FP32 | FP32 vs FP16 / FP32 vs BF16 |
|---|---|---|---|
| 1 – Poli continui | $a = -\exp(A_{\log})$, dai pesi | spettro per strato, distribuzione di \|a\| | stesso spettro con A_log arrotondato, errore relativo su a e limite teorico |
| 2 – Passo Δ | Δ token × canale, dal testo | fascia per strato, mappa strato × token | Δ della copia, errore mediano/P99/max per profondità, Δ azzerati, non finiti |
| 3 – Margini di stabilità | ā = exp(Δ·a) e margine 1 − ā | margine minimo, poli su 1 e in 0, distribuzione dei margini | stesse grandezze nella copia, poli nuovi (per Δ = 0 o per margine sotto 2⁻²⁵) e persi, errore sul margine |

- **Solo FP32** non carica nessuna seconda copia: è veloce come le vecchie Fasi 1, 2a, 2b.
- **Con un confronto** ogni pagina mostra riferimento e formato ridotto insieme: indicatori "valore FP32 → valore
  ridotto" con la differenza, tabella con ogni colonna FP32 seguita dalla stessa colonna nel formato ridotto e poi
  le metriche del confronto, grafici con il FP32 in blu e il formato ridotto in arancione sullo stesso grafico (il
  rosso è riservato alle soglie, per esempio FP32 = 2⁻²⁵).
- Δ e ā restano in due fasi separate di proposito: separarli permette di capire se un polo su 1 dipende da a o da Δ.
- Le colonne FP32 vengono dallo stesso calcolo di "Solo FP32" (Δ ricalcolato in FP32 e in FP64); gli errori del
  confronto usano come riferimento il Δ del modello FP32 con le operazioni di transformers. Sul 130M i due
  riferimenti danno gli stessi numeri; su un altro modello o dispositivo i poli su 1 FP32 delle due colonne
  potrebbero differire di poco, solo per questo.
- Un file Excel per esecuzione: con un confronto le colonne del riferimento e del confronto stanno nella stessa
  tabella, e il confronto è nel nome del file (es. `..._Fase3_FP32-vs-FP16.xlsx`) e nelle informazioni.

## Struttura

```
app.py            interfaccia (solo presentazione)
nucleo/           modelli, testo, sonde, precisione, confronto, formato dei numeri, grafica, Excel
fasi/             una fase per file: ogni file con FASE = Fase(...) diventa una voce del menu
  fase1_poli_continui.py
  fase2_delta.py
  fase3_margini.py
  _modello_nuova_fase.py   modello da copiare (ignorato: inizia con "_")
  _archivio/               le sei fasi precedenti (1, 2a, 2b, 3a, 3b, 3c), fuori dal menu
prove/            script da riga di comando (vedi Note)
```

## Aggiungere una fase

1. Copiare `fasi/_modello_nuova_fase.py` in `fasi/fase4_nome.py`.
2. Scrivere il calcolo in `esegui(ctx)` e restituire un `Risultato`: colonne, righe per strato,
   indicatori, legenda, grafici.
3. Se servono scelte dell'utente, aggiungerle in `parametri=[Parametro(...)]`: compaiono da sole come menu
   sopra i risultati. Per un confronto FP32 contro FP16/BF16 si riusa `nucleo.confronto.PARAMETRO` con le
   funzioni dello stesso modulo, come nelle tre fasi esistenti.
4. Premere *Ricarica le fasi* nella barra laterale. Se si modifica `nucleo/`, bisogna riavviare l'app.

## Note

- Il riferimento è il modello in FP32; il margine 1 − ā è calcolato esattamente in FP64 con `expm1`, e la
  ricorrenza resta in FP32 anche nel confronto (pesi in FP16/BF16, ā calcolato in FP32 come fa il modello).
- Nel confronto la copia con i pesi in FP16 o BF16 è letta dal disco; il modello FP32 caricato non viene mai
  modificato (l'impronta dei pesi viene verificata). Sulle GPU Turing (RTX 20xx) BF16 non è nativo: PyTorch lo
  emula, e il risultato lo indica.
- Le Fasi 2 e 3 usano lo stesso passaggio nella copia a bassa precisione: viene fatto una volta per modello,
  testo e formato e tenuto in memoria (su CPU) per la sessione, quindi passare dalla Fase 2 alla 3 non ricarica
  il modello. Col 1.4B e 1024 token occupa circa 400 MB di RAM per formato; si svuota cambiando modello o testo.
- Sulla GPU, durante quel passaggio, il modello FP32 lascia spazio alla copia: va su CPU se la memoria virtuale
  di Windows (RAM + file di paging) basta, altrimenti (tipicamente col 1.4B) i suoi pesi vengono tolti e alla
  fine riletti dal disco. Il riquadro "Formato dei pesi" dice quale strada è stata usata.
- `prove/` contiene tre script da riga di comando:
  - `prova_delta_modello.py`: Δ ricalcolato = Δ interno del modello in FP32/FP16/BF16, e modello FP32 intatto
    dopo la copia;
  - `fotografa_fasi.py cartella [--confronta riferimento] [--fasi 1,2]`: esegue le fasi in tutte le
    combinazioni sul testo di esempio, più il controllo nascosto "FP32 vs FP32", che deve dare errori nulli,
    e salva righe e indicatori in JSON;
  - `verifica_fusione.py nuova prove/fotografia`: confronta le fasi attuali con `prove/fotografia/`, la
    fotografia delle sei fasi precedenti (130M su CPU). Esito al momento della riorganizzazione: identiche.
- **I poli su 1 in FP32 dipendono dal dispositivo.** Δ e margini esatti sono gli stessi su CPU e GPU, ma l'exp
  in FP32 della GPU non è arrotondato correttamente vicino a 1 e porta a 1 anche margini oltre 2⁻²⁴: sul testo di
  prova il 130M ha 50 958 poli su 1 su CPU e 99 515 su GPU, il 1.4B 495 310 su CPU e 919 870 su GPU. I poli in 0 non
  cambiano (underflow). La Fase 3 riporta anche "Poli su 1, arrotondamento esatto" (margine esatto < 2⁻²⁵), uguale
  su ogni dispositivo. Quando si riportano i risultati conviene indicare il dispositivo, che compare nel file Excel.
- Il controllo automatico dei file di Streamlit è spento (`.streamlit/config.toml`), perché
  riempirebbe il terminale di errori innocui provenienti da `transformers`.
