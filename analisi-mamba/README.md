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

1. **Modello.** La cartella predefinita è la cache di Hugging Face
   (`C:\Users\<nome>\.cache\huggingface\hub`), dove sono già il 130M e il 1.4B.
   Si sceglie il modello e si preme *Carica e verifica*. Il programma lo carica in FP32 senza rete
   e fa sei controlli: architettura, tokenizer, pesi, struttura interna, valori finiti, prova di lettura.
2. **Testo.** Si sceglie tra testo di esempio (fisso, per risultati ripetibili), testo incollato
   o file `.txt`. Serve solo alle fasi 2a e 2b.
3. **Fase.** Si sceglie la fase e si preme *Esegui*. Si ottengono riquadri di sintesi, grafici
   con la guida alla lettura, la tabella per strato e un file Excel scaricabile.
   Ogni risultato viene salvato anche in `risultati/`.

## Struttura

```
app.py            interfaccia (solo presentazione)
nucleo/           modelli, testo, sonde, formato dei numeri, grafica, Excel
fasi/             una fase per file: ogni file con FASE = Fase(...) diventa una voce del menu
  fase1_poli_continui.py
  fase2a_delta.py
  fase2b_poli_discreti.py
  _modello_nuova_fase.py   modello da copiare (ignorato: inizia con "_")
```

## Aggiungere una fase (es. 3a)

1. Copiare `fasi/_modello_nuova_fase.py` in `fasi/fase3a_pesi_bassa_precisione.py`.
2. Scrivere il calcolo in `esegui(ctx)` e restituire un `Risultato`: colonne, righe per strato,
   indicatori, legenda, grafici.
3. Se servono scelte dell'utente (es. FP16/BF16), aggiungerle in `parametri=[Parametro(...)]`:
   compaiono da sole come menu sopra il pulsante *Esegui*.
4. Premere *Ricarica le fasi* nella barra laterale. Se si modifica `nucleo/`, bisogna riavviare l'app.

## Note

- Tutte le misure sono in FP32, con riferimento esatto in FP64 per il margine 1 − ā.
- GPU e CPU possono dare conteggi di "ā = 1" leggermente diversi: quando si riportano i risultati
  conviene indicare il dispositivo, che compare nel file Excel.
- Il controllo automatico dei file di Streamlit è spento (`.streamlit/config.toml`), perché
  riempirebbe il terminale di errori innocui provenienti da `transformers`.
