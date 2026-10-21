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
   `.txt`. *Esegui tutte le fasi* esegue ogni fase, e le fasi con parametri in tutte le combinazioni
   (3a, 3b, 3c in FP16, BF16 e FP32 controllo).
3. **Risultati.** Un menu sceglie la fase (e il formato, per la Fase 3): riquadri di sintesi, grafici con la
   guida alla lettura, tabella per strato, download in Excel. Niente viene scritto su disco finché non si
   preme *Salva* (una fase o tutte) nel riquadro in fondo: i file vanno in `risultati/`.

## Struttura

```
app.py            interfaccia (solo presentazione)
nucleo/           modelli, testo, sonde, formato dei numeri, grafica, Excel
fasi/             una fase per file: ogni file con FASE = Fase(...) diventa una voce del menu
  fase1_poli_continui.py
  fase2a_delta.py
  fase2b_poli_discreti.py
  fase3a_poli_continui_arrotondati.py   Fase 3: pesi in FP16/BF16, ricorrenza in FP32
  fase3b_delta_bassa_precisione.py
  fase3c_poli_discreti_bassa_precisione.py
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

- Le Fasi 1 e 2 sono in FP32, con riferimento esatto in FP64 per il margine 1 − ā.
- La Fase 3 confronta con la baseline FP32 una copia del modello con i pesi in FP16 o BF16, letta dal disco;
  il modello FP32 caricato non viene mai modificato. "FP32 (controllo)" deve dare errori esattamente nulli.
  Sulle GPU Turing (RTX 20xx) BF16 non è nativo: PyTorch lo emula, e il risultato lo indica.
- Nelle fasi 3b e 3c, sulla GPU, il modello FP32 lascia spazio alla copia: va su CPU se la memoria
  virtuale di Windows (RAM + file di paging) basta, altrimenti (tipicamente col 1.4B) i suoi pesi vengono
  tolti e alla fine riletti dal disco. Il riquadro "Formato dei pesi" dice quale strada è stata usata.
- `prove/` contiene due script da riga di comando: `prova_delta_modello.py` (Δ ricalcolato = Δ interno
  del modello in FP32/FP16/BF16) e `riesegui_fasi_1_2.py` (numeri delle Fasi 1-2 da confrontare).
- GPU e CPU possono dare conteggi di "ā = 1" leggermente diversi: quando si riportano i risultati
  conviene indicare il dispositivo, che compare nel file Excel.
- Il controllo automatico dei file di Streamlit è spento (`.streamlit/config.toml`), perché
  riempirebbe il terminale di errori innocui provenienti da `transformers`.
