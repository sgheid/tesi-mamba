# Analisi dei modi di Mamba

App locale (si apre nel browser, gira tutta sul PC, senza rete) che studia un modello Mamba-1 come sistema dinamico:
poli continui $a$, passo Δ, poli discreti $\bar a = e^{\Delta a}$ e margini $1-\bar a$, strato per strato, e come la
bassa precisione (FP16, BF16) li modifica e cambia l'uscita del modello.

L'app è nella cartella `analisi-mamba/`: tutti i percorsi citati più sotto sono relativi a quella cartella.

## Installazione

Con l'ambiente virtuale `.venv` nella cartella superiore (`tesi-mamba/.venv`):

```
cd tesi-mamba
.venv\Scripts\activate
cd analisi-mamba
pip install -r requirements-lock.txt
```

`requirements-lock.txt` è l'ambiente esatto con cui sono stati prodotti i risultati: Python 3.14, PyTorch 2.14.1 con
CUDA 12.6 (dall'indice di PyTorch indicato nel file), transformers 5.17.0, Streamlit 1.66. `requirements.txt` ha solo
le versioni minime; in quel caso PyTorch va installato prima con l'indice CUDA giusto (numero da `nvidia-smi`), per
esempio `pip install torch --index-url https://download.pytorch.org/whl/cu126`.

**transformers resta fissata alla 5.17.0**: le Fasi 5 e 6 sostituiscono `modeling_mamba.mamba_selective_scan`, che ha
questa forma solo nella 5.x. Con un'altra versione vanno rieseguite le prove (vedi *Verifiche*).

## Avvio

Doppio clic su `analisi-mamba/avvia.bat`, oppure da `analisi-mamba` con l'ambiente attivo `streamlit run app.py` (si apre `http://localhost:8501`).
Il controllo automatico dei file è spento (`.streamlit/config.toml`): dopo una modifica al codice l'app va riavviata.

## Uso

La timeline in alto riassume modello, testo e fasi e si aggiorna mentre il programma lavora (una tacca per analisi).

1. **Modello.** Cartella dei modelli (predefinita: la cache di Hugging Face) e dispositivo, poi *Carica e verifica*:
   il modello è caricato in FP32 e controllato (architettura, tokenizer, pesi, struttura, valori finiti, prova di lettura).
2. **Testo e fasi.** Fonte del testo: testo di prova, articolo estratto da un corpus, testo scritto o file `.txt`,
   troncato a *Token massimi* (predefinito 2048, la lunghezza di addestramento di Mamba). Sotto, una **scheda per fase**
   con spiegazione e pulsante *Esegui*; *Esegui tutte le fasi* lancia le Fasi 1–5, le fasi extra si lanciano solo dalla
   loro scheda. Durante l'esecuzione tutti i comandi sono bloccati. I risultati **si accumulano** sullo stesso testo;
   cambiando testo o modello vengono cancellati (con avviso).
3. **Risultati.** Un menu sceglie fase e confronto. In primo piano le misure fondamentali; verifiche e metriche di
   contorno sono nei riquadri compressi "Misure secondarie", "Grafici secondari" e "Tabella completa". Niente va su disco
   finché non si preme *Salva*:
   - **Salva Fase n**: l'analisi visualizzata, con un foglio in più per i dati di ogni grafico;
   - **Salva tutto in un file Excel**: tutte le analisi in memoria in un solo foglio (informazioni comuni, indice,
     sintesi e tabella per ogni analisi), senza i dati dei grafici.

   I file vanno in `risultati/`.

## Fasi

| Fase | Grandezza | Parametro |
|---|---|---|
| 1 – Poli continui | $a = -\exp(A_{\log})$ dai pesi: spettro, polo più lento, errore dell'arrotondamento | Confronto FP32 vs FP16 / BF16 |
| 2 – Passo Δ | Δ per token e canale: Δ azzerati (underflow FP16), errore tipico per strato | Confronto |
| 3 – Margini di stabilità | $\bar a$, margine $1-\bar a$, regimi di memoria sul testo, poli su 1 rilevanti | Confronto |
| 4 – Effetto sull'uscita | perplexity, KL dal modello FP32, accordo top-1, pendenza della KL lungo il testo | Confronto |
| 5 – Caso limite: ricorrenza ridotta | $\bar a$ (B1) o anche ΔBu e stato (B2) in formato ridotto, pesi FP32: poli portati a 1, memoria persa, uscita | Formato FP16 / BF16 |
| 6 – Extra · diagnostica B2 < B1 | ablazione B1 / B1 + stato / B1 + ingresso / B2 e incrementi assorbiti o azzerati | Formato FP16 / BF16 |

- **Confronto (Fasi 1–4)**: modello FP32 di riferimento affiancato a una copia con i pesi in FP16 o BF16 letta dal
  disco; la ricorrenza resta in FP32, come nei runtime reali. Il modello FP32 non viene mai modificato (impronta dei
  pesi verificata). Le Fasi 2–4 condividono il passaggio nella copia ridotta, calcolato una volta per testo e formato.
- **Fasi 5 e 6 sono analisi di sensibilità**: i runtime tengono $\bar a$ e lo stato in FP32; qui la ricorrenza è
  simulata in bassa precisione sostituendo `mamba_selective_scan` dentro un context manager, che la ripristina sempre.
- **Regimi di memoria (Fasi 3, 5, 6)**: con $\theta = \sum \Delta$ sul testo, un modo dimentica se $|a|\theta > 10$, è
  critico tra 0,1 e 10, è di fatto un integratore sotto 0,1. Un polo portato a 1 è "potenzialmente rilevante" solo se
  $|a|\theta \ge 0{,}1$.
- **Pendenza della KL**: retta di $\log$ KL contro $\log$ posizione su blocchi di 32 token; circa 0 = errore costante,
  circa 2 = errore che cresce come $t^2$. Servono almeno 160 token.
- **Fase 6**: in un passaggio FP32 le quattro ricorrenze ridotte girano "in ombra" sugli stessi ingressi, senza
  cambiare l'uscita (verificato bit a bit), per contare gli incrementi *assorbiti* (fl(ā·h + inc) = fl(ā·h)) e
  *azzerati* (ΔBu → 0) sui poli diventati integratori e sugli altri.

## Testi reali

Una volta sola, con la connessione: `python scarica_testi.py`. Scarica in `testi/corpora/` (pochi MB, fuori dal
repository) WikiText-2 test e un mese di notizie BBC; poi l'app li legge senza rete.

| Corpus | Dataset | Visto in addestramento? |
|---|---|---|
| WikiText-2 · test | `Salesforce/wikitext`, `wikitext-2-raw-v1` | Possibile: Wikipedia è nel Pile |
| BBC News · mese | `RealTimeData/bbc_news_alltime` | No: notizie dal 2021, dopo il Pile (2020) |

Altri mesi: `--mesi-bbc 2023-11,2023-12`; un solo corpus: `--solo wikitext|bbc`; riscaricare: `--forza`. Sono proposti
solo gli articoli lunghi almeno *Token massimi*. Negli Excel compaiono corpus, revisione del dataset, articolo, data e
collegamento, così ogni misura è ripetibile.

## Struttura

```
README.md
analisi-mamba/
  app.py             interfaccia
  avvia.bat          avvio con doppio clic
  scarica_testi.py   download dei corpora (unico passaggio con la rete)
  requirements*.txt  dipendenze (lock = ambiente esatto)
  nucleo/            modelli e caricamento, testo e corpora, sonde su Δ, copie a bassa precisione, confronto, margini,
                     metriche d'uscita, ricorrenza simulata (Fase 5), ricorrenza a blocchi, diagnostica (Fase 6),
                     formato dei numeri, grafica, Excel
  fasi/              una fase per file (FASE = Fase(...)); _modello_nuova_fase.py è il modello da copiare
  prove/             verifiche da riga di comando e fotografia di regressione
```

## Verifiche

Dalla cartella `analisi-mamba`, con l'ambiente attivo (`--gpu` per eseguirle sulla GPU):

- `prove/prova_soglie.py`: soglie di arrotondamento di $\bar a$ verso 1 e verso 0, in modo esaustivo;
- `prove/prova_delta_modello.py`: Δ ricalcolato = Δ interno del modello, modello FP32 intatto dopo la copia;
- `prove/prova_ricorrenza.py`: ricorrenza simulata identica bit a bit all'originale in FP32;
- `prove/prova_blocchi.py`: ricorrenza a blocchi identica bit a bit a quella intera;
- `prove/prova_diagnostica_b2.py`: varianti della Fase 6 identiche bit a bit a B1 e B2, uscita FP32 invariata;
- `prove/fotografa_fasi.py cartella [--confronta riferimento] [--fasi 1,2]`: esegue le fasi (le extra solo se chieste
  con `--fasi`) sul testo di prova, più i controlli FP32 contro FP32 che devono dare errori nulli, e salva i risultati
  in JSON. Test di regressione: generare una volta il riferimento con `python prove/fotografa_fasi.py prove/fotografia`,
  poi dopo ogni modifica `python prove/fotografa_fasi.py nuova --confronta prove/fotografia` deve dire "Tutte identiche".

## Note tecniche

- **Riferimento**: modello in FP32; il margine $1-\bar a$ esatto è calcolato in FP64 con `expm1`.
- **I poli su 1 in FP32 dipendono dal dispositivo**: l'exp FP32 della GPU porta a 1 anche margini oltre $2^{-24}$
  (sul testo di prova il 130M ha 50 958 poli su 1 su CPU e 99 515 su GPU). La Fase 3 riporta anche il conteggio con
  arrotondamento esatto, uguale ovunque; il dispositivo è indicato in ogni Excel.
- **BF16 su GPU Turing** (RTX 20xx) non è nativo: PyTorch lo emula, e i risultati lo indicano.
- **Ricorrenza a blocchi** (`nucleo/scansione.py`): col 1.4B a 2048 token i tensori della discretizzazione (512 MB
  l'uno in FP32) non entrano in una GPU da 8 GB insieme ai pesi. Sulla GPU, sopra 256 MB per tensore, la
  discretizzazione è fatta a blocchi di token con le stesse operazioni: risultati identici bit a bit. Per le prove si
  possono imporre i blocchi con `set MAMBA_TOKEN_PER_BLOCCO=50`.
- **Caricamento dei pesi** (`nucleo/modelli.py`): un tensore alla volta dai file `.safetensors`, senza mmap, per
  contenere la memoria virtuale di Windows (col 1.4B in FP32 il picco scende da 11,3 a 7,5 GB ed evita l'errore 1455,
  "file di paging troppo piccolo"); se fallisce si torna a `from_pretrained`. Pesi e logit sono identici bit a bit.
- **Memoria**: metriche d'uscita a blocchi di 64 posizioni (mai un tensore token × vocabolario intero); sulla GPU, durante
  la copia ridotta, il modello FP32 va su CPU se la memoria basta, altrimenti i suoi pesi sono riletti dal disco.

## Aggiungere una fase

1. Copiare `fasi/_modello_nuova_fase.py` in `fasi/fase7_nome.py`, cambiare i campi di `FASE` e scrivere `esegui(ctx)`,
   che restituisce un `Risultato` (colonne, righe per strato, indicatori, legenda, grafici). Con `extra=True` la fase
   non parte con *Esegui tutte le fasi*.
2. Le scelte dell'utente vanno in `parametri=[Parametro(...)]`; per un confronto FP32 contro FP16/BF16 si riusa
   `nucleo.confronto.PARAMETRO`.
3. Premere *Ricarica le fasi* (se si modifica `nucleo/` va riavviata l'app).
