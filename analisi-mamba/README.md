# Analisi dei modi di Mamba

App locale (si apre nel browser, gira tutta sul PC) per l'analisi spettrale di modelli Mamba-1:
poli continui, passo Δ e poli discreti ā, strato per strato, e l'effetto della bassa precisione sull'uscita.

## Installazione (una volta sola)

Si mette la cartella `analisi-mamba` dentro `tesi-mamba` e si riusa l'ambiente `.venv` che c'è già.

```
cd tesi-mamba
.venv\Scripts\activate
cd analisi-mamba
pip install -r requirements-lock.txt
```

`requirements-lock.txt` è l'ambiente esatto del PC fisso con cui sono stati prodotti i risultati (`pip freeze`):
Python 3.14, PyTorch 2.14.1 con CUDA 12.6 (preso dall'indice di PyTorch indicato nel file), transformers 5.17.0,
Streamlit 1.66. Le versioni di PyTorch e transformers compaiono anche nelle informazioni di ogni file Excel.

In alternativa, `requirements.txt` ha solo le versioni minime, ma **transformers resta fissata alla 5.17.0**: la Fase 5
sostituisce `modeling_mamba.mamba_selective_scan`, che esiste in quella forma solo nella 5.x. Con un'altra versione la
Fase 5 si ferma con un errore se la funzione manca; se c'è ma è cambiata potrebbe misurare altro, quindi prima di
usarla va rieseguito `prove/prova_ricorrenza.py` (e conviene confrontare tutte le fasi con `prove/fotografia/`, vedi
Note). In questo caso PyTorch va installato prima, con l'indice CUDA giusto (numero da `nvidia-smi`), per esempio:
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

La procedura è a passi. La timeline in cima alla pagina (sempre visibile) riassume modello, testo e fasi e si
aggiorna mentre il programma lavora: i controlli del modello uno per uno, poi una tacca per ogni analisi, che
diventa verde o rossa quando finisce (passando il mouse su una tacca si leggono esito e durata).

1. **Modello.** All'apertura si sceglie il modello dalla cartella (predefinita: la cache di Hugging Face,
   `C:\Users\<nome>\.cache\huggingface\hub`) e il dispositivo, poi *Carica e verifica*. Il programma lo
   carica in FP32 senza rete e fa sei controlli: architettura, tokenizer, pesi, struttura interna, valori
   finiti, prova di lettura.
2. **Testo.** Quattro fonti: testo di prova (visibile per intero, fisso per risultati ripetibili), **articolo estratto
   da un corpus** (vedi *Testi reali* qui sotto), testo scritto o incollato, file `.txt`. *Esegui tutte le fasi* esegue
   ogni fase in tutti i valori del parametro **Confronto**.
3. **Risultati.** Un menu sceglie la fase e il confronto: riquadri di sintesi, grafici con la guida alla lettura,
   tabella per strato, download in Excel. In ogni fase i **risultati fondamentali** sono in primo piano; le misure di
   contorno (conteggi di verifica, metriche ridondanti, grafici di dettaglio, colonne secondarie) stanno nei riquadri
   compressi "Misure secondarie", "Grafici secondari" e "Tabella completa". L'Excel contiene sempre tutto: nella
   sintesi i secondari sono sotto l'etichetta "Misure secondarie". Niente viene scritto su disco finché non si preme *Salva* nel riquadro
   in fondo: la singola analisi (un file con anche i dati dei grafici) oppure *Salva tutto in un file Excel*. I file
   vanno in `risultati/`.

*Ricarica le fasi* (nella riga dei pulsanti delle pagine Testo e Risultati) rilegge i file in `fasi/`.

## Testi reali: WikiText-2 e notizie BBC

Una volta sola, con la connessione, dalla cartella `analisi-mamba` con l'ambiente attivo:
```
python scarica_testi.py
```
Scarica in `testi/corpora/` due corpora (pochi MB in tutto; nessuna libreria nuova, i file passano da una cartella
temporanea e non restano nella cache di Hugging Face). Da quel momento l'app li legge senza rete.

| Corpus | Dataset | Il modello può averlo visto? |
|---|---|---|
| WikiText-2 · test | `Salesforce/wikitext`, `wikitext-2-raw-v1` (Merity et al., 2016) | Sì, è possibile: Wikipedia in inglese è nel Pile, su cui è addestrato Mamba |
| BBC News · mese | `RealTimeData/bbc_news_alltime` (Li et al., 2024) | No: notizie della prima pagina BBC di un mese dal 2021 in poi, dopo la raccolta del Pile (2020) |

- WikiText-2 è il riferimento della letteratura per la perplexity; il testo è ricostruito (punteggiatura riattaccata,
  ` @-@ ` → `-`) come in lm-evaluation-harness di EleutherAI. Le notizie BBC sono la verifica sui testi mai visti:
  se l'effetto di FP16/BF16 è lo stesso sui due corpora, il risultato non dipende dalla contaminazione.
- Predefinito: il mese BBC più recente disponibile. Altri mesi: `python scarica_testi.py --mesi-bbc 2023-11,2023-12`
  (i mesi precedenti al 2021 sono rifiutati). `--solo wikitext|bbc` ne scarica uno, `--forza` riscarica.
- Ogni file registra la revisione (commit) del dataset, così l'estrazione è ripetibile.
- Nell'app, fonte *Estrai da un corpus*: si sceglie il corpus (con l'indicazione "mai visto dal modello" o "forse
  visto in addestramento"), poi un articolo a caso con **🎲 Estrai un altro** oppure per titolo dal menu; l'anteprima
  mostra titolo, data, collegamento all'originale e testo intero. Sono proposti **solo gli articoli lunghi almeno
  quanto "Token massimi"**: ogni testo estratto ha quindi esattamente quel numero di token e i risultati sono
  confrontabili fra articoli.
- Negli Excel, oltre a "Fonte del testo", compaiono corpus, dataset, revisione, numero e titolo dell'articolo, data,
  collegamento e la riga "Nei dati di addestramento di Mamba": con questi dati la stessa misura si ripete, per esempio
  sul 1.4B con lo stesso articolo.
- `testi/` è esclusa dal repository (`.gitignore`): i testi BBC restano solo sul PC.

## Fasi: grandezza × confronto

Le fasi sono organizzate per **grandezza**. Le Fasi 1–4 hanno il parametro **Confronto** con due valori,
FP32 vs FP16 e FP32 vs BF16: non c'è più "Solo FP32", perché ogni confronto contiene già tutte le grandezze del modello
FP32 (colonne FP32 nella tabella, valore FP32 a sinistra della freccia negli indicatori). La Fase 5, un caso limite, ha
il suo parametro **Formato della ricorrenza** (FP16 / BF16).

| Fase | Grandezza | In primo piano | In secondo piano |
|---|---|---|---|
| 1 – Poli continui | $a = -\exp(A_{\log})$, dai pesi (non dipende dal testo) | spettro e distribuzione di \|a\|, polo più lento, errore relativo tipico e massimo su a, poli spostati oltre l'1% | poli instabili (zero per costruzione), polo più veloce, limite teorico, errore per strato |
| 2 – Passo Δ | Δ token × canale, dal testo | Δ azzerati (underflow FP16), errore tipico su Δ per profondità, passo tipico più piccolo, Δ minimo/massimo | P99/max, non finiti, mappa strato × token |
| 3 – Margini di stabilità | ā = exp(Δ·a), margine 1 − ā e memoria sul testo | **regimi di memoria sul testo**, **poli su 1 potenzialmente rilevanti**, **variazione della memoria sul testo**, poli su 1 con arrotondamento esatto, congelati e integratori, poli nuovi, distribuzione dei margini | conteggio con l'exp del dispositivo, margine minimo, poli in 0, persi, errore sul margine |
| 4 – Effetto sull'uscita | predizione del token successivo | perplexity FP32 → ridotto, KL media, KL primo → ultimo quarto, **pendenza della KL lungo il testo** | KL massima, accordo top-1, ΔNLL, istogramma della KL |

| Fase | Grandezza | FP16 / BF16 (formato della ricorrenza, pesi sempre FP32) |
|---|---|---|
| 5 – Caso limite: ricorrenza ridotta | ā, ΔBu e stato nel formato ridotto | A: poli nuovi su 1 e a 0 per arrotondamento di ā (soglie verificate 2⁻¹² / 2⁻⁹), **quanti dei nuovi su 1 sono potenzialmente rilevanti**, memoria persa, errore sul margine; B: uscita con la ricorrenza simulata (B1 solo ā, B2 anche ΔBu e stato) con le metriche della Fase 4, **pendenza della KL** e **stato massimo FP32 → B1 → B2** |

- **Memoria sul testo (Fasi 3 e 5).** Per ogni modo si somma il passo su tutto il testo, θ = Σ Δ (tempo interno del
  canale), e si calcola |a|θ: la frazione di stato che il modo conserva sull'intero testo è ρ = e^(−|a|θ). Regimi:
  |a|θ > 10 dimentica, 0,1–10 critico, < 0,1 integratore di fatto (conserva più del 90%). Un polo su 1 è
  "potenzialmente rilevante" se è un integratore (Δ > 0) in un modo con |a|θ ≥ 0,1; gli altri sono certamente
  trascurabili su quel testo. Le soglie sono `REGIME_BASSO` e `REGIME_ALTO` in `nucleo/margini.py`. I regimi dipendono
  dalla lunghezza del testo.
- **Pendenza della KL (Fasi 4 e 5).** Retta dei minimi quadrati di log10(KL media) contro log10(posizione), su blocchi di
  32 posizioni saltando il primo (`nucleo/uscita.py`, `pendenza_loglog`): circa 0 = errore costante, circa 2 = cresce
  come t². Servono almeno 4 blocchi utili, cioè circa 160 token.
- **Stato massimo in FP32 (Fase 5).** Misurato con un passaggio in più, la ricorrenza simulata in FP32 (identica bit a bit
  all'originale): la Fase 5 dura di conseguenza circa un terzo in più.
- **Con un confronto** ogni pagina mostra riferimento e formato ridotto insieme: indicatori "valore FP32 → valore
  ridotto" con la differenza, tabella con ogni colonna FP32 seguita dalla stessa colonna nel formato ridotto e poi
  le metriche del confronto, grafici con il FP32 in blu e il formato ridotto in arancione sullo stesso grafico (il
  rosso è riservato alle soglie, per esempio FP32 = 2⁻²⁵).
- La **Fase 4** ha una riga per blocco di 64 posizioni del testo invece che per strato (la colonna "Posizioni").
- La **Fase 5 non corrisponde ai runtime reali**: transformers tiene ā e lo stato in FP32 anche con i pesi in FP16/BF16.
  È un'analisi di sensibilità: cosa succederebbe se anche la ricorrenza fosse in bassa precisione.
- Δ e ā restano in due fasi separate di proposito: separarli permette di capire se un polo su 1 dipende da a o da Δ.
- Le colonne FP32 vengono dal calcolo FP32 della fase (Δ ricalcolato in FP32 e in FP64); gli errori del
  confronto usano come riferimento il Δ del modello FP32 con le operazioni di transformers. Sul 130M i due
  riferimenti danno gli stessi numeri; su un altro modello o dispositivo i poli su 1 FP32 delle due colonne
  potrebbero differire di poco, solo per questo.
- Excel della singola analisi: con un confronto le colonne del riferimento e del confronto stanno nella stessa
  tabella, e il confronto è nel nome del file (es. `..._Fase3_FP32-vs-FP16.xlsx`) e nelle informazioni; un foglio
  in più per i dati di ogni grafico.
- Excel di tutte le analisi (`..._Tutte-le-fasi.xlsx`): un solo foglio con le informazioni comuni una volta sola,
  un indice con i collegamenti alle sezioni e, per ogni fase, una fascia di titolo, poi per ogni confronto sintesi e
  tabella per strato, e la legenda delle colonne alla fine della fase. Le fasi si possono comprimere (struttura di
  Excel). I dati dei grafici non ci sono: restano nell'Excel della singola analisi.

## Misure per la tesi

- **Dispositivo fisso: CPU.** I poli su 1 calcolati con l'exp FP32 dipendono dal dispositivo (vedi Note); per la tesi
  si riportano i valori con arrotondamento esatto, ma conviene eseguire tutto sullo stesso dispositivo.
- **Più testi reali da 2048 token**, estratti dai corpora (alcuni da WikiText-2 e alcuni dalle notizie BBC mai viste),
  uno alla volta con *Esegui tutte le fasi* e *Salva tutto in un file Excel*: le medie e la variabilità tra i testi, e il
  confronto fra i due corpora, vanno nel Capitolo 5.
- **Entrambi i modelli**, 130M e 1.4B, con gli stessi testi.
- La Fase 1 non dipende dal testo: basta un'esecuzione per modello.

## Struttura

```
app.py            interfaccia (solo presentazione)
scarica_testi.py  scarica una volta sola i corpora in testi/corpora/ (unico passaggio con la rete)
nucleo/           modelli, testo, corpora (lettura e pulizia dei corpora), sonde, precisione, confronto, margini,
                  uscita (metriche della Fase 4 e 5), ricorrenza (soglie e ricorrenza simulata della Fase 5),
                  formato dei numeri, grafica, Excel
testi/corpora/    corpora scaricati (un JSON per corpus; fuori dal repository)
fasi/             una fase per file: ogni file con FASE = Fase(...) diventa una voce del menu
  fase1_poli_continui.py
  fase2_delta.py
  fase3_margini.py
  fase4_uscita.py
  fase5_ricorrenza_ridotta.py
  _modello_nuova_fase.py   modello da copiare (ignorato: inizia con "_")
  _archivio/               le sei fasi precedenti (1, 2a, 2b, 3a, 3b, 3c), fuori dal menu
prove/            script da riga di comando (vedi Note)
```

## Aggiungere una fase

1. Copiare `fasi/_modello_nuova_fase.py` in `fasi/fase6_nome.py`.
2. Scrivere il calcolo in `esegui(ctx)` e restituire un `Risultato`: colonne, righe per strato,
   indicatori, legenda, grafici.
3. Se servono scelte dell'utente, aggiungerle in `parametri=[Parametro(...)]`: compaiono da sole come menu
   sopra i risultati. Per un confronto FP32 contro FP16/BF16 si riusa `nucleo.confronto.PARAMETRO` con le
   funzioni dello stesso modulo, come nelle Fasi 1–4.
4. Premere *Ricarica le fasi*. Se si modifica `nucleo/`, bisogna riavviare l'app.

## Note

- Il riferimento è il modello in FP32; il margine 1 − ā è calcolato esattamente in FP64 con `expm1`, e la
  ricorrenza resta in FP32 anche nel confronto (pesi in FP16/BF16, ā calcolato in FP32 come fa il modello).
- Nel confronto la copia con i pesi in FP16 o BF16 è letta dal disco; il modello FP32 caricato non viene mai
  modificato (l'impronta dei pesi viene verificata). Sulle GPU Turing (RTX 20xx) BF16 non è nativo: PyTorch lo
  emula, e il risultato lo indica.
- Le Fasi 2, 3 e 4 usano lo stesso passaggio nella copia a bassa precisione: viene fatto una volta per modello,
  testo e formato e tenuto in memoria (su CPU) per la sessione, quindi passare da una fase all'altra non ricarica
  il modello. Col 1.4B e 1024 token occupa circa 400 MB di RAM per formato, più circa 200 MB per il `lm_head` della
  copia usato dalla Fase 4; si svuota cambiando modello o testo. Anche lo stato finale del modello FP32 (Fasi 4 e 5)
  è calcolato una volta sola.
- La Fase 4 calcola logit, NLL e KL a blocchi di 64 posizioni: in memoria non c'è mai un tensore token × vocabolario
  intero. I logit sono nel formato dei pesi, come fa `lm_head`; `log_softmax`, NLL e KL sono in FP64.
- **Fase 5, ricorrenza simulata.** In transformers 5.x il ciclo della ricorrenza è la funzione di modulo
  `modeling_mamba.mamba_selective_scan` (r. 213 `discrete_A`, r. 215 `deltaB_u`, r. 262 `ssm_state` nella 5.17),
  chiamata da `MambaMixer.forward`. `nucleo/ricorrenza.py` la sostituisce solo dentro un context manager, con una copia
  delle stesse operazioni più gli arrotondamenti, e la ripristina sempre; un contatore verifica che sia usata in ogni
  strato. Con il formato FP32 dà logit identici bit a bit all'originale (`prove/prova_ricorrenza.py`). In B2 ogni passo
  `ā·h + ΔBu` è calcolato in FP32 con operandi nel formato ridotto e poi lo stato è arrotondato al formato.
- **Soglie della Fase 5**: ā arrotondato va a 1 se 1 − ā ≤ 2⁻¹² (FP16) o ≤ 2⁻⁹ (BF16), a 0 se ā ≤ 2⁻²⁵ (FP16) o
  ≤ 2⁻¹³⁴ (BF16); i casi esattamente a metà seguono l'arrotondamento al pari. Sono ricavate a ogni esecuzione sul
  dispositivo e verificate in modo esaustivo da `prove/prova_soglie.py` (CPU e GPU: identiche). Sul testo di prova il
  130M ha 4 905 929 poli nuovi su 1 in FP16 e 11 926 453 in BF16 su CPU (4 857 273 e 11 877 830 su GPU, dove l'exp
  FP32 porta a 1 già più poli).
- **Testi lunghi**: "Token massimi" vale 2048 di predefinito, la lunghezza di addestramento di Mamba (Gu e Dao, 2023,
  §4.2); si può salire fino a 8192, con un avviso oltre 2048. I testi più lunghi vengono troncati.
  Col 1.4B a 4096 token le Fasi 2–3, eseguite insieme, tengono in RAM circa 3 GB di Δ più circa 1,6 GB per formato; sulla GPU da 8 GB la
  Fase 5B usa la stessa memoria di un passaggio normale del modello, che a 4096 token può non bastare.
- **Caricamento dei pesi** (`nucleo/modelli.py`): il modello è costruito vuoto, allocato direttamente sul dispositivo e
  riempito leggendo i file `.safetensors` un tensore alla volta, senza mmap. Su Windows safetensors e transformers
  impegnano memoria virtuale (commit) pari all'intero file dei pesi mentre i pesi arrivano sulla GPU: col 1.4B in
  FP32 il picco del processo era 11,3 GB (5,1 GB di pesi sulla GPU, che con WDDM contano anche nel commit, più 4,6 GB
  del primo file) e con RAM + file di paging già occupati da altri programmi dava l'errore 1455 ("file di paging
  troppo piccolo"). Ora il picco è 7,5 GB, con pesi e logit identici bit a bit a `from_pretrained` (FP32, FP16,
  BF16); se il caricamento diretto fallisce per altri motivi si torna a `from_pretrained`.
- Sulla GPU, durante quel passaggio, il modello FP32 lascia spazio alla copia: va su CPU se la memoria virtuale
  di Windows (RAM + file di paging) basta, altrimenti (tipicamente col 1.4B) i suoi pesi vengono tolti e alla
  fine riletti dal disco. Il riquadro "Formato dei pesi" dice quale strada è stata usata.
- `prove/` contiene cinque script da riga di comando:
  - `prova_delta_modello.py`: Δ ricalcolato = Δ interno del modello in FP32/FP16/BF16, e modello FP32 intatto
    dopo la copia;
  - `fotografa_fasi.py cartella [--confronta riferimento] [--fasi 1,2]`: esegue le fasi in tutte le
    combinazioni sul testo di esempio, più i controlli nascosti "FP32 vs FP32" (Fasi 1–4) e "ricorrenza in FP32"
    (Fase 5), che devono dare errori nulli, e salva righe, indicatori e versioni delle librerie in JSON.
    **Test di regressione:** `prove/fotografia/` è la fotografia delle fasi al 22/10/2026 (130M su CPU). Dopo la
    riorganizzazione del 28/10/2026 (via "Solo FP32", nuove misure, nuovi indicatori) va rigenerata una volta con
    `python prove/fotografa_fasi.py prove/fotografia` dopo aver verificato che l'app funzioni; da quel momento torna a
    essere il riferimento.
    Dopo ogni modifica al codice, `python prove/fotografa_fasi.py nuova --confronta prove/fotografia` deve dire
    "Tutte identiche al riferimento" (due esecuzioni sullo stesso PC danno numeri identici); se cambiano le versioni
    delle librerie lo script lo segnala prima delle differenze;
  - `prova_soglie.py [--gpu]`: soglie di arrotondamento di ā verso 1 e verso 0, in modo esaustivo;
  - `prova_ricorrenza.py [--gpu]`: ricorrenza simulata identica bit a bit in FP32 e funzione originale ripristinata;
  - `verifica_fusione.py nuova prove/fotografia_fasi_precedenti`: confrontava le Fasi 1–3 con la fotografia delle sei
    fasi precedenti (esito al momento della fusione: identiche). Usa "Solo FP32", che non esiste più: è storico e non
    va più eseguito.
- **I poli su 1 in FP32 dipendono dal dispositivo.** Δ e margini esatti sono gli stessi su CPU e GPU, ma l'exp
  in FP32 della GPU non è arrotondato correttamente vicino a 1 e porta a 1 anche margini oltre 2⁻²⁴: sul testo di
  prova il 130M ha 50 958 poli su 1 su CPU e 99 515 su GPU, il 1.4B 495 310 su CPU e 919 870 su GPU. I poli in 0 non
  cambiano (underflow). La Fase 3 riporta anche "Poli su 1, arrotondamento esatto" (margine esatto < 2⁻²⁵), uguale
  su ogni dispositivo. Quando si riportano i risultati conviene indicare il dispositivo, che compare nel file Excel.
- Il controllo automatico dei file di Streamlit è spento (`.streamlit/config.toml`), perché
  riempirebbe il terminale di errori innocui provenienti da `transformers`.
