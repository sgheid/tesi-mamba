import json
import os
import random
import re
from functools import lru_cache

# Corpora di testi reali, scaricati una volta sola da scarica_testi.py (unico passaggio che usa la rete) e letti da qui
# senza connessione. Un file JSON per corpus: intestazione (nome, dataset, revisione, rapporto con i dati di
# addestramento di Mamba) e l'elenco degli articoli {titolo, testo, data, link}.
CARTELLA_CORPORA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testi", "corpora")

# Lunghezza di addestramento di Mamba e Pythia (Gu e Dao, 2023, §4.2: "context length 2048"): oltre questo valore il
# modello lavora fuori dal regime di addestramento
LUNGHEZZA_ADDESTRAMENTO = 2048


# WikiText ha la punteggiatura staccata dalle parole (" , ", " 's", " @-@ "): stessa ricostruzione del testo di
# lm-evaluation-harness di EleutherAI (task wikitext, wikitext_detokenizer), così il testo è quello su cui la
# letteratura misura la perplexity
def detokenizza_wikitext(s):
    s = s.replace("s '", "s'")
    s = re.sub(r"/' [0-9]/", r"/'[0-9]/", s)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for p in (":", ";", ".", "!", "?", ","):
        s = s.replace(f" {p} ", f"{p} ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" " + chr(176) + " ", chr(176))
    s = s.replace(" \n", "\n").replace("\n ", "\n")
    s = s.replace(" N ", " 1 ")
    return s.replace(" 's", "'s")


# Titolo di articolo in WikiText: riga " = Titolo = " con un solo "=" per lato (le sezioni hanno " = = ... = = ")
_TITOLO_WIKITEXT = re.compile(r"^ = ([^=].*?) = $")


# Le righe dello split (una riga di testo per elemento, come nel dataset) divise in articoli
def articoli_wikitext(righe):
    articoli, titolo, corpo = [], None, []

    def chiudi():
        if titolo is not None:
            testo = detokenizza_wikitext("".join(corpo)).strip()
            articoli.append(dict(titolo=detokenizza_wikitext(titolo).strip(), testo=testo, data="", link=""))

    for riga in righe:
        m = _TITOLO_WIKITEXT.match(riga.rstrip("\n"))
        if m:
            chiudi()
            titolo, corpo = m.group(1), [riga]
        else:
            corpo.append(riga)
    chiudi()
    return [a for a in articoli if a["testo"]]


def _pulisci_spazi(s):
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


# Notizie BBC di RealTimeData (colonne title, published_date, content, link): titolo in testa, spazi ripuliti,
# senza articoli vuoti o ripetuti
def articoli_bbc(righe):
    articoli, visti = [], set()
    for r in righe:
        titolo, corpo = _pulisci_spazi(r.get("title")), _pulisci_spazi(r.get("content"))
        if len(corpo) < 200 or (titolo or corpo[:200]) in visti:
            continue
        visti.add(titolo or corpo[:200])
        testo = corpo if not titolo or corpo.startswith(titolo) else f"{titolo}\n\n{corpo}"
        data = str(r.get("published_date") or "")
        articoli.append(dict(titolo=titolo or "(senza titolo)", testo=testo,
                             data=data[:10] if re.match(r"\d{4}-\d{2}-\d{2}", data) else data,
                             link=str(r.get("link") or "")))
    return articoli


def salva_corpus(nome_file, corpus):
    os.makedirs(CARTELLA_CORPORA, exist_ok=True)
    percorso = os.path.join(CARTELLA_CORPORA, nome_file)
    with open(percorso, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    return percorso


@lru_cache(maxsize=16)
def _leggi(percorso, _mtime):
    with open(percorso, encoding="utf-8") as f:
        corpus = json.load(f)
    corpus["percorso"] = percorso
    return corpus


def carica_corpus(percorso):
    return _leggi(percorso, os.path.getmtime(percorso))


# Corpora presenti in testi/corpora, prima quelli mai visti dal modello; i file illeggibili sono ignorati
def elenco_corpora():
    if not os.path.isdir(CARTELLA_CORPORA):
        return []
    elenco = []
    for nome in sorted(os.listdir(CARTELLA_CORPORA)):
        if nome.endswith(".json"):
            try:
                elenco.append(carica_corpus(os.path.join(CARTELLA_CORPORA, nome)))
            except (OSError, ValueError):
                pass
    return sorted(elenco, key=lambda c: (not c.get("mai_visto"), c.get("nome", "")))


# Token di ogni articolo con lo stesso tokenizer e la stessa chiamata di testo.prepara
def conta_token(tok, corpus):
    return [len(x) for x in tok([a["testo"] for a in corpus["articoli"]]).input_ids]


# Un articolo a caso fra gli idonei, diverso da quello attuale se possibile
def estrai(idonei, attuale=None):
    scelte = [k for k in idonei if k != attuale] or list(idonei)
    return random.choice(scelte)


# Righe per le informazioni dell'Excel: da dove viene il testo e se il modello può averlo visto
def dettagli(corpus, k, n_token):
    a = corpus["articoli"][k]
    voci = [("Corpus", corpus["nome"]),
            ("Dataset", f"{corpus['dataset']} · {corpus['file']}"),
            ("Revisione del dataset", corpus.get("revisione", "")),
            ("Articolo", f"n. {k + 1} di {len(corpus['articoli'])} · {a['titolo']} · {n_token} token in tutto"),
            ("Nei dati di addestramento di Mamba", corpus.get("addestramento", ""))]
    if a.get("data"):
        voci.append(("Data di pubblicazione", a["data"]))
    if a.get("link"):
        voci.append(("Collegamento", a["link"]))
    return voci
