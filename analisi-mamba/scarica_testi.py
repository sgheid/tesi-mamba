# Scarica una volta sola i corpora di testi reali in testi/corpora/ (pochi MB in tutto). È l'unico passaggio che usa la
# rete: l'app poi li legge senza connessione. Non servono librerie nuove: huggingface_hub arriva con transformers,
# pandas e pyarrow con Streamlit. I file sono scaricati in una cartella temporanea e convertiti, senza copie nella cache.
#   - WikiText-2, split test (Merity et al., 2016): il riferimento della letteratura sulla perplexity. Wikipedia in
#     inglese fa parte del Pile, su cui è addestrato Mamba: il modello può aver visto questi articoli.
#   - BBC News di RealTimeData (Li et al., 2024): articoli della prima pagina BBC raccolti mese per mese. Un mese dal
#     2021 in poi è certamente successivo al Pile (2020): testi che il modello non ha mai visto.
# Uso (dalla cartella analisi-mamba, con l'ambiente attivo):
#   python scarica_testi.py                      WikiText-2 e il mese BBC più recente disponibile
#   python scarica_testi.py --mesi-bbc 2023-11,2023-12
#   python scarica_testi.py --forza              riscarica anche i corpora già presenti
import argparse
import os
import re
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nucleo.corpora import CARTELLA_CORPORA, articoli_bbc, articoli_wikitext, salva_corpus

WIKITEXT = "Salesforce/wikitext"
FILE_WIKITEXT = "wikitext-2-raw-v1/test-"
BBC = "RealTimeData/bbc_news_alltime"
# Il Pile (Gao et al., 2020) è stato raccolto fino al 2020: dal 2021 i testi sono certamente nuovi per Mamba
PRIMO_MESE_NUOVO = "2021-01"


def file_del_dataset(api, repo):
    revisione = api.dataset_info(repo).sha
    return revisione, api.list_repo_files(repo, repo_type="dataset", revision=revisione)


def leggi_parquet(repo, file, revisione):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory() as cartella:
        percorso = hf_hub_download(repo, file, repo_type="dataset", revision=revisione, local_dir=cartella)
        return pd.read_parquet(percorso)


def scarica_wikitext(api, forza):
    nome_file = "wikitext2_test.json"
    if os.path.exists(os.path.join(CARTELLA_CORPORA, nome_file)) and not forza:
        print(f"WikiText-2: già presente ({nome_file}), salto. Con --forza lo riscarico.")
        return
    revisione, files = file_del_dataset(api, WIKITEXT)
    file = next((f for f in files if f.startswith(FILE_WIKITEXT) and f.endswith(".parquet")), None)
    if file is None:
        raise SystemExit(f"Split test di WikiText-2 non trovato in {WIKITEXT}: la struttura del dataset è cambiata.")
    righe = leggi_parquet(WIKITEXT, file, revisione)["text"].tolist()
    articoli = articoli_wikitext(righe)
    salva_corpus(nome_file, dict(
        id="wikitext2_test", nome="WikiText-2 · test", dataset=WIKITEXT, file=file, revisione=revisione,
        scaricato=date.today().isoformat(), citazione="Merity et al., 2016", mai_visto=False,
        breve="Articoli di Wikipedia «Good» e «Featured», versione raw dello split test: il riferimento della "
              "letteratura per la perplexity. Testo ricostruito come in lm-evaluation-harness.",
        addestramento="possibile (Wikipedia in inglese è una delle componenti del Pile)",
        articoli=articoli))
    print(f"WikiText-2: {len(articoli)} articoli salvati in {nome_file}")


def scarica_bbc(api, mesi, forza):
    revisione, files = file_del_dataset(api, BBC)
    disponibili = {}
    for f in files:
        m = re.match(r"^(\d{4}-\d{2})/.*\.parquet$", f)
        if m:
            disponibili.setdefault(m.group(1), f)
    if not disponibili:
        raise SystemExit(f"Nessun mese trovato in {BBC}: la struttura del dataset è cambiata.")
    if not mesi:
        mesi = [max(disponibili)]
    for mese in mesi:
        if mese not in disponibili:
            raise SystemExit(f"Mese {mese} non disponibile. Mesi dal {PRIMO_MESE_NUOVO}: "
                             + ", ".join(m for m in sorted(disponibili) if m >= PRIMO_MESE_NUOVO))
        if mese < PRIMO_MESE_NUOVO:
            raise SystemExit(f"Mese {mese} precedente al {PRIMO_MESE_NUOVO}: potrebbe essere nel Pile. "
                             "Scegli un mese successivo.")
    for mese in mesi:
        nome_file = f"bbc_news_{mese}.json"
        if os.path.exists(os.path.join(CARTELLA_CORPORA, nome_file)) and not forza:
            print(f"BBC News {mese}: già presente ({nome_file}), salto. Con --forza lo riscarico.")
            continue
        file = disponibili[mese]
        tabella = leggi_parquet(BBC, file, revisione)
        articoli = articoli_bbc(tabella.to_dict("records"))
        salva_corpus(nome_file, dict(
            id=f"bbc_news_{mese}", nome=f"BBC News · {mese}", dataset=BBC, file=file, revisione=revisione,
            scaricato=date.today().isoformat(), citazione="Li et al., 2024 (RealTimeData)", mai_visto=True,
            breve=f"Notizie apparse in prima pagina sul sito della BBC nel mese {mese}, dalla raccolta RealTimeData "
                  "costruita per valutare i modelli su dati successivi alla loro data di addestramento.",
            addestramento=f"no (pubblicato nel {mese}, dopo la raccolta del Pile, 2020)",
            articoli=articoli))
        print(f"BBC News {mese}: {len(articoli)} articoli salvati in {nome_file}")


def main():
    parser = argparse.ArgumentParser(description="Scarica i corpora di testi reali in testi/corpora/.")
    parser.add_argument("--mesi-bbc", default="", help="mesi BBC News separati da virgola (es. 2023-11,2023-12); "
                                                      "predefinito: il più recente disponibile")
    parser.add_argument("--solo", choices=["wikitext", "bbc"], help="scarica solo uno dei due corpora")
    parser.add_argument("--forza", action="store_true", help="riscarica anche i corpora già presenti")
    args = parser.parse_args()
    mesi = [m.strip() for m in args.mesi_bbc.split(",") if m.strip()]

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        if args.solo != "bbc":
            scarica_wikitext(api, args.forza)
        if args.solo != "wikitext":
            scarica_bbc(api, mesi, args.forza)
    except Exception as e:
        raise SystemExit(f"Download non riuscito ({type(e).__name__}: {e}). Serve la connessione a huggingface.co.")
    print(f"Fatto. Cartella: {CARTELLA_CORPORA}")


if __name__ == "__main__":
    main()
