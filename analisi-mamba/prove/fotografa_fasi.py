# Fotografia dei risultati: esegue ogni fase di fasi/ in tutte le combinazioni dei parametri sul testo di esempio
# e salva in JSON righe per strato, indicatori, evidenze e formato; con --confronta segnala ogni differenza.
# Le fasi con il parametro "Confronto" girano anche nel controllo FP32 contro FP32, la Fase 5 con la ricorrenza in FP32:
# controlli che non compaiono nel menu e devono dare errori nulli.
# Uso:  python prove/fotografa_fasi.py cartella_uscita [--confronta cartella_riferimento] [--fasi 1,2]
#                                      [--modello 130m] [--gpu]
import argparse
import itertools
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fasi import carica_fasi
from nucleo.confronto import CONTROLLO
from nucleo.modelli import carica_e_verifica, cartella_cache_predefinita, trova_modelli
from nucleo.ricorrenza import CONTROLLO as CONTROLLO_RICORRENZA
from nucleo.risultato import Contesto
from nucleo.testo import TESTO_ESEMPIO, prepara


def contesto(nome, dispositivo):
    modello = next(m for m in trova_modelli(cartella_cache_predefinita()) if m.is_mamba and nome in m.nome)
    model, tok, info, controlli = carica_e_verifica(modello, dispositivo)
    if model is None:
        raise SystemExit(f"Caricamento fallito: {controlli[-1]}")
    ids, token, testo, _ = prepara(tok, TESTO_ESEMPIO, dispositivo)
    return Contesto(model=model, tok=tok, info_modello=info, dispositivo=dispositivo, nome_dispositivo=dispositivo,
                    ids=ids, token=token, testo=testo, fonte_testo="testo di esempio")


def uguali(x, y):
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


# Le scelte del menu più il controllo FP32 nascosto: FP32 contro FP32 per il parametro "Confronto",
# ricorrenza in FP32 per il "Formato della ricorrenza" della Fase 5
CONTROLLI = {"confronto": CONTROLLO, "formato_ricorrenza": CONTROLLO_RICORRENZA}


def scelte(parametro):
    return parametro.scelte + ([CONTROLLI[parametro.chiave]] if parametro.chiave in CONTROLLI else [])


# NaN e infiniti come stringhe, per un JSON valido
def _pulisci(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _pulisci(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_pulisci(v) for v in x]
    return x.item() if hasattr(x, "item") else x


def fotografia(ris):
    return _pulisci(dict(
        titolo=ris.titolo, note_formato=ris.note_formato, righe=ris.righe,
        indicatori=[dict(etichetta=i.etichetta, valore=i.valore, nota=i.nota, stato=i.stato) for i in ris.indicatori],
        evidenzia=ris.evidenzia))


def nome_file(fase, parametri):
    return "_".join([fase.codice] + [re.sub(r"\W+", "-", str(v)).strip("-") for v in parametri.values()]) + ".json"


# Differenze campo per campo tra due fotografie (uguaglianza esatta, NaN = NaN)
def differenze(rif, nuova, percorso=""):
    if isinstance(rif, dict) and isinstance(nuova, dict):
        for k in sorted(set(rif) | set(nuova)):
            yield from differenze(rif.get(k), nuova.get(k), f"{percorso}.{k}" if percorso else k)
    elif isinstance(rif, list) and isinstance(nuova, list) and len(rif) == len(nuova):
        for k, (a, b) in enumerate(zip(rif, nuova)):
            yield from differenze(a, b, f"{percorso}[{k}]")
    elif not uguali(rif, nuova):
        yield f"{percorso}: {rif!r} -> {nuova!r}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("uscita")
    p.add_argument("--confronta")
    p.add_argument("--fasi", help="codici separati da virgole (predefinito: tutte)")
    p.add_argument("--modello", default="130m")
    p.add_argument("--gpu", action="store_true")
    arg = p.parse_args()

    import torch
    dispositivo = "cuda" if arg.gpu and torch.cuda.is_available() else "cpu"
    ctx = contesto(arg.modello, dispositivo)
    fasi, errori = carica_fasi()
    if errori:
        raise SystemExit("Fasi non caricate: " + ", ".join(nome for nome, _ in errori))
    os.makedirs(arg.uscita, exist_ok=True)
    print(f"{ctx.info_modello['nome']} su {dispositivo}, {ctx.n_token} token")

    diverse = 0
    if arg.fasi:
        fasi = [f for f in fasi if f.codice in arg.fasi.split(",")]
    for fase in fasi:
        for combinazione in itertools.product(*[scelte(q) for q in fase.parametri]):
            ctx.parametri = {q.chiave: v for q, v in zip(fase.parametri, combinazione)}
            dati = fotografia(fase.esegui(ctx))
            nome = nome_file(fase, ctx.parametri)
            with open(os.path.join(arg.uscita, nome), "w", encoding="utf-8") as f:
                json.dump(dati, f, ensure_ascii=False, indent=1)
            esito = ""
            if arg.confronta:
                with open(os.path.join(arg.confronta, nome), encoding="utf-8") as f:
                    elenco = list(differenze(json.load(f), dati))
                diverse += len(elenco)
                esito = " identica" if not elenco else f" {len(elenco)} differenze"
                for d in elenco[:20]:
                    print(f"    {d}")
            print(f"  {nome}{esito}")
    if arg.confronta:
        print("Tutte identiche al riferimento." if diverse == 0 else f"{diverse} differenze in totale.")


if __name__ == "__main__":
    main()
