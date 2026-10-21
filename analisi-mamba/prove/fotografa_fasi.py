# Fotografia dei risultati: esegue ogni fase di fasi/ in tutte le combinazioni dei parametri sul testo di esempio
# e salva in JSON righe per strato, indicatori, evidenze e formato; con --confronta segnala ogni differenza.
# Uso:  python prove/fotografa_fasi.py cartella_uscita [--confronta cartella_riferimento] [--modello 130m] [--gpu]
import argparse
import itertools
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fasi import carica_fasi
from prove.riesegui_fasi_1_2 import contesto, uguali


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
    for fase in fasi:
        for scelte in itertools.product(*[q.scelte for q in fase.parametri]):
            ctx.parametri = {q.chiave: v for q, v in zip(fase.parametri, scelte)}
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
