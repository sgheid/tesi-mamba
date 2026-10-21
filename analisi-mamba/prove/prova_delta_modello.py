# Prova di nucleo/sonde.delta_modello e nucleo/precisione.copia_in_formato sul 130M.
# 1. In FP32, FP16 e BF16 il Delta ricalcolato coincide bit a bit con quello interno del modello
#    (catturato dall'uscita di F.softplus durante il forward).
# 2. A_log della copia caricata in un formato coincide con A_log FP32 arrotondato con .to(formato).
# 3. Dopo copia_in_formato il modello FP32 ha gli stessi pesi e produce gli stessi Delta di prima.
# Uso:  python prove/prova_delta_modello.py [--modello 130m] [--cpu]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from nucleo.modelli import _carica_pesi, carica_e_verifica, cartella_cache_predefinita, trova_modelli
from nucleo import precisione
from nucleo.precisione import FORMATI, copia_in_formato
from nucleo.risultato import Contesto
from nucleo.sonde import delta_modello, esegui_con_sonde
from nucleo.testo import TESTO_ESEMPIO, prepara


# Delta di ogni strato: (interno al modello, ricalcolato da delta_modello)
def delta_interni_e_ricalcolati(model, ids):
    interni, ricalcolati = [], []
    originale = F.softplus
    nel_hook = [False]

    # Registra solo le chiamate del modello, non quella fatta da delta_modello dentro l'hook
    def spia(x, *a, **k):
        y = originale(x, *a, **k)
        if not nel_hook[0]:
            interni.append(y[0].T.clone())
        return y

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            nel_hook[0] = True
            try:
                ricalcolati.append(delta_modello(mixer, uscita))
            finally:
                nel_hook[0] = False
        return hook

    F.softplus = spia
    try:
        esegui_con_sonde(model, ids, crea_hook)
    finally:
        F.softplus = originale
    return interni, ricalcolati


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modello", default="130m")
    p.add_argument("--cpu", action="store_true")
    arg = p.parse_args()
    dispositivo = "cpu" if arg.cpu or not torch.cuda.is_available() else "cuda"

    modello = next(m for m in trova_modelli(cartella_cache_predefinita()) if m.is_mamba and arg.modello in m.nome)
    model, tok, info, _ = carica_e_verifica(modello, dispositivo)
    ids = prepara(tok, TESTO_ESEMPIO, dispositivo)[0]
    ctx = Contesto(model=model, tok=tok, info_modello=info, dispositivo=dispositivo, nome_dispositivo=dispositivo, ids=ids)
    fallite = 0

    def esito(ok, testo):
        nonlocal fallite
        fallite += not ok
        print(f"  [{'ok' if ok else 'FALLITA'}] {testo}")

    print(f"{info['nome']} su {dispositivo}, {ids.shape[1]} token")
    _, delta_prima = delta_interni_e_ricalcolati(model, ids)
    for nome, tipo in FORMATI.items():
        print(f"== {nome}")
        copia = _carica_pesi(info["percorso"], dispositivo, tipo).eval()
        tipi = {n: getattr(copia.backbone.layers[0].mixer, n).dtype for n in ("A_log", "D")}
        tipi.update(dt_proj=copia.backbone.layers[0].mixer.dt_proj.weight.dtype,
                    x_proj=copia.backbone.layers[0].mixer.x_proj.weight.dtype)
        print("  dtype:", ", ".join(f"{k} {v}" for k, v in tipi.items()))
        interni, ricalcolati = delta_interni_e_ricalcolati(copia, ids)
        diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(interni, ricalcolati))
        esito(len(interni) == len(ricalcolati) == len(copia.backbone.layers)
              and all(torch.equal(a, b) for a, b in zip(interni, ricalcolati)),
              f"Delta interno = delta_modello in tutti gli strati ({ricalcolati[0].dtype}), differenza massima {diff}")
        esito(all(torch.equal(lc.mixer.A_log, lm.mixer.A_log.to(tipo))
                  for lc, lm in zip(copia.backbone.layers, model.backbone.layers)),
              "A_log caricato = A_log FP32 arrotondato con .to(formato)")
        del copia, interni, ricalcolati

    # Entrambe le strategie: FP32 su CPU e FP32 riletto dal disco (forzata con un margine enorme)
    for margine, atteso in ((precisione.MARGINE_GB, None), (1e9, "disco")):
        precisione.MARGINE_GB = margine
        print(f"== copia_in_formato (margine {margine} GB)")
        with copia_in_formato(ctx, torch.float16) as (copia, info_copia):
            esito(next(copia.parameters()).dtype == torch.float16,
                  f"copia in FP16, modo {info_copia['modo']}, FP32 su {next(model.parameters()).device}")
        if atteso and dispositivo == "cuda":
            esito(info_copia["modo"] == atteso, f"modo {atteso}")
        esito(info_copia["fp32_intatto"] and next(model.parameters()).device.type == dispositivo,
              f"modello FP32 intatto e di nuovo su {dispositivo}")
        esito(model.lm_head.weight is model.backbone.embeddings.weight, "lm_head ancora legato agli embedding")
        _, delta_dopo = delta_interni_e_ricalcolati(model, ids)
        esito(all(torch.equal(a, b) for a, b in zip(delta_prima, delta_dopo)), "Delta FP32 identici a prima")

    print("Tutte le prove superate." if fallite == 0 else f"{fallite} prove fallite.")
    sys.exit(1 if fallite else 0)


if __name__ == "__main__":
    main()
