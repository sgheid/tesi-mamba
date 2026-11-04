# Verifica la Fase 6 (nucleo/diagnostica_b2.py):
#  1. la ricorrenza generica con (stato, ingresso) = (FP32, FP32) dà logit identici bit a bit a B1 della Fase 5,
#     con (ridotto, ridotto) identici a B2, in FP16 e BF16: le varianti intermedie usano la stessa aritmetica;
#  2. con il formato FP32 tutte e quattro le varianti danno logit identici al modello originale;
#  3. il passaggio con le ricorrenze ombra lascia l'uscita del modello FP32 identica bit a bit;
#  4. la funzione originale di transformers è ripristinata all'uscita, anche dopo un errore dentro il "with";
#  5. i pesi del modello non cambiano (impronta).
# Uso:  python prove/prova_diagnostica_b2.py [--modello 130m] [--gpu]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nucleo import diagnostica_b2 as db
from nucleo import ricorrenza as rc
from nucleo.precisione import impronta


def logit(ctx):
    with torch.no_grad():
        return ctx.model(ctx.ids, use_cache=False).logits


def verifica(ctx):
    n = len(ctx.model.backbone.layers)
    modulo = rc._modulo_mamba()
    originale = modulo.mamba_selective_scan
    prima = impronta(ctx.model)
    errori = []
    riferimento = logit(ctx)

    for nome, tipo in rc.FORMATI_RICORRENZA.items():
        for livello in rc.LIVELLI:
            with rc.ricorrenza_in_formato(tipo, livello):
                fase5 = logit(ctx)
            with db.ricorrenza_variante(tipo, livello) as strati:
                generica = logit(ctx)
            uguali = torch.equal(fase5, generica)
            print(f"  {livello} in {nome}: generica identica alla Fase 5 = {uguali}, strati = {len(strati)} di {n}")
            if not uguali:
                errori.append(f"{livello} {nome}: max |differenza| {(fase5 - generica).abs().max().item():.3e}")
        for v in db.INTERMEDIE:
            with db.ricorrenza_variante(tipo, v):
                diff = (logit(ctx) - riferimento).abs().max().item()
            print(f"  {v} in {nome}: max |Δ logit| dal modello FP32 = {diff:.3e}")
            if not diff > 0:
                errori.append(f"{v} {nome}: logit invariati, la sostituzione non ha effetto")

    for v in db.VARIANTI:
        with db.ricorrenza_variante(torch.float32, v):
            uguali = torch.equal(logit(ctx), riferimento)
        print(f"  {v} in FP32: logit identici al modello = {uguali}")
        if not uguali:
            errori.append(f"{v} in FP32: logit diversi dal modello")

    for nome, tipo in rc.FORMATI_RICORRENZA.items():
        with db.ricorrenze_ombra(tipo) as strati:
            ombra = logit(ctx)
        uguali = torch.equal(ombra, riferimento)
        print(f"  ricorrenze ombra in {nome}: uscita FP32 identica = {uguali}, strati = {len(strati)} di {n}, "
              f"modi integratori rilevanti = {sum(s['modi_S'] for s in strati):,}")
        if not uguali or len(strati) != n:
            errori.append(f"ricorrenze ombra {nome}: uscita cambiata o strati mancanti")

    if modulo.mamba_selective_scan is not originale:
        errori.append("funzione originale non ripristinata")
    try:
        with db.ricorrenze_ombra(torch.float16):
            raise KeyboardInterrupt("errore simulato")
    except KeyboardInterrupt:
        pass
    ripristinata = modulo.mamba_selective_scan is originale
    print(f"  dopo un errore dentro il with: funzione originale ripristinata = {ripristinata}")
    if not ripristinata:
        errori.append("funzione originale non ripristinata dopo un errore")
    if not torch.equal(logit(ctx), riferimento):
        errori.append("logit del modello cambiati dopo le prove")
    if impronta(ctx.model) != prima:
        errori.append("pesi del modello cambiati")
    return errori


def main():
    from prove.fotografa_fasi import contesto
    p = argparse.ArgumentParser()
    p.add_argument("--modello", default="130m")
    p.add_argument("--gpu", action="store_true")
    arg = p.parse_args()
    dispositivo = "cuda" if arg.gpu and torch.cuda.is_available() else "cpu"
    ctx = contesto(arg.modello, dispositivo)
    print(f"{ctx.info_modello['nome']} su {dispositivo}, {ctx.n_token} token")
    errori = verifica(ctx)
    print("\nTutte le verifiche superate." if not errori else "\nPROBLEMI:\n" + "\n".join(errori))
    sys.exit(1 if errori else 0)


if __name__ == "__main__":
    main()
