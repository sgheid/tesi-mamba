# Verifica la ricorrenza simulata della Fase 5 (nucleo.ricorrenza.ricorrenza_in_formato):
#  1. con il formato FP32, B1 e B2 danno logit identici bit a bit al modello originale (torch.equal);
#  2. la funzione sostituita è chiamata in ogni strato e, con FP16/BF16, i logit cambiano davvero;
#  3. la funzione originale di transformers è ripristinata all'uscita, anche dopo un errore dentro il "with";
#  4. i pesi del modello non cambiano (impronta).
# Uso:  python prove/prova_ricorrenza.py [--modello 130m] [--gpu]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nucleo.precisione import impronta
from nucleo.ricorrenza import FORMATI_RICORRENZA, LIVELLI, _modulo_mamba, ricorrenza_in_formato
from prove.fotografa_fasi import contesto


def logit(ctx):
    with torch.no_grad():
        return ctx.model(ctx.ids, use_cache=False).logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modello", default="130m")
    p.add_argument("--gpu", action="store_true")
    arg = p.parse_args()
    dispositivo = "cuda" if arg.gpu and torch.cuda.is_available() else "cpu"
    ctx = contesto(arg.modello, dispositivo)
    n = len(ctx.model.backbone.layers)
    modulo = _modulo_mamba()
    originale = modulo.mamba_selective_scan
    prima = impronta(ctx.model)
    errori = []
    print(f"{ctx.info_modello['nome']} su {dispositivo}, {ctx.n_token} token")

    riferimento = logit(ctx)
    for livello in LIVELLI:
        with ricorrenza_in_formato(torch.float32, livello) as strati:
            simulati = logit(ctx)
        uguali = torch.equal(simulati, riferimento)
        print(f"  {livello} in FP32: logit identici bit a bit = {uguali}, strati con la ricorrenza sostituita = "
              f"{len(strati)} di {n}")
        if not uguali:
            errori.append(f"{livello} FP32: max |differenza| {(simulati - riferimento).abs().max().item():.3e}")
        if len(strati) != n:
            errori.append(f"{livello} FP32: sostituta chiamata in {len(strati)} strati su {n}")
        if modulo.mamba_selective_scan is not originale:
            errori.append(f"{livello}: funzione originale non ripristinata")

        for nome, tipo in FORMATI_RICORRENZA.items():
            with ricorrenza_in_formato(tipo, livello) as strati:
                simulati = logit(ctx)
            diff = (simulati - riferimento).abs().max().item()
            print(f"  {livello} in {nome}: max |Δ logit| = {diff:.3e}, poli nuovi su 1 = "
                  f"{sum(s['nuovi_1'] for s in strati):,}, stato non finito = {sum(s['non_finiti_stato'] for s in strati)}")
            if not diff > 0:
                errori.append(f"{livello} {nome}: logit invariati, la sostituzione non ha effetto")

    try:
        with ricorrenza_in_formato(torch.float16, "B2"):
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

    print("\nTutte le verifiche superate." if not errori else "\nPROBLEMI:\n" + "\n".join(errori))
    sys.exit(1 if errori else 0)


if __name__ == "__main__":
    main()
