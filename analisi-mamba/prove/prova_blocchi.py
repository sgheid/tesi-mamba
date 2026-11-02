# Ricorrenza a blocchi di token (nucleo/scansione.py) identica bit a bit a quella intera, sul modello vero e sul testo
# di esempio: logit del modello FP32 e ricorrenza simulata della Fase 5 (B1, B2 in FP16, BF16 e FP32), con blocchi
# imposti di 1, 7, 50 e 100 token contro il calcolo intero. Al termine la funzione di transformers è quella installata.
# Uso:  python prove/prova_blocchi.py [--gpu] [--modello 130m]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nucleo import scansione
from nucleo.modelli import carica_e_verifica, cartella_cache_predefinita, trova_modelli
from nucleo.ricorrenza import ricorrenza_in_formato
from nucleo.testo import TESTO_ESEMPIO, prepara


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modello", default="130m")
    p.add_argument("--gpu", action="store_true")
    arg = p.parse_args()
    dispositivo = "cuda" if arg.gpu and torch.cuda.is_available() else "cpu"
    modello = next(m for m in trova_modelli(cartella_cache_predefinita()) if m.is_mamba and arg.modello in m.nome)
    model, tok, info, controlli = carica_e_verifica(modello, dispositivo)
    if model is None:
        raise SystemExit(f"Caricamento fallito: {controlli[-1]}")
    ids = prepara(tok, TESTO_ESEMPIO, dispositivo)[0]
    print(f"{info['nome']} su {dispositivo}, {ids.shape[1]} token")

    def logit():
        with torch.no_grad():
            return model(ids, use_cache=False).logits

    formati = [(t, l) for t in (torch.float16, torch.bfloat16, torch.float32) for l in ("B1", "B2")]
    os.environ.pop(scansione.VARIABILE_PROVA, None)
    riferimento = logit()
    rif_ricorrenza = {}
    for tipo, livello in formati:
        with ricorrenza_in_formato(tipo, livello) as strati:
            rif_ricorrenza[tipo, livello] = (logit(), list(strati))

    tutto = True
    try:
        for blocco in (1, 7, 50, 100):
            os.environ[scansione.VARIABILE_PROVA] = str(blocco)
            esiti = [torch.equal(logit(), riferimento)]
            for (tipo, livello), (l0, s0) in rif_ricorrenza.items():
                with ricorrenza_in_formato(tipo, livello) as strati:
                    l1 = logit()
                esiti.append(torch.equal(l1, l0) and list(strati) == s0)
            tutto &= all(esiti)
            nomi = ["modello"] + [f"{str(t)[6:]} {l}" for t, l in formati]
            print(f"  blocchi da {blocco:>3} token: " + " · ".join(f"{n} {'=' if e else 'DIVERSO'}"
                                                              for n, e in zip(nomi, esiti)))
    finally:
        os.environ.pop(scansione.VARIABILE_PROVA, None)
    from transformers.models.mamba import modeling_mamba
    ripristinata = modeling_mamba.mamba_selective_scan is scansione._a_blocchi
    print("Funzione di transformers ripristinata dopo la ricorrenza simulata: " + ("sì" if ripristinata else "NO"))
    print("Tutto identico bit a bit." if tutto and ripristinata else "DIFFERENZE: non usare i risultati a blocchi.")


if __name__ == "__main__":
    main()
