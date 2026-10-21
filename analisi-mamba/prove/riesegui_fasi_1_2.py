# Riesegue le Fasi 1, 2a e 2b sul testo di esempio, senza Streamlit, e salva i numeri per strato in JSON.
# Uso:  python prove/riesegui_fasi_1_2.py uscita.json [--confronta riferimento.json] [--modello 130m] [--cpu]
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fasi import fase1_poli_continui, fase2a_delta, fase2b_poli_discreti
from nucleo.modelli import carica_e_verifica, cartella_cache_predefinita, trova_modelli
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


def numeri(ris):
    return dict(righe=ris.righe, indicatori=[(i.etichetta, i.valore) for i in ris.indicatori])


def uguali(x, y):
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("uscita")
    p.add_argument("--confronta")
    p.add_argument("--modello", default="130m")
    p.add_argument("--cpu", action="store_true")
    arg = p.parse_args()

    import torch
    dispositivo = "cpu" if arg.cpu or not torch.cuda.is_available() else "cuda"
    ctx = contesto(arg.modello, dispositivo)
    tutti = {f.FASE.codice: numeri(f.FASE.esegui(ctx)) for f in (fase1_poli_continui, fase2a_delta, fase2b_poli_discreti)}
    with open(arg.uscita, "w", encoding="utf-8") as f:
        json.dump(tutti, f, ensure_ascii=False, indent=1)
    print(f"{ctx.info_modello['nome']} su {dispositivo}, {ctx.n_token} token: salvato {arg.uscita}")

    if arg.confronta:
        with open(arg.confronta, encoding="utf-8") as f:
            rif = json.load(f)
        tutti = json.loads(json.dumps(tutti))
        diverse = 0
        for codice, dati in rif.items():
            for r_rif, r_nuova in zip(dati["righe"], tutti[codice]["righe"]):
                for k, v in r_rif.items():
                    if not uguali(v, r_nuova.get(k)):
                        diverse += 1
                        print(f"  Fase {codice}, strato {r_rif['strato']}, {k}: {v} -> {r_nuova.get(k)}")
            if dati["indicatori"] != tutti[codice]["indicatori"]:
                diverse += 1
                print(f"  Fase {codice}: indicatori diversi")
        print("Numeri identici al riferimento." if diverse == 0 else f"{diverse} differenze.")


if __name__ == "__main__":
    main()
