# Verifica le soglie di arrotondamento della Fase 5 (abar FP32 portato in FP16/BF16):
#  1. le soglie trovate per bisezione da nucleo.ricorrenza coincidono con quelle attese (2^-12 / 2^-9 verso 1,
#     2^-25 / 2^-134 verso 0);
#  2. prova esaustiva: per OGNI valore FP32 in [0.5, 1] vale  x.to(formato) == 1  <=>  1 - x <= soglia_1, e per ogni
#     valore FP32 in [soglia_0 / 8, soglia_0 * 8] vale  x.to(formato) == 0  <=>  x <= soglia_0;
#  3. i casi esattamente a metà (arrotondamento al pari) e i valori FP32 adiacenti.
# Uso:  python prove/prova_soglie.py [--gpu]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nucleo.ricorrenza import FORMATI_RICORRENZA, SOGLIE_ATTESE, potenza, soglie_verificate


def bit(x):
    return int(torch.tensor([x], dtype=torch.float32).view(torch.int32).item())


def valori_tra(da, a, dispositivo):
    return torch.arange(bit(da), bit(a) + 1, dtype=torch.int32, device=dispositivo).view(torch.float32)


def adiacenti(x, dispositivo):
    t = torch.tensor([x], dtype=torch.float32)
    return [torch.nextafter(t, torch.tensor([-1.0])).item(), x, torch.nextafter(t, torch.tensor([2.0])).item()]


def prova(nome, tipo, dispositivo):
    errori = []
    s1, s0 = soglie_verificate(tipo, dispositivo)
    attese = SOGLIE_ATTESE[tipo]
    print(f"{nome} su {dispositivo}: verso 1 margine <= {potenza(s1)} (atteso {potenza(attese[0])}), "
          f"verso 0 valore <= {potenza(s0)} (atteso {potenza(attese[1])})")
    if (s1, s0) != attese:
        errori.append(f"{nome}: soglie {s1, s0} diverse dalle attese {attese}")

    x = valori_tra(0.5, 1.0, dispositivo)
    diretto = x.to(tipo) == 1
    da_soglia = (1.0 - x) <= s1
    if not torch.equal(diretto, da_soglia):
        errori.append(f"{nome}: {int((diretto != da_soglia).sum())} valori in [0.5, 1] non seguono la soglia verso 1")
    print(f"  esaustiva verso 1: {x.numel():,} valori FP32, {int(diretto.sum()):,} vanno a 1")

    x = valori_tra(s0 / 8, s0 * 8, dispositivo)
    diretto = x.to(tipo) == 0
    da_soglia = x <= s0
    if not torch.equal(diretto, da_soglia):
        errori.append(f"{nome}: {int((diretto != da_soglia).sum())} valori vicino a 0 non seguono la soglia verso 0")
    print(f"  esaustiva verso 0: {x.numel():,} valori FP32 tra soglia/8 e soglia·8, {int(diretto.sum()):,} vanno a 0")

    for etichetta, centro, atteso in (("1 - soglia", 1.0 - attese[0], [False, True, True]),
                                      ("soglia_0", attese[1], [True, True, False])):
        casi = adiacenti(centro, "cpu")
        if etichetta == "1 - soglia":
            ottenuto = [bool(torch.tensor([v], device=dispositivo).to(tipo).item() == 1) for v in casi]
        else:
            ottenuto = [bool(torch.tensor([v], device=dispositivo).to(tipo).item() == 0) for v in casi]
        print(f"  {etichetta}: sotto/esatto/sopra -> {ottenuto} (atteso {atteso})")
        if ottenuto != atteso:
            errori.append(f"{nome}: casi a metà attorno a {etichetta} = {ottenuto}, attesi {atteso}")
    return errori


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    arg = p.parse_args()
    dispositivi = ["cpu"] + (["cuda"] if arg.gpu and torch.cuda.is_available() else [])
    errori = []
    for dispositivo in dispositivi:
        for nome, tipo in FORMATI_RICORRENZA.items():
            errori += prova(nome, tipo, dispositivo)
    print("\nTutte le soglie verificate." if not errori else "\nPROBLEMI:\n" + "\n".join(errori))
    sys.exit(1 if errori else 0)


if __name__ == "__main__":
    main()
