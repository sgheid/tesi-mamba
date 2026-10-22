# Verifica che le fasi riorganizzate per grandezza diano gli stessi numeri delle sei fasi vecchie.
# "Solo FP32" deve coincidere in tutto (righe, indicatori, evidenze, formato) con la vecchia fase FP32;
# con un confronto ogni campo delle righe della vecchia fase FP32 e della vecchia 3x deve avere lo stesso valore,
# gli indicatori della 3x devono essere uguali e quelli FP32 devono iniziare con il vecchio valore ("valore → ...").
# Uso:  python prove/verifica_fusione.py cartella_nuova prove/fotografia_fasi_precedenti --fasi 1,2,3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prove.fotografa_fasi import differenze, uguali

# nuova fase -> (vecchia fase FP32, vecchia fase di confronto)
MAPPA = {"1": ("1", "3a"), "2": ("2a", "3b"), "3": ("2b", "3c")}
CONFRONTI = {"FP32-vs-FP16": "FP16", "FP32-vs-BF16": "BF16", "FP32-vs-FP32-controllo": "FP32-controllo"}


def leggi(cartella, nome):
    with open(os.path.join(cartella, nome + ".json"), encoding="utf-8") as f:
        return json.load(f)


def verifica_confronto(nuova, v32, vx):
    problemi, controllati = [], 0
    for r_nuova, r32, rx in zip(nuova["righe"], v32["righe"], vx["righe"]):
        for origine, r in (("FP32", r32), ("confronto", rx)):
            for k, v in r.items():
                controllati += 1
                if not uguali(v, r_nuova.get(k)):
                    problemi.append(f"righe, strato {r['strato']}, {k} ({origine}): {v!r} -> {r_nuova.get(k)!r}")
    if len(nuova["righe"]) != len(v32["righe"]):
        problemi.append("numero di strati diverso")

    per_etichetta = {i["etichetta"]: i for i in nuova["indicatori"]}
    for i in vx["indicatori"]:
        controllati += 1
        n = per_etichetta.get(i["etichetta"])
        if n is None:
            problemi.append(f"indicatore '{i['etichetta']}' (confronto) mancante")
        elif n != i and i["etichetta"] not in (j["etichetta"] for j in v32["indicatori"]):
            problemi.append(f"indicatore '{i['etichetta']}': {i} -> {n}")
    for i in v32["indicatori"]:
        controllati += 1
        n = per_etichetta.get(i["etichetta"])
        if n is None:
            problemi.append(f"indicatore '{i['etichetta']}' (FP32) mancante")
        elif n["valore"] != i["valore"] and not n["valore"].startswith(i["valore"] + " → "):
            problemi.append(f"indicatore '{i['etichetta']}': valore FP32 {i['valore']!r} -> {n['valore']!r}")

    for vecchia in (v32, vx):
        for strato, nota in vecchia["evidenzia"].items():
            controllati += 1
            if nota not in nuova["evidenzia"].get(strato, ""):
                problemi.append(f"evidenza strato {strato}: {nota!r} assente in {nuova['evidenzia'].get(strato)!r}")
    return problemi, controllati


def main():
    p = argparse.ArgumentParser()
    p.add_argument("nuova")
    p.add_argument("vecchia")
    p.add_argument("--fasi", default="1,2,3")
    arg = p.parse_args()

    totale = 0
    for codice in arg.fasi.split(","):
        vecchia_32, vecchia_x = MAPPA[codice]
        v32 = leggi(arg.vecchia, vecchia_32)
        nuova = leggi(arg.nuova, f"{codice}_Solo-FP32")
        # Tutto ciò che c'era deve restare identico; colonne e indicatori aggiunti dopo vengono solo elencati
        vecchie_chiavi = set(v32["righe"][0])
        vecchie_etichette = {i["etichetta"] for i in v32["indicatori"]}
        ridotta = dict(righe=[{k: v for k, v in r.items() if k in vecchie_chiavi} for r in nuova["righe"]],
                       indicatori=[i for i in nuova["indicatori"] if i["etichetta"] in vecchie_etichette],
                       evidenzia=nuova["evidenzia"], note_formato=nuova["note_formato"])
        elenco = list(differenze({k: v32[k] for k in ridotta}, ridotta))
        aggiunte = sorted(set(nuova["righe"][0]) - vecchie_chiavi) + [
            i["etichetta"] for i in nuova["indicatori"] if i["etichetta"] not in vecchie_etichette]
        print(f"Fase {codice} · Solo FP32 contro vecchia {vecchia_32}: "
              + ("identica (righe, indicatori, evidenze, formato)" if not elenco else f"{len(elenco)} differenze")
              + (f"; aggiunti: {', '.join(aggiunte)}" if aggiunte else ""))
        for d in elenco[:20]:
            print(f"    {d}")
        totale += len(elenco)

        for suffisso, formato in CONFRONTI.items():
            nuova = leggi(arg.nuova, f"{codice}_{suffisso}")
            problemi, controllati = verifica_confronto(nuova, v32, leggi(arg.vecchia, f"{vecchia_x}_{formato}"))
            print(f"Fase {codice} · {suffisso} contro vecchie {vecchia_32} + {vecchia_x} {formato}: "
                  + (f"identica ({controllati} valori controllati)" if not problemi else f"{len(problemi)} differenze"))
            for d in problemi[:20]:
                print(f"    {d}")
            totale += len(problemi)
            print(f"    indicatori: " + " | ".join(f"{i['etichetta']} = {i['valore']}" for i in nuova["indicatori"]))
    print("Nessuna differenza." if totale == 0 else f"{totale} differenze in totale.")


if __name__ == "__main__":
    main()
