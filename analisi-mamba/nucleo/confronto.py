import math
from dataclasses import replace

from nucleo.formato import fmt_int, fmt_perc
from nucleo.risultato import Colonna, Indicatore, Parametro

# Parametro "Confronto" delle fasi: riferimento FP32 da solo o affiancato a una copia con i pesi in FP16/BF16
SOLO_FP32 = "Solo FP32"
CONFRONTI = {SOLO_FP32: None, "FP32 vs FP16": "FP16", "FP32 vs BF16": "BF16"}
# Non compare nel menu: lo usa prove/fotografa_fasi.py, con errori che devono essere esattamente nulli
CONTROLLO = "FP32 vs FP32 (controllo)"

PARAMETRO = Parametro("confronto", "Confronto", list(CONFRONTI), SOLO_FP32,
                      "Solo FP32: il modello caricato, senza seconda copia. FP32 vs FP16/BF16: affianca una copia "
                      "con i pesi nel formato ridotto, letta dal disco.")


# Nome del formato ridotto (chiave di nucleo.precisione.FORMATI), None per "Solo FP32"
def formato_confronto(ctx):
    valore = ctx.parametri.get("confronto", SOLO_FP32)
    return "FP32 (controllo)" if valore == CONTROLLO else CONFRONTI[valore]


# Contesto con l'avanzamento ristretto all'intervallo [da, a] della barra
def tratto(ctx, da, a):
    return replace(ctx, avanzamento=lambda f, m="": ctx.avanzamento(da + (a - da) * f, m))


def _segno(x):
    return "+" if x >= 0 else "−"


def differenza(v32, vr):
    if not all(isinstance(v, (int, float)) for v in (v32, vr)):
        return ""
    if v32 == vr:
        return "nessuna differenza"
    if isinstance(v32, int) and isinstance(vr, int):
        return f"differenza {_segno(vr - v32)}{fmt_int(abs(vr - v32))}"
    if v32 != 0 and math.isfinite(v32) and math.isfinite(vr):
        rel = (vr - v32) / abs(v32)
        return f"differenza relativa {_segno(rel)}{fmt_perc(abs(rel))}"
    return ""


# Indicatore affiancato: "valore FP32 → valore ridotto", con la differenza nella nota
def indicatore(etichetta, v32, vr, fmt, nome, nota="", stato=None):
    note = [f"FP32 → {nome}", differenza(v32, vr), nota]
    return Indicatore(etichetta, f"{fmt(v32)} → {fmt(vr)}", " · ".join(n for n in note if n), stato)


# Ogni colonna FP32 seguita dalla stessa colonna nel formato ridotto (chiave con suffisso _r)
def colonne_affiancate(colonne, nome):
    elenco = []
    for c in colonne:
        if c.chiave == "strato":
            elenco.append(c)
            continue
        sotto = lambda f: f"{c.sottotitolo} · {f}" if c.sottotitolo else f
        elenco += [Colonna(c.chiave, c.titolo, sotto("FP32"), c.formato),
                   Colonna(c.chiave + "_r", c.titolo, sotto(nome), c.formato)]
    return elenco


# Righe per strato unite: FP32, formato ridotto (_r) e metriche del confronto; le chiavi in comune devono coincidere
def righe_unite(righe_fp32, righe_rid, righe_confronto):
    unite = []
    for r32, rr, rc in zip(righe_fp32, righe_rid, righe_confronto):
        riga = dict(r32, **{k + "_r": v for k, v in rr.items() if k != "strato"})
        for k, v in rc.items():
            if k in riga and riga[k] != v:
                raise ValueError(f"Strato {r32['strato']}: '{k}' diverso tra riferimento ({riga[k]}) e confronto ({v})")
            riga[k] = v
        unite.append(riga)
    return unite


def evidenze_unite(*evidenze):
    unite = {}
    for e in evidenze:
        for strato, nota in e.items():
            unite[strato] = f"{unite[strato]} · {nota}" if strato in unite else nota
    return unite
