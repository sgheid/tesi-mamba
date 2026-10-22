import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import confronto as cf
from nucleo import grafica as g
from nucleo import ricorrenza as rc
from nucleo import uscita as u
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.margini import ERR_MAX, ERR_MIN, N_ERR, mediana_errore
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Parametro, Risultato, valori

SPIEGAZIONE = r"""
**Attenzione: è un caso limite, non il comportamento dei runtime reali.** In `transformers` (e nei kernel di
`mamba-ssm`) il polo $\bar a = e^{\Delta a}$ e lo stato $h$ restano in FP32 anche quando i pesi sono in FP16 o BF16:
l'abbiamo verificato nel codice installato. Questa fase è un'**analisi di sensibilità**: chiede cosa succederebbe se
la ricorrenza stessa fosse in bassa precisione. I pesi restano sempre in FP32, così l'effetto misurato viene solo
dalla ricorrenza.

**Cosa misura.** *Parte A*: quanti poli $\bar a$, calcolati in FP32 come nel modello, cambiano classe se
arrotondati a FP16 o BF16. Vicino a 1 il passo tra numeri rappresentabili è $2^{-11}$ in FP16 e $2^{-8}$ in BF16,
quindi ogni $\bar a$ con margine $1-\bar a \le 2^{-12}$ (FP16) o $\le 2^{-9}$ (BF16) diventa **esattamente 1**: il modo
smette di dimenticare. Queste soglie sono verificate a ogni esecuzione sul dispositivo. *Parte B*: l'effetto
sull'uscita del modello quando la ricorrenza gira davvero nel formato ridotto (B1: solo $\bar a$ arrotondato;
B2: anche $\Delta B u$ e lo stato a ogni passo), con le stesse metriche della Fase 4.

**Perché conta.** In FP32 la soglia è $2^{-25} \approx 3\cdot10^{-8}$ e i poli su 1 sono rari. In FP16 la soglia sale
a $2^{-12} \approx 2{,}4\cdot10^{-4}$: tutti i modi con memoria oltre circa 4 000 token diventano integratori. In BF16
($2^{-9} \approx 2\cdot10^{-3}$) basta una memoria di circa 500 token.

**Esempio.** Un modo con $1-\bar a = 10^{-4}$ dimentica in circa $1/10^{-4} = 10\,000$ token. In FP16 il suo
$\bar a = 0{,}9999$ si arrotonda a 1, perché $10^{-4} < 2^{-12} \approx 2{,}44\cdot10^{-4}$: lo stato accumula
senza fine. La colonna "memoria persa" riporta proprio questo $\tau = 1/(1-\bar a)$ dei poli che finiscono su 1.
"""

PARAMETRO = Parametro("formato_ricorrenza", "Formato della ricorrenza", list(rc.FORMATI_RICORRENZA), "FP16",
                      "Formato in cui arrotondare abar (Parte A e B1) e anche deltaB_u e lo stato (B2). "
                      "I pesi restano in FP32.")

COLONNE = [Colonna("strato", "Strato", "", "int"),
           Colonna("su_1_fp32", "Poli su 1 in FP32", "riferimento, come Fase 3", "int"),
           Colonna("nuovi_1", "Nuovi su 1", "abar32 < 1, arrotondato = 1", "int"),
           Colonna("perc_nuovi_1", "Nuovi su 1", "% dello strato", "perc"),
           Colonna("tau_mediana", "Memoria persa", "mediana di 1/(1-abar), token", "sci"),
           Colonna("tau_max", "Memoria persa", "massimo, token", "sci"),
           Colonna("nuovi_0", "Nuovi a 0", "abar32 > 0, arrotondato = 0", "int"),
           Colonna("err_mediana", "Errore tipico sul margine", "mediana, relativo", "sci"),
           Colonna("non_finiti", "Non finiti", "inf/NaN", "int")]

LEGENDA = [("Poli su 1 in FP32", "Poli (token × canali × stati) con abar = 1 già in FP32, con l'exp del dispositivo: "
                                 "lo stesso conteggio della Fase 3."),
           ("Nuovi su 1", "Poli con abar < 1 in FP32 che il formato arrotonda a esattamente 1: margine 1 - abar sotto "
                          "la soglia verificata (2^-12 in FP16, 2^-9 in BF16)."),
           ("Memoria persa", "tau = 1/(1 - abar) dei poli nuovi su 1, con il margine esatto (FP64) che avevano in FP32: "
                             "quanti token ci mettevano a dimenticare prima di diventare integratori."),
           ("Nuovi a 0", "Poli con abar > 0 in FP32 che il formato porta a 0 (sotto 2^-25 in FP16; in BF16 solo sotto "
                         "2^-134): modi senza memoria, innocui per la stabilità."),
           ("Errore tipico sul margine", "Mediana di |(1 - abar_r)/(1 - abar32) - 1| sui poli non già su 1; per i nuovi "
                                         "su 1 vale 1. Istogramma a centesimi di decade."),
           ("Non finiti", "Poli inf/NaN dopo l'arrotondamento: contati ed esclusi.")]


def esegui(ctx):
    nome = ctx.parametri.get("formato_ricorrenza", "FP16")
    controllo = nome == rc.CONTROLLO
    soglie = rc.verifica_soglie(ctx.dispositivo)
    nomi = list(rc.FORMATI_RICORRENZA) + ([nome] if controllo else [])
    riferimento, per_formato = rc.poli_arrotondati(cf.tratto(ctx, 0.0, 0.4), nomi,
                                                   {k: (soglie[k]["soglia_1"], soglie[k]["soglia_0"]) for k in nomi})
    uscite = _parte_b(ctx, rc.TIPI[nome])
    righe = [dict(r, su_1_fp32=r32["su_1"], nuovi_1_b1=b1["nuovi_1"], stato_max_b2=b2["stato_max"],
                  non_finiti_stato_b2=b2["non_finiti_stato"])
             for r, r32, b1, b2 in zip(per_formato[nome], riferimento, uscite["B1"]["strati"], uscite["B2"]["strati"])]
    n = len(righe)
    somma = lambda chiave, elenco=righe: sum(valori(elenco, chiave))
    totale, nuovi = somma("totale"), somma("nuovi_1")
    s = soglie[nome]
    soglie_ok = all(v["ok"] for v in soglie.values())
    dal_conteggio = all(somma("nuovi_1", per_formato[k]) == somma("nuovi_1_soglia", per_formato[k])
                        and somma("nuovi_0", per_formato[k]) == somma("nuovi_0_soglia", per_formato[k]) for k in nomi)
    isto_tau = sum(r["isto_tau"] for r in righe)
    tau_max = max((r["tau_max"] for r in righe if r["nuovi_1"]), default=float("nan"))
    err_tipico = mediana_errore(somma("err_nulli"), sum(r["isto_err"] for r in righe))
    peggiori = [r for r in sorted(righe, key=lambda r: -r["nuovi_1"]) if r["nuovi_1"]][:3]

    indicatori = [
        Indicatore("Ricorrenza in", nome, "caso limite: transformers tiene abar e stato in FP32 · pesi sempre in FP32",
                   "attenzione"),
        Indicatore("Soglie verificate", f"1 − ā ≤ {rc.potenza(s['soglia_1'])} → 1 · ā ≤ {rc.potenza(s['soglia_0'])} → 0",
                   f"ricavate sul dispositivo ({ctx.nome_dispositivo}) " + ("uguali alle attese per FP16 e BF16" if soglie_ok
                                                                           else "DIVERSE dalle attese"),
                   "ok" if soglie_ok else "attenzione"),
        Indicatore("Poli nuovi su 1", fmt_int(nuovi),
                   f"{fmt_perc(nuovi / totale)} dei poli · " + " · ".join(
                       f"{k}: {fmt_int(somma('nuovi_1', per_formato[k]))}" for k in rc.FORMATI_RICORRENZA),
                   "attenzione" if nuovi else "ok"),
        Indicatore("Conteggio dalle soglie", "coincide" if dal_conteggio else "NON coincide",
                   "nuovi su 1 e a 0 contati sia arrotondando sia con le soglie verificate",
                   "ok" if dal_conteggio else "attenzione"),
        Indicatore("Memoria persa (τ dei nuovi su 1)",
                   f"mediana {fmt_sci(rc.mediana_tau(isto_tau))} · max {fmt_sci(tau_max)}",
                   "token che quei modi impiegavano a dimenticare in FP32"),
        Indicatore("Strati con più poli nuovi", ", ".join(str(r["strato"]) for r in peggiori) or "nessuno",
                   " · ".join(f"{r['strato']}: {fmt_int(r['nuovi_1'])}" for r in peggiori)),
        Indicatore("Poli nuovi a 0", fmt_int(somma("nuovi_0")), "senza memoria, innocui per la stabilità"),
        Indicatore("Errore tipico sul margine 1 − ā", fmt_sci(err_tipico), "mediana su tutti i poli non già su 1"),
        Indicatore("Poli su 1 in FP32 (riferimento)", fmt_int(sum(r["su_1"] for r in riferimento)),
                   f"exp del dispositivo · margine esatto < 2⁻²⁵: {fmt_int(sum(r['su_1_esatto'] for r in riferimento))}"),
        Indicatore("Non finiti", fmt_int(somma("non_finiti")), "nessuno" if somma("non_finiti") == 0 else "esclusi",
                   "ok" if somma("non_finiti") == 0 else "attenzione"),
        Indicatore("Dispositivo", ctx.nome_dispositivo, "in FP32 la GPU porta a 1 più poli della CPU (vedi Fase 3)"),
    ]
    indicatori += _indicatori_b(uscite, righe, nome)
    if controllo:
        ok = (nuovi == 0 and somma("nuovi_0") == 0 and somma("non_finiti") == 0 and err_tipico == 0
              and all(v["s"]["identiche"] and v["s"]["kl_max"] == 0 and v["s"]["top1"] == 1 for v in uscite.values()))
        indicatori.append(Indicatore("Controllo FP32", "superato" if ok else "NON superato",
                                     "abar arrotondato a FP32: nessun polo cambia; B1 e B2 danno KL nulla e NLL identiche "
                                     "bit a bit", "ok" if ok else "attenzione"))

    return Risultato(
        codice="5",
        titolo=f"Fase 5 - Caso limite: ricorrenza in {nome}",
        descrizione="Parte A: abar = exp(Delta·a) in FP32 come il modello, arrotondato al formato; margini esatti in FP64. "
                    "Parte B: uscita con la ricorrenza simulata (B1, B2) contro il modello FP32, metriche della Fase 4.",
        colonne=COLONNE + COLONNE_B,
        righe=[{k: v for k, v in r.items() if not k.startswith("isto_")} for r in righe],
        indicatori=indicatori,
        legenda=LEGENDA + LEGENDA_B,
        grafici=[_grafico_nuovi(riferimento, per_formato, nome), _grafico_memoria(righe, nome),
                 _grafico_errore(per_formato, nome), _grafico_nll_b(uscite, nome), _grafico_kl_b(uscite, nome),
                 _grafico_istogramma_b(uscite, nome), _grafico_stato(righe, nome)],
        evidenzia={r["strato"]: f"n. {k} per poli nuovi su 1" for k, r in enumerate(peggiori, start=1)},
        note_formato=f"pesi FP32, abar arrotondato a {nome}; margini in FP64 · dispositivo: {ctx.nome_dispositivo}",
    )


def _colore(k, nome):
    return g.ARANCIO if k == nome else g.ARANCIO_CHIARO


def _grafico_nuovi(riferimento, per_formato, nome):
    x = valori(riferimento, "strato")
    fig = g.nuova_figura("Strato", "Poli  (% dello strato)", hover_unificato=False)
    y32 = [r["su_1"] / r["totale"] for r in riferimento]
    fig.add_trace(go.Bar(x=x, y=y32, name="su 1 già in FP32 (Fase 3)", marker=dict(color=g.BLU, line_width=0),
                         customdata=valori(riferimento, "su_1"),
                         hovertemplate="strato %{x}: %{customdata:,} su 1 in FP32 (%{y:.3%})<extra></extra>"))
    dati = pd.DataFrame({"strato": x, "su 1 in FP32": valori(riferimento, "su_1"), "quota su 1 in FP32": y32})
    for k, righe in per_formato.items():
        fig.add_trace(go.Bar(x=x, y=valori(righe, "perc_nuovi_1"), name=f"nuovi su 1 in {k}",
                             marker=dict(color=_colore(k, nome), line_width=0), customdata=valori(righe, "nuovi_1"),
                             hovertemplate=f"strato %{{x}}: %{{customdata:,}} nuovi su 1 in {k} (%{{y:.3%}})<extra></extra>"))
        dati[f"nuovi su 1 in {k}"], dati[f"quota nuovi su 1 in {k}"] = valori(righe, "nuovi_1"), valori(righe, "perc_nuovi_1")
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    g.asse_strati(fig, len(x))
    return Grafico(
        "Poli nuovi su 1 per arrotondamento, per strato", fig,
        "Quota dei poli di ogni strato che il formato porta a esattamente 1, con i poli già su 1 in FP32 come riferimento.",
        f"""
- **Barra blu** = poli già su 1 in FP32 (lo stesso conteggio della Fase 3). **Barre arancioni** = poli **in più** che
  diventano 1 arrotondando $\\bar a$: arancione pieno = {nome}, arancione chiaro = l'altro formato.
- BF16 ha una soglia 8 volte più alta di FP16 ($2^{{-9}}$ contro $2^{{-12}}$), quindi le sue barre sono più alte.
- Gli strati con barre alte sono quelli che tengono molti modi con memoria lunga.
- **Esempio di lettura:** una barra BF16 al 6% in uno strato da 1536 canali × 16 stati × 211 token vuol dire circa
  311 000 poli che in BF16 smetterebbero di dimenticare, contro una barra blu quasi invisibile in FP32.
""",
        dati)


def _grafico_memoria(righe, nome):
    x = valori(righe, "strato")
    med, mx = valori(righe, "tau_mediana"), valori(righe, "tau_max")
    fig = g.nuova_figura("Strato", "Memoria persa τ = 1/(1 − ā)  (token, scala logaritmica)", log_y=True)
    fig.add_trace(go.Scatter(x=x, y=mx, mode="markers", name="massimo", marker=g.marcatore(7, g.ARANCIO_CHIARO),
                             hovertemplate="massimo %{y:.2e} token<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=med, mode="lines+markers", name="mediana", line=dict(color=g.ARANCIO, width=2.2),
                             marker=g.marcatore(7, g.ARANCIO), hovertemplate="mediana %{y:.2e} token<extra></extra>"))
    finiti = [v for v in med + mx if isinstance(v, float) and math.isfinite(v) and v > 0]
    if finiti:
        g.scala_log(fig, "y", min(finiti), max(finiti))
    else:
        fig.add_annotation(text="Nessun polo nuovo su 1", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                           font=dict(size=14, color=g.TENUE))
    g.asse_strati(fig, len(x))
    return Grafico(
        f"Memoria persa dai poli nuovi su 1 in {nome}, per strato", fig,
        "Per i poli che il formato porta a 1: quanti token impiegavano a dimenticare in FP32 (mediana e massimo).",
        f"""
- Ogni polo che diventa 1 aveva in FP32 una memoria $\\tau = 1/(1-\\bar a)$ finita; dopo l'arrotondamento è infinita.
- Il **minimo** possibile è circa $2^{{12}} \\approx 4\\,096$ token in FP16 e $2^9 = 512$ in BF16: sotto quella memoria il polo
  non viene arrotondato a 1.
- **Esempio di lettura:** mediana $10^4$ e massimo $10^8$ in uno strato: tipicamente diventano integratori modi che
  avrebbero dimenticato dopo circa 10 000 token, ma anche modi che in FP32 duravano cento milioni di token.
- Strati senza punti: nessun polo nuovo su 1.
""",
        pd.DataFrame({"strato": x, "tau mediana (token)": med, "tau massimo (token)": mx,
                      "nuovi su 1": valori(righe, "nuovi_1")}))


def _grafico_errore(per_formato, nome):
    bordi = np.linspace(ERR_MIN, ERR_MAX, N_ERR + 1)
    # A decimi di decade per la lettura: dieci bin dell'istogramma per barra
    bordi10, centri = bordi[::10], (bordi[::10][:-1] + bordi[::10][1:]) / 2
    fig = g.nuova_figura("|Errore relativo| sul margine 1 − ā  (decadi)", "Quota dei poli  (scala logaritmica)",
                         log_y=True, hover_unificato=False)
    dati = pd.DataFrame({"log10|err| da": bordi10[:-1], "log10|err| a": bordi10[1:]})
    quote = []
    for k, righe in per_formato.items():
        isto = sum(r["isto_err"] for r in righe)
        conteggi = isto.reshape(-1, 10).sum(axis=1)
        totale = isto.sum() + sum(valori(righe, "err_nulli"))
        quota = conteggi / totale if totale else conteggi
        vis = quota > 0
        fig.add_trace(go.Bar(x=centri[vis], y=quota[vis], width=0.1 * (0.9 if k == nome else 0.45), name=k,
                             marker=dict(color=_colore(k, nome), line_width=0),
                             hovertemplate=f"{k}: %{{y:.2e}} dei poli<extra></extra>"))
        dati[f"poli {k}"], dati[f"quota {k}"] = conteggi.astype(np.int64), quota
        quote += list(quota[vis])
    fig.update_layout(barmode="overlay")
    g.asse_decadi(fig, "x", ERR_MIN, ERR_MAX)
    if quote:
        g.scala_log(fig, "y", min(quote), max(quote))
    return Grafico(
        "Errore relativo sul margine dovuto all'arrotondamento di ā", fig,
        "Distribuzione di |(1 − ā_r)/(1 − ā32) − 1| su tutti i poli non già su 1, a decimi di decade.",
        f"""
- Il margine $1-\\bar a$ di un polo lontano da 1 cambia di circa mezzo passo del formato diviso il margine: errori
  piccoli per i poli veloci, grandi per quelli lenti.
- La **barra a $10^0$** raccoglie i poli nuovi su 1 (margine azzerato: errore $-1$, cioè 100%).
- **Esempio di lettura:** in FP16 un polo con margine $10^{{-2}}$ ha un errore tipico di circa $2^{{-12}}/10^{{-2}}
  \\approx 2{{,}}4\\%$; con margine $10^{{-3}}$ l'errore sale al 24%. Arancione pieno = {nome}, chiaro = l'altro formato.
""",
        dati)


# Parte B: due passaggi del modello FP32 condiviso con la ricorrenza simulata (B1, B2), confrontati con il passaggio
# FP32 originale con le metriche della Fase 4 (stesso lm_head FP32)
COLONNE_B = [Colonna("nuovi_1_b1", "Nuovi su 1 in B1", "contati nella ricorrenza simulata", "int"),
             Colonna("stato_max_b2", "Stato massimo in B2", "max |h| finito", "sci"),
             Colonna("non_finiti_stato_b2", "Stato non finito in B2", "valori inf/NaN, tutti i passi", "int")]

LEGENDA_B = [("Nuovi su 1 in B1", "Poli nuovi su 1 contati dentro la ricorrenza simulata B1. Nello strato 0 coincide con "
                                  "la Parte A; negli strati successivi può differire di poco, perché l'arrotondamento "
                                  "degli strati precedenti cambia il Delta."),
             ("Stato massimo in B2", "Il più grande |h| finito dello strato su tutti i passi, con lo stato nel formato "
                                     "ridotto. In FP16 oltre 65 504 lo stato diventa infinito."),
             ("Stato non finito in B2", "Valori dello stato inf/NaN, sommati su tutti i passi: con l'overflow il modello "
                                        "produce logit non finiti, contati negli indicatori ed esclusi dalle metriche.")]

NOMI_B = {"B1": "B1: solo ā", "B2": "B2: ā, ΔBu e stato"}
COLORI_B = {"B1": g.ARANCIO_CHIARO, "B2": g.ARANCIO}


def _parte_b(ctx, tipo):
    h32 = u.stato_finale_fp32(cf.tratto(ctx, 0.4, 0.5))
    W32 = ctx.model.lm_head.weight
    uscite = {}
    for k, livello in enumerate(rc.LIVELLI):
        da = 0.5 + 0.25 * k
        ctx.avanzamento(da, f"{NOMI_B[livello]}: passaggio con la ricorrenza simulata")
        h, strati = rc.stato_finale_simulato(ctx, tipo, livello)
        m = u.metriche(h32, W32, ctx.ids, h, W32, cf.tratto(ctx, da + 0.2, da + 0.25).avanzamento)
        uscite[livello] = dict(m=m, s=u.sintesi(m), strati=strati)
    return uscite


def _indicatori_b(uscite, righe, nome):
    s1, s2 = uscite["B1"]["s"], uscite["B2"]["s"]
    tipo = rc.TIPI[nome]
    limite = torch.finfo(tipo).max
    stato_max = max(valori(righe, "stato_max_b2"))
    non_finiti_stato = sum(valori(righe, "non_finiti_stato_b2"))
    coerente = righe[0]["nuovi_1"] == righe[0]["nuovi_1_b1"]
    quarti = {}
    for livello, v in uscite.items():
        kl = v["m"]["kl"]
        q = max(1, len(kl) // 4)
        quarti[livello] = (np.nanmean(kl[:q]), np.nanmean(kl[-q:]))
    rel = lambda a, b: cf.differenza(a, b).replace("differenza relativa ", "") or "="
    return [
        Indicatore("Perplexity FP32 → B1 → B2", f"{s1['ppl']:.4f} → {s1['ppl_r']:.4f} → {s2['ppl_r']:.4f}",
                   f"B1 {rel(s1['ppl'], s1['ppl_r'])} · B2 {rel(s2['ppl'], s2['ppl_r'])} · su "
                   f"{fmt_int(s1['posizioni'])} predizioni"),
        Indicatore("KL media B1 · B2", f"{fmt_sci(s1['kl_media'])} · {fmt_sci(s2['kl_media'])}",
                   f"nat · mediana {fmt_sci(s1['kl_mediana'])} · {fmt_sci(s2['kl_mediana'])} · "
                   f"P99 {fmt_sci(s1['kl_p99'])} · {fmt_sci(s2['kl_p99'])}"),
        Indicatore("KL massima B1 · B2", f"{fmt_sci(s1['kl_max'])} · {fmt_sci(s2['kl_max'])}",
                   f"posizioni {s1['pos_kl_max']} · {s2['pos_kl_max']}"),
        Indicatore("Accordo top-1 B1 · B2", f"{fmt_perc(s1['top1'])} · {fmt_perc(s2['top1'])}",
                   "posizioni con lo stesso token più probabile del modello FP32"),
        Indicatore("ΔNLL media B1 · B2", f"{s1['dnll']:+.2e} · {s2['dnll']:+.2e} nat",
                   "positiva = il token vero riceve meno probabilità che in FP32"),
        Indicatore("KL: primo → ultimo quarto del testo",
                   " · ".join(f"{l} {fmt_sci(a)} → {fmt_sci(b)}" for l, (a, b) in quarti.items()),
                   "un aumento indica errore che si accumula con il contesto"),
        Indicatore("Stato in B2", f"max |h| {fmt_sci(stato_max)} · non finiti {fmt_int(non_finiti_stato)}",
                   f"massimo finito di {nome}: {fmt_sci(limite)}",
                   "ok" if non_finiti_stato == 0 else "attenzione"),
        Indicatore("Logit non finiti B1 · B2", f"{fmt_int(s1['non_finite'])} · {fmt_int(s2['non_finite'])}",
                   "posizioni escluse dalle metriche",
                   "ok" if s1["non_finite"] == s2["non_finite"] == 0 else "attenzione"),
        Indicatore("Coerenza Parte A / B1", "coincide" if coerente else "NON coincide",
                   f"strato 0: {fmt_int(righe[0]['nuovi_1'])} nuovi su 1 nella Parte A, "
                   f"{fmt_int(righe[0]['nuovi_1_b1'])} nella ricorrenza B1", "ok" if coerente else "attenzione"),
    ]


def _grafico_nll_b(uscite, nome):
    m1 = uscite["B1"]["m"]
    serie = [("NLL FP32", m1["nll"], g.BLU)] + [(f"NLL {NOMI_B[l]}", v["m"]["nll_r"], COLORI_B[l])
                                                for l, v in uscite.items()]
    fig = u.figura_posizioni("NLL  (nat)", serie)
    dati = u.dati_posizioni(m1, {"nll": "NLL FP32", "nll_r": "NLL B1"})
    dati["NLL B2"], dati[f"NLL B2, media mobile su {u.FINESTRA}"] = uscite["B2"]["m"]["nll_r"], u.media_mobile(
        uscite["B2"]["m"]["nll_r"])
    return Grafico(
        f"NLL lungo il testo con la ricorrenza in {nome}", fig,
        "−log P(token vero) a ogni posizione: modello FP32 originale e ricorrenza simulata.",
        f"""
- **Blu** = modello FP32 originale; **arancione chiaro** = B1 (solo $\\bar a$ in {nome}); **arancione** = B2 (anche
  $\\Delta B u$ e stato in {nome}). Linee = media mobile su {u.FINESTRA} posizioni.
- Se le linee arancioni si staccano dalla blu **verso destra**, la ricorrenza ridotta peggiora la predizione man mano
  che il testo si allunga: è l'effetto atteso dei modi diventati integratori, che accumulano invece di dimenticare.
- **Esempio di lettura:** blu a 3,0 nat e arancione a 3,3 nat dopo 800 token: la perplexity locale passa da
  $e^{{3{{,}}0}} \\approx 20$ a $e^{{3{{,}}3}} \\approx 27$.
""",
        dati)


def _grafico_kl_b(uscite, nome):
    serie = [(f"KL {NOMI_B[l]}", v["m"]["kl"], COLORI_B[l]) for l, v in uscite.items()]
    fig = u.figura_posizioni("KL(P_FP32 ‖ P_simulato)  (nat, scala logaritmica)", serie, log_y=True)
    if not any(np.nanmax(v["m"]["kl"], initial=0) > 0 for v in uscite.values()):
        fig.add_annotation(text="KL nulla in ogni posizione (controllo FP32)", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    dati = u.dati_posizioni(uscite["B1"]["m"], {"kl": "KL B1"})
    dati["KL B2"], dati[f"KL B2, media mobile su {u.FINESTRA}"] = uscite["B2"]["m"]["kl"], u.media_mobile(
        uscite["B2"]["m"]["kl"])
    return Grafico(
        f"KL dal modello FP32 lungo il testo, ricorrenza in {nome}", fig,
        "Distanza tra la distribuzione del modello FP32 e quella con la ricorrenza simulata, a ogni posizione.",
        f"""
- **Asse y logaritmico.** Arancione chiaro = B1, arancione = B2; linee = media mobile su {u.FINESTRA} posizioni.
- **Media mobile in salita** = errore che si accumula con il contesto: con la ricorrenza ridotta è il comportamento
  atteso, perché i poli portati a 1 sommano tutto il passato invece di dimenticarlo.
- La distanza tra B1 e B2 dice quanto pesa arrotondare anche $\\Delta B u$ e lo stato, oltre ad $\\bar a$.
- **Esempio di lettura:** sul 130M col testo di prova, con la ricorrenza in BF16 la media mobile di B1 sale da circa
  $10^{{-5}}$ nel primo quarto a $2\\cdot10^{{-4}}$ nell'ultimo: l'errore cresce di oltre venti volte con il contesto.
  Nella Fase 4 (pesi ridotti, ricorrenza in FP32) la KL resta invece piatta lungo il testo.
""",
        dati)


def _grafico_istogramma_b(uscite, nome):
    fig, dati = u.figura_istogramma_kl([(NOMI_B[l], v["m"]["kl"], COLORI_B[l]) for l, v in uscite.items()])
    return Grafico(
        f"Distribuzione della KL sulle posizioni, ricorrenza in {nome}", fig,
        "Quante posizioni hanno KL in ogni quarto di decade (posizioni con KL esattamente 0 escluse).",
        """
- Barre larghe arancione chiaro = B1, barre strette arancioni = B2, sugli stessi intervalli.
- Una distribuzione spostata a destra = distribuzioni più lontane da quelle del modello FP32.
- **Esempio di lettura:** il picco di B1 a $10^{-3}$ e quello di B2 a $10^{-2}$: arrotondare anche lo stato sposta
  di una decade la KL tipica.
""",
        dati)


def _grafico_stato(righe, nome):
    x, y = valori(righe, "strato"), valori(righe, "stato_max_b2")
    limite = torch.finfo(rc.TIPI[nome]).max
    fig = g.nuova_figura("Strato", "Massimo |stato| in B2  (scala logaritmica)", log_y=True)
    fig.add_trace(go.Scatter(x=x, y=[v if v > 0 else None for v in y], mode="lines+markers", name="max |h|",
                             line=dict(color=g.ARANCIO, width=2.2), marker=g.marcatore(7, g.ARANCIO),
                             customdata=valori(righe, "non_finiti_stato_b2"),
                             hovertemplate="max |h| %{y:.2e}<br>%{customdata:,} valori non finiti<extra></extra>"))
    positivi = [v for v in y if v > 0]
    if limite < 1e10:
        g.linea_soglia(fig, "y", limite, f"massimo finito {nome} = {limite:,.0f}".replace(",", " "))
        positivi.append(limite)
    if positivi:
        g.scala_log(fig, "y", min(positivi), max(positivi))
    fig.update_layout(showlegend=False)
    g.asse_strati(fig, len(x))
    return Grafico(
        f"Stato più grande per strato, con lo stato in {nome} (B2)", fig,
        "Il più grande valore assoluto dello stato h su tutti i token e i canali, per strato.",
        f"""
- Con i poli portati a 1 lo stato non decade più e può crescere token dopo token.
- **Linea rossa** (solo FP16) = 65 504, il più grande numero finito di FP16: un punto che la raggiunge vuol dire
  overflow, cioè stato infinito. BF16 ha la stessa gamma di FP32 (circa $3\\cdot10^{{38}}$) e non ha questo problema.
- **Esempio di lettura:** un massimo di $10^3$ in uno strato è lontano dal limite di FP16; ma con un testo dieci volte
  più lungo uno stato che cresce linearmente arriverebbe a $10^4$, vicino all'overflow.
""",
        pd.DataFrame({"strato": x, "max |h| B2": y, "valori non finiti B2": valori(righe, "non_finiti_stato_b2"),
                      "nuovi su 1 B1": valori(righe, "nuovi_1_b1")}))


FASE = Fase(
    codice="5",
    nome="Caso limite: ricorrenza ridotta",
    breve="abar e stato in FP16/BF16 con pesi FP32: poli portati a 1 ed effetto sull'uscita (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[PARAMETRO],
)
