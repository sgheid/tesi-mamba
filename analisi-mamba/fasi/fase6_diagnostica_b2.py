import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import confronto as cf
from nucleo import diagnostica_b2 as db
from nucleo import grafica as g
from nucleo import ricorrenza as rc
from nucleo import uscita as u
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Parametro, Risultato, gerarchia, valori

SPIEGAZIONE = r"""
**Fase extra: non parte con "Esegui tutte le fasi".** Serve a spiegare un risultato della Fase 5: nel caso limite
**B2** (ā, $\Delta B u$ e stato nel formato ridotto) sbaglia **meno** di **B1** (solo ā ridotto), anche se arrotonda di più.
I pesi restano sempre in FP32, come nella Fase 5.

**Le due spiegazioni candidate.** I modi che il formato porta a $\bar a = 1$ hanno Δ piccolo, quindi ricevono incrementi
$\bar b x$ piccoli. In B1 lo stato FP32 li somma tutti e la deriva cresce con i token. In B2 possono andare persi in due modi:
- **(a) assorbimento**: l'incremento è più piccolo di metà della spaziatura del formato attorno a $h$, e $h + \bar b x$
  si arrotonda di nuovo a $h$. *Esempio FP16:* con $h \approx 1$ la spaziatura è $2^{-10} \approx 9{,}8\cdot10^{-4}$; un
  incremento di $3\cdot10^{-4}$ è sotto la metà ($4{,}9\cdot10^{-4}$) e va perso: in B1 dopo 2000 token lo stato sarebbe
  derivato di $2000 \cdot 3\cdot10^{-4} = 0{,}6$, in B2 di 0.
- **(b) azzeramento**: $\Delta B u$ va a 0 già prima della somma, per underflow (in FP16 sotto $2^{-25}$; in BF16 quasi mai,
  perché ha la gamma di FP32). Polo 1 e ingresso 0 sono un **modo congelato**, che non accumula nulla.

**Cosa misura.**
1. **Ablazione (decisiva).** Due varianti intermedie end-to-end: **B1 + stato** (ā e $h$ ridotti, $\Delta B u$ in FP32)
   isola l'assorbimento; **B1 + ingresso** (ā e $\Delta B u$ ridotti, $h$ in FP32) isola l'azzeramento. Quella che si
   avvicina a B2 indica la causa. B1 e B2 sono quelli della Fase 5, ricalcolati sullo stesso testo.
2. **Contatori.** In un passaggio FP32, in ogni strato e sugli stessi ingressi, girano "in ombra" le quattro ricorrenze
   ridotte. Ogni aggiornamento (passo × canale × stato) è classificato: **assorbito** (incremento ≠ 0, stato arrotondato
   invariato), **azzerato** (incremento esatto ≠ 0, ridotto = 0) o normale; separatamente per gli aggiornamenti in cui il
   polo è un **integratore spurio** ($\bar a_{32} < 1$, $\bar a_r = 1$) e per tutti gli altri.
3. **Errore di stato** sui modi diventati integratori con memoria non trascurabile sul testo ($|a|\theta \ge 0{,}1$, come
   nelle Fasi 3 e 5): $\sum |h_v - h_{32}| / \sum |h_{32}|$ lungo il testo. Atteso: in B1 cresce circa come $t$
   (pendenza 1), in B2 si appiattisce se l'ipotesi (a) è giusta.

**Come si legge.** B1 + stato ≈ B2 e assorbiti concentrati sugli integratori spuri → assorbimento confermato.
B1 + ingresso ≈ B2 e azzerati prevalenti → è di nuovo il congelamento, lato ingresso. Nessuna delle due → resta un
punto aperto, con due cause escluse.

**Nota.** Nei contatori l'errore di uno strato non si propaga ai successivi, perché l'uscita resta quella FP32: è
l'effetto locale della ricorrenza. L'ablazione invece è end-to-end, come la KL della Fase 5.
"""

PARAMETRO = Parametro("formato_ricorrenza", "Formato della ricorrenza", list(rc.FORMATI_RICORRENZA), "FP16",
                      "Formato in cui arrotondare ā e, a seconda della variante, ΔBu e lo stato. Pesi in FP32.")

NOMI = list(db.VARIANTI)
COLORI = {"B1": g.ARANCIO_CHIARO, "B1 + stato": g.BLU_CHIARO, "B1 + ingresso": g.VERDE_STATO, "B2": g.ARANCIO}
CHIAVI = {"B1": "b1", "B1 + stato": "bh", "B1 + ingresso": "bu", "B2": "b2"}

COLONNE = [Colonna("strato", "Strato", "", "int"),
           Colonna("modi_S", "Modi integratori rilevanti", "integratore spurio almeno una volta, |a|θ ≥ 0,1", "int"),
           Colonna("ass_I_b2", "Assorbiti, integratori", "B2 · % degli aggiornamenti", "perc"),
           Colonna("ass_O_b2", "Assorbiti, altri poli", "B2 · % degli aggiornamenti", "perc"),
           Colonna("persa_I_b2", "Massa persa, integratori", "B2 · % di Σ|ΔBu|", "perc"),
           Colonna("ass_I_bh", "Assorbiti, integratori", "B1 + stato · %", "perc"),
           Colonna("azz_I", "Azzerati, integratori", "ingresso ridotto · %", "perc"),
           Colonna("azz_O", "Azzerati, altri poli", "ingresso ridotto · %", "perc")] + [
           Colonna(f"err_{CHIAVI[v]}", "Errore di stato a fine testo", f"{v} · relativo, modi rilevanti", "sci")
           for v in NOMI]

LEGENDA = [("Modi integratori rilevanti", "Modi (canale, stato) con almeno un passo in cui il formato porta a 1 un polo "
                                          "che in FP32 è < 1, e con |a|θ ≥ 0,1 (memoria non trascurabile sul testo)."),
           ("Assorbiti", "Aggiornamenti con incremento ≠ 0 in cui lo stato arrotondato non cambia: "
                         "fl(ā·h + inc) = fl(ā·h). Solo nelle varianti con lo stato ridotto (B1 + stato, B2)."),
           ("Massa persa", "Σ|ΔBu esatto − incremento applicato| / Σ|ΔBu esatto|: quota dell'ingresso che non entra nello "
                           "stato, per assorbimento totale o parziale e per l'arrotondamento dell'ingresso."),
           ("Azzerati", "Aggiornamenti con ΔBu ≠ 0 in FP32 e = 0 nel formato ridotto (underflow). Uguali per B1 + ingresso "
                        "e B2, che usano lo stesso ingresso ridotto."),
           ("Integratori / altri poli", "Integratori: aggiornamenti in cui ā32 < 1 e ā arrotondato = 1 (polo che diventa 1 "
                                        "in quel passo). Altri: tutti gli altri aggiornamenti."),
           ("Errore di stato a fine testo", "Σ|h_variante − h32| / Σ|h32| sui modi integratori rilevanti dello strato, "
                                            "all'ultimo token, con gli ingressi del modello FP32 (effetto locale).")]

SECONDARI = dict(
    indicatori=("Passaggio FP32 durante i contatori", "Assorbiti, B1 + stato", "Logit non finiti", "Stato non finito",
                "Accordo top-1", "Modi integratori rilevanti"),
    colonne=("ass_O_b2", "ass_I_bh", "azz_O", "err_b1", "err_bh", "err_bu"),
    grafici=("Assorbiti per strato",))


# Percentuale anche per i gruppi vuoti (NaN): fmt_perc non accetta NaN
def _perc(x):
    return "-" if not math.isfinite(x) else fmt_perc(x)


def _frazione(c, chiave):
    return c[chiave] / c["n"] if c["n"] else float("nan")


def _persa(c):
    return c["persa"] / c["massa"] if c["massa"] else float("nan")


def _unisci(gruppi_strati, variante, gruppo):
    tot = dict(n=0, assorbiti=0, azzerati=0, massa=0.0, persa=0.0)
    for gr in gruppi_strati:
        for k in tot:
            tot[k] += gr[variante][gruppo][k]
    return tot


def esegui(ctx):
    nome = ctx.parametri.get("formato_ricorrenza", "FP16")
    tipo = rc.TIPI[nome]
    W32 = ctx.model.lm_head.weight

    # 1. Contatori e errore di stato: un passaggio FP32 con le ricorrenze ombra
    h32 = u.stato_finale_fp32(cf.tratto(ctx, 0.0, 0.05))
    ctx.avanzamento(0.05, "contatori: passaggio FP32 con le quattro ricorrenze ridotte in ombra")
    h_ombra, strati = db.passaggio_ombra(ctx, tipo)
    fp32_intatto = torch.equal(h_ombra, h32)

    # 2. Ablazione end-to-end: B1 e B2 della Fase 5, varianti intermedie da nucleo/diagnostica_b2.py
    uscite = {}
    for k, v in enumerate(NOMI):
        da = 0.45 + 0.135 * k
        ctx.avanzamento(da, f"{v}: passaggio con la ricorrenza simulata")
        if v in rc.LIVELLI:
            h, info = rc.stato_finale_simulato(ctx, tipo, v)
        else:
            h, info = db.stato_finale_variante(ctx, tipo, v)
        m = u.metriche(h32, W32, ctx.ids, h, W32, cf.tratto(ctx, da + 0.1, da + 0.135).avanzamento)
        uscite[v] = dict(m=m, s=u.sintesi(m), non_finiti=sum(x["non_finiti_stato"] for x in info))

    righe = []
    for i, st in enumerate(strati):
        gr = st["gruppi"]
        r = dict(strato=i, modi_S=st["modi_S"], modi_spuri=st["modi_spuri"], modi_totali=st["modi_totali"],
                 ass_I_b2=_frazione(gr["B2"]["I"], "assorbiti"), ass_O_b2=_frazione(gr["B2"]["O"], "assorbiti"),
                 persa_I_b2=_persa(gr["B2"]["I"]), persa_O_b2=_persa(gr["B2"]["O"]),
                 ass_I_bh=_frazione(gr["B1 + stato"]["I"], "assorbiti"),
                 azz_I=_frazione(gr["B2"]["I"], "azzerati"), azz_O=_frazione(gr["B2"]["O"], "azzerati"),
                 agg_I=gr["B2"]["I"]["n"])
        for k, v in enumerate(NOMI):
            r[f"err_{CHIAVI[v]}"] = float(st["num"][k, -1] / st["den"][-1]) if st["den"][-1] > 0 else float("nan")
        righe.append(r)

    # Errore di stato aggregato su tutti gli strati, per posizione
    num = sum(st["num"] for st in strati)
    den = sum(st["den"] for st in strati)
    errore = {v: np.where(den > 0, num[k] / np.where(den > 0, den, 1), np.nan) for k, v in enumerate(NOMI)}

    return gerarchia(Risultato(
        codice="6",
        titolo=f"Fase 6 - Diagnostica B2 < B1: ricorrenza in {nome}",
        descrizione="Ablazione end-to-end (B1, B1 + stato, B1 + ingresso, B2) e contatori di incrementi assorbiti e "
                    "azzerati con le ricorrenze ridotte in ombra su un passaggio FP32. Pesi in FP32.",
        colonne=COLONNE,
        righe=righe,
        indicatori=_indicatori(uscite, strati, righe, errore, fp32_intatto, nome, ctx),
        legenda=LEGENDA,
        grafici=[_grafico_kl(uscite, nome), _grafico_errore(errore, nome), _grafico_assorbiti(righe, nome)],
        evidenzia=_evidenze(righe),
        note_formato=f"pesi FP32, ā arrotondato a {nome}; stato e/o ΔBu ridotti secondo la variante · dispositivo: "
                     f"{ctx.nome_dispositivo}",
    ), **SECONDARI)


def _evidenze(righe):
    con = [r for r in righe if r["modi_S"]]
    peggiori = sorted(con, key=lambda r: -r["modi_S"])[:3]
    return {r["strato"]: f"n. {k} per modi integratori rilevanti" for k, r in enumerate(peggiori, start=1)}


# Quota del guadagno di B2 su B1 (in decadi di KL media) spiegata da ciascuna variante intermedia
def _quote(kl):
    g2 = math.log10(kl["B1"] / kl["B2"]) if kl["B1"] > 0 and kl["B2"] > 0 else float("nan")
    quote = {v: (math.log10(kl["B1"] / kl[v]) / g2 if g2 and math.isfinite(g2) and kl[v] > 0 else float("nan"))
             for v in db.INTERMEDIE}
    return g2, quote


def _verdetto(g2, quote):
    qs, qi = quote["B1 + stato"], quote["B1 + ingresso"]
    if not math.isfinite(g2) or g2 <= 0:
        return "B2 non è sotto B1 su questo testo", "attenzione"
    if qs >= 0.7 and qi < 0.3:
        return "lo stato ridotto (assorbimento)", "ok"
    if qi >= 0.7 and qs < 0.3:
        return "l'ingresso ridotto (azzeramento)", "ok"
    if qs >= 0.3 and qi >= 0.3:
        return "entrambi gli arrotondamenti", None
    return "nessuno dei due da solo", "attenzione"


def _indicatori(uscite, strati, righe, errore, fp32_intatto, nome, ctx):
    kl = {v: x["s"]["kl_media"] for v, x in uscite.items()}
    g2, quote = _quote(kl)
    testo, stato = _verdetto(g2, quote)
    fmt_q = lambda q: "-" if not math.isfinite(q) else f"{100 * q:.0f} %"
    gruppi = [st["gruppi"] for st in strati]
    b2_I, b2_O = _unisci(gruppi, "B2", "I"), _unisci(gruppi, "B2", "O")
    bh_I, bh_O = _unisci(gruppi, "B1 + stato", "I"), _unisci(gruppi, "B1 + stato", "O")
    modi_S, modi_tot = sum(st["modi_S"] for st in strati), sum(st["modi_totali"] for st in strati)
    pend = {v: u.pendenza_loglog(e) for v, e in errore.items()}
    pend_kl = {v: u.pendenza_loglog(x["m"]["kl"]) for v, x in uscite.items()}
    fmt_p = lambda p: "-" if not math.isfinite(p) else f"{p:+.2f}"
    fine = {v: (e[np.isfinite(e)][-1] if np.isfinite(e).any() else float("nan")) for v, e in errore.items()}
    non_finiti_logit = {v: x["s"]["non_finite"] for v, x in uscite.items()}
    non_finiti_stato = {v: x["non_finiti"] for v, x in uscite.items()}
    elenco = lambda d, f: " · ".join(f(d[v]) for v in NOMI)
    controllo = []
    if nome == rc.CONTROLLO:
        ok = fp32_intatto and all(x["s"]["identiche"] and x["s"]["kl_max"] == 0 for x in uscite.values()) and all(
            np.nanmax(e, initial=0) == 0 for e in errore.values())
        controllo = [Indicatore("Controllo FP32", "superato" if ok else "NON superato",
                                "ricorrenza in FP32: le quattro varianti danno KL nulla e stato identico a FP32",
                                "ok" if ok else "attenzione")]
        testo, stato = "controllo FP32: nessun guadagno da spiegare", None
    return controllo + [
        Indicatore("Ricorrenza in", nome, "caso limite, come la Fase 5 · pesi sempre in FP32", "attenzione"),
        Indicatore("Causa di B2 < B1 (lettura dell'ablazione)", testo,
                   f"quota del guadagno di B2 spiegata: B1 + stato {fmt_q(quote['B1 + stato'])} · B1 + ingresso "
                   f"{fmt_q(quote['B1 + ingresso'])} · guadagno di B2 = {g2:.2f} decadi di KL media"
                   if math.isfinite(g2) else "KL non confrontabili", stato),
        Indicatore("KL media: " + " · ".join(NOMI), elenco(kl, fmt_sci),
                   "nat · rapporto con B1: " + " · ".join(
                       f"{kl[v] / kl['B1']:.2f}" if kl["B1"] > 0 else "-" for v in NOMI)),
        Indicatore("Pendenza della KL lungo il testo", elenco({v: p[0] for v, p in pend_kl.items()}, fmt_p),
                   "nell'ordine " + " · ".join(NOMI) + " · circa 2 = accumulo come t² (Fase 5)"),
        Indicatore("Assorbiti in B2: integratori · altri poli",
                   f"{_perc(_frazione(b2_I, 'assorbiti'))} · {_perc(_frazione(b2_O, 'assorbiti'))}",
                   f"aggiornamenti con incremento perso del tutto · massa persa {_perc(_persa(b2_I))} · "
                   f"{_perc(_persa(b2_O))} · {fmt_int(b2_I['n'])} aggiornamenti su integratori spuri"),
        Indicatore("Azzerati dall'ingresso ridotto: integratori · altri poli",
                   f"{_perc(_frazione(b2_I, 'azzerati'))} · {_perc(_frazione(b2_O, 'azzerati'))}",
                   "ΔBu ≠ 0 in FP32 e = 0 nel formato (underflow) · uguali per B1 + ingresso e B2"),
        Indicatore("Errore di stato a fine testo, modi rilevanti", elenco(fine, fmt_sci),
                   "Σ|h_v − h32| / Σ|h32| su tutti gli strati, ultimo token · nell'ordine " + " · ".join(NOMI)),
        Indicatore("Crescita dell'errore di stato (pendenza)", elenco({v: p[0] for v, p in pend.items()}, fmt_p),
                   "log errore contro log posizione · circa 1 = deriva lineare (integratore), circa 0 = errore che "
                   "non si accumula"),
        Indicatore("Modi integratori rilevanti", fmt_int(modi_S),
                   f"{fmt_perc(modi_S / modi_tot) if modi_tot else '-'} dei modi (canale, stato) · "
                   f"integratori spuri almeno una volta: {fmt_int(sum(st['modi_spuri'] for st in strati))}"),
        Indicatore("Assorbiti, B1 + stato: integratori · altri poli",
                   f"{_perc(_frazione(bh_I, 'assorbiti'))} · {_perc(_frazione(bh_O, 'assorbiti'))}",
                   f"massa persa {_perc(_persa(bh_I))} · {_perc(_persa(bh_O))}"),
        Indicatore("Accordo top-1", elenco({v: x["s"]["top1"] for v, x in uscite.items()}, fmt_perc),
                   "nell'ordine " + " · ".join(NOMI)),
        Indicatore("Logit non finiti", elenco(non_finiti_logit, fmt_int), "posizioni escluse dalle metriche",
                   "ok" if not any(non_finiti_logit.values()) else "attenzione"),
        Indicatore("Stato non finito", elenco(non_finiti_stato, fmt_int), "valori inf/NaN dello stato, tutti i passi",
                   "ok" if not any(non_finiti_stato.values()) else "attenzione"),
        Indicatore("Passaggio FP32 durante i contatori", "identico" if fp32_intatto else "NON identico",
                   "le ricorrenze ombra non cambiano l'uscita del modello FP32 (confronto bit a bit)",
                   "ok" if fp32_intatto else "attenzione"),
    ]


def _grafico_kl(uscite, nome):
    serie = [(v, x["m"]["kl"], COLORI[v]) for v, x in uscite.items()]
    fig = u.figura_posizioni("KL(P_FP32 ‖ P_variante)  (nat, scala logaritmica)", serie, log_y=True)
    dati = pd.DataFrame({"posizione": np.arange(len(serie[0][1]))})
    for v, kl, _ in serie:
        dati[f"KL {v}"], dati[f"KL {v}, media mobile su {u.FINESTRA}"] = kl, u.media_mobile(kl)
    return Grafico(
        f"Ablazione: KL lungo il testo con la ricorrenza in {nome}", fig,
        "Distanza dal modello FP32 per le quattro varianti della ricorrenza ridotta, a ogni posizione.",
        f"""
- **Arancione chiaro** = B1 (solo $\\bar a$), **arancione** = B2 (anche $\\Delta B u$ e stato), **azzurro** = B1 + stato,
  **verde** = B1 + ingresso. Linee = media mobile su {u.FINESTRA} posizioni.
- La variante intermedia che si sovrappone a B2 indica quale arrotondamento produce il guadagno di B2 su B1.
- **Esempio di lettura:** se l'azzurra coincide con l'arancione e la verde con l'arancione chiaro, il guadagno viene
  tutto dallo stato ridotto, cioè dall'assorbimento; l'ingresso ridotto non conta.
""",
        dati)


def _grafico_errore(errore, nome):
    serie = [(v, e, COLORI[v]) for v, e in errore.items()]
    fig = u.figura_posizioni("Σ|h_v − h32| / Σ|h32|  (scala logaritmica)", serie, log_y=True)
    dati = pd.DataFrame({"posizione": np.arange(len(serie[0][1]))})
    for v, e, _ in serie:
        dati[f"errore di stato {v}"] = e
    return Grafico(
        f"Errore di stato sui modi diventati integratori, ricorrenza in {nome}", fig,
        "Deriva dello stato rispetto a FP32, sommata su tutti gli strati, sui modi integratori con memoria non "
        "trascurabile sul testo. Ingressi del modello FP32: effetto locale della ricorrenza.",
        """
- Un modo che dovrebbe dimenticare e invece integra accumula una differenza che cresce **linearmente** con i token:
  in scala logaritmica è una retta di pendenza 1 contro il log della posizione.
- Se la curva di B2 (e di B1 + stato) si appiattisce mentre quella di B1 sale, l'arrotondamento dello stato frena la
  deriva: gli incrementi piccoli vengono assorbiti.
- **Esempio di lettura:** B1 da $10^{-3}$ a 500 token a $4\\cdot10^{-3}$ a 2000 token (cresce come $t$), B2 fermo
  a $10^{-3}$: lo stato ridotto ha tolto la deriva.
""",
        dati)


def _grafico_assorbiti(righe, nome):
    x = valori(righe, "strato")
    fig = g.nuova_figura("Strato", "Aggiornamenti assorbiti in B2  (%)", hover_unificato=False)
    for chiave, etichetta, colore in (("ass_I_b2", "su integratori spuri", g.ARANCIO), ("ass_O_b2", "su altri poli", g.BLU)):
        fig.add_trace(go.Bar(x=x, y=valori(righe, chiave), name=etichetta, marker=dict(color=colore, line_width=0),
                             hovertemplate=f"strato %{{x}}: %{{y:.2%}} {etichetta}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    g.asse_strati(fig, len(x))
    return Grafico(
        f"Assorbiti per strato in B2, ricorrenza in {nome}", fig,
        "Quota degli aggiornamenti in cui l'incremento va perso nella somma, separando i poli che il formato porta a 1.",
        """
- **Arancione** = aggiornamenti su integratori spuri ($\\bar a_{32} < 1$, $\\bar a_r = 1$); **blu** = tutti gli altri.
- L'ipotesi dell'assorbimento prevede barre arancioni molto più alte delle blu: gli integratori spuri hanno Δ piccolo e
  ricevono incrementi piccoli.
- **Esempio di lettura:** 40 % contro 1 % in uno strato: in quello strato quasi metà degli incrementi dei poli su 1 non
  entra nello stato.
""",
        pd.DataFrame({"strato": x, "assorbiti integratori": valori(righe, "ass_I_b2"),
                      "assorbiti altri": valori(righe, "ass_O_b2"), "massa persa integratori": valori(righe, "persa_I_b2"),
                      "massa persa altri": valori(righe, "persa_O_b2"), "aggiornamenti su integratori": valori(righe, "agg_I"),
                      "modi integratori rilevanti": valori(righe, "modi_S")}))


FASE = Fase(
    codice="6",
    nome="Extra · diagnostica B2 < B1",
    breve="perché nel caso limite B2 sbaglia meno di B1: ablazione e incrementi assorbiti/azzerati (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[PARAMETRO],
    extra=True,
)
