import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import confronto as cf
from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import FORMATI, descrivi_formato
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Risultato, valori

SOGLIA = 0.01  # un polo è "molto perturbato" se |a| cambia di più dell'1%
PASSO = 0.25   # binning di |a| a quarti di decade

SPIEGAZIONE = r"""
**Cosa misura.** In ogni strato, ogni canale ha 16 modi del primo ordine, ciascuno con
il suo polo continuo $a = -\exp(A_{\log})$. Sono parametri imparati in addestramento e
si leggono direttamente dai pesi: **non dipendono dal testo**.

**Perché conta.** Il polo fissa la velocità con cui un modo dimentica. Più $a$ è vicino a zero,
più il modo è lento, cioè ha memoria lunga. Il segno meno garantisce $a<0$, quindi tutti i modi
sono asintoticamente stabili per costruzione.

**Esempio.** Un modo con $a = -0.01$ si comporta come $\dot{x} = -0.01\,x$, quindi
$x(t) = x_0\,e^{-0.01 t}$. La costante di tempo è $\tau = 1/|a| = 100$: dopo $t=100$ resta
il 37% dello stato iniziale. Con $a=-10$ invece $\tau = 0.1$, e il modo si spegne quasi subito.
Quanti *token* corrispondano a $\tau$ lo dice il passo $\Delta$ della Fase 2.

**Confronto con FP16/BF16.** Con un confronto attivo, accanto al modello FP32 compare lo stesso spettro con
$A_{\log}$ arrotondato al formato ridotto, come quando il modello viene caricato con i pesi in FP16 o BF16: il
modello calcola $a$ in FP32, ma a partire da $A_{\log}$ già arrotondato, quindi l'errore nasce tutto lì. Se
$A_{\log}$ si sposta di $\delta$, il polo diventa $a\,e^{\delta}$: l'errore relativo su $a$ è circa l'errore
**assoluto** su $A_{\log}$. La spaziatura dei numeri in virgola mobile cresce con il valore, quindi i poli più
colpiti sono i più lenti ($|a|\ll 1$, $A_{\log}$ molto negativo), quelli con la memoria più lunga, e i più veloci
($|a|\gg 1$). Per esempio, con $A_{\log} \approx -13.1$ ($|a| \approx 2\cdot10^{-6}$, $\tau \approx 5\cdot10^{5}$)
FP16 ha spaziatura $2^{-7} \approx 0.0078$ tra 8 e 16, quindi $A_{\log}$ si sposta al più di $0.0039$ e $a$
cambia fino a $\pm 0.4\%$; BF16 ha spaziatura $2^{-4} = 0.0625$ e $a$ cambia fino a $\pm 3\%$.
"""

DESCRIZIONE_CONFRONTO = "a = -exp(A_log) con A_log arrotondato al formato dei pesi; errori calcolati in FP64."

COLONNE = [Colonna("strato", "Strato", "", "int"),
           Colonna("a_min", "Polo più veloce", "a_min", "sci"),
           Colonna("a_mediana", "Polo tipico", "mediana", "sci"),
           Colonna("a_max", "Polo più lento", "a_max", "sci"),
           Colonna("tau", "Costante di tempo", "max = 1/|a_max|", "sci"),
           Colonna("instabili", "Poli instabili", "a ≥ 0", "int")]

LEGENDA = [("Polo più veloce", "Il polo più negativo dello strato: il modo che si estingue più in fretta."),
           ("Polo tipico", "La mediana dei poli dello strato: metà sono più veloci, metà più lenti."),
           ("Polo più lento", "Il polo più vicino a zero: il modo con la memoria più lunga."),
           ("Costante di tempo", "1/|a_max| in unità di tempo continuo; in token dipende da Delta (Fase 2)."),
           ("Poli instabili", "Poli con a ≥ 0. Per costruzione (a = -exp) devono essere zero.")]


# Poli continui letti dai pesi, strato per strato; con tipo, da A_log arrotondato a quel formato (come nella copia)
def _poli(ctx, tipo=None):
    strati = ctx.model.backbone.layers
    n = len(strati)
    righe, log_abs = [], []
    with torch.no_grad():
        for i, layer in enumerate(strati):
            A_log = layer.mixer.A_log.float()
            if tipo is not None:
                A_log = A_log.to(tipo).float()
            a = -torch.exp(A_log)
            a_max = a.max().item()
            righe.append(dict(strato=i, a_min=a.min().item(), a_mediana=a.median().item(),
                              a_max=a_max, tau=1.0 / abs(a_max) if a_max != 0 else float("inf"),
                              instabili=int((a >= 0).sum().item()), totale=a.numel()))
            log_abs.append(torch.log10((-a).clamp_min(1e-45)).flatten().cpu().numpy())
            ctx.avanzamento((i + 1) / n, f"Strato {i + 1} di {n}")
    return righe, log_abs


def _sintesi(righe):
    lento = max(righe, key=lambda r: r["a_max"])
    veloce = min(righe, key=lambda r: r["a_min"])
    instabili = sum(valori(righe, "instabili"))
    totale = sum(valori(righe, "totale"))
    decadi = math.log10(veloce["a_min"] / lento["a_max"])
    return lento, veloce, instabili, totale, decadi


def _evidenzia(lento, veloce):
    if lento["strato"] == veloce["strato"]:
        return {lento["strato"]: "polo più lento e più veloce"}
    return {lento["strato"]: "polo più lento", veloce["strato"]: "polo più veloce"}


def _esaminati(ctx, totale):
    n = ctx.info_modello["strati"]
    return Indicatore("Poli continui esaminati", fmt_int(totale),
                      f"{n} strati × {ctx.info_modello['canali']} canali × {ctx.info_modello['stati']} stati")


def esegui(ctx):
    nome = cf.formato_confronto(ctx)
    return _solo_fp32(ctx) if nome is None else _con_confronto(ctx, nome)


# Solo FP32: il calcolo della vecchia Fase 1, senza seconda copia
def _solo_fp32(ctx):
    righe, log_abs = _poli(ctx)
    lento, veloce, instabili, totale, decadi = _sintesi(righe)

    indicatori = [
        _esaminati(ctx, totale),
        Indicatore("Poli instabili (a ≥ 0)", fmt_int(instabili),
                   "tutti i modi sono asintoticamente stabili" if instabili == 0
                   else "esistono modi non stabili", "ok" if instabili == 0 else "attenzione"),
        Indicatore("Polo più lento", fmt_sci(lento["a_max"]),
                   f"strato {lento['strato']}, τ = {fmt_sci(lento['tau'])}"),
        Indicatore("Polo più veloce", fmt_sci(veloce["a_min"]), f"strato {veloce['strato']}"),
        Indicatore("Ampiezza dello spettro", f"{decadi:.1f} decadi",
                   "all'inizializzazione: 1.2 decadi (da −1 a −16)"),
    ]
    return Risultato(
        codice="1",
        titolo="Fase 1 - Poli continui  a = -exp(A_log)",
        descrizione="Letti dai pesi del modello: non dipendono dal testo.",
        colonne=COLONNE,
        righe=righe,
        indicatori=indicatori,
        legenda=LEGENDA,
        grafici=[_grafico_spettro(righe), _grafico_distribuzione(log_abs)],
        evidenzia=_evidenzia(lento, veloce),
    )


# FP32 e formato ridotto affiancati, più gli errori della vecchia Fase 3a
def _con_confronto(ctx, nome):
    tipo = FORMATI[nome]
    righe, log_abs = _poli(cf.tratto(ctx, 0.0, 0.3))
    righe_r, log_abs_r = _poli(cf.tratto(ctx, 0.3, 0.6), tipo)
    errori = _errori(cf.tratto(ctx, 0.6, 1.0), nome)
    lento, veloce, instabili, totale, decadi = _sintesi(righe)
    lento_r, veloce_r, instabili_r, _, decadi_r = _sintesi(righe_r)

    indicatori = [
        _esaminati(ctx, totale),
        cf.indicatore("Poli instabili (a ≥ 0)", instabili, instabili_r, fmt_int, nome,
                      "tutti i modi restano asintoticamente stabili" if instabili == instabili_r == 0 else "",
                      "ok" if instabili == instabili_r == 0 else "attenzione"),
        cf.indicatore("Polo più lento", lento["a_max"], lento_r["a_max"], fmt_sci, nome,
                      f"strato {lento['strato']} → {lento_r['strato']}, "
                      f"τ = {fmt_sci(lento['tau'])} → {fmt_sci(lento_r['tau'])}"),
        cf.indicatore("Polo più veloce", veloce["a_min"], veloce_r["a_min"], fmt_sci, nome,
                      f"strato {veloce['strato']} → {veloce_r['strato']}"),
        cf.indicatore("Ampiezza dello spettro", decadi, decadi_r, lambda x: f"{x:.1f} decadi", nome),
    ] + [i for i in errori.indicatori if i.etichetta != "Poli continui esaminati"]

    return Risultato(
        codice="1",
        titolo=f"Fase 1 - Poli continui  a = -exp(A_log), FP32 contro {nome}",
        descrizione=f"Letti dai pesi del modello: non dipendono dal testo. {DESCRIZIONE_CONFRONTO}",
        colonne=cf.colonne_affiancate(COLONNE, nome) + errori.colonne[1:],
        righe=cf.righe_unite(righe, righe_r, errori.righe),
        indicatori=indicatori,
        legenda=LEGENDA + errori.legenda
                + [(f"FP32 e {nome}", f"Le grandezze della Fase 1 compaiono due volte: nel modello FP32 e con A_log "
                                      f"arrotondato a {nome}, come nella copia del modello in quel formato.")],
        grafici=[_grafico_spettro(righe, righe_r, nome), _grafico_distribuzione(log_abs, log_abs_r, nome)]
                + errori.grafici,
        evidenzia=cf.evidenze_unite(_evidenzia(lento, veloce), errori.evidenzia),
        note_formato="FP32 e " + errori.note_formato,
    )


def _grafico_spettro(righe, righe_r=None, nome=None):
    x = valori(righe, "strato")
    veloce, tipico, lento = (valori(righe, k, abs) for k in ("a_min", "a_mediana", "a_max"))
    suffisso = " FP32" if righe_r else ""
    fig = g.nuova_figura("Strato", "|a|  (scala logaritmica)", log_y=True)
    g.zona(fig, "y", 1, 16, "inizializzazione S4D-Real: |a| da 1 a 16")
    g.fascia_min_mediana_max(fig, x, lento, tipico, veloce,
                             tuple(f"{s}{suffisso}" for s in ("polo più veloce", "polo tipico (mediana)", "polo più lento")))
    dati = pd.DataFrame({"strato": x, "|a| polo piu' veloce": veloce,
                         "|a| mediana": tipico, "|a| polo piu' lento": lento})
    tutti_lenti, tutti_veloci = lento, veloce
    if righe_r:
        veloce_r, tipico_r, lento_r = (valori(righe_r, k, abs) for k in ("a_min", "a_mediana", "a_max"))
        g.fascia_min_mediana_max(fig, x, lento_r, tipico_r, veloce_r,
                                 (f"polo più veloce {nome}", f"polo tipico {nome}", f"polo più lento {nome}"),
                                 colore=g.ARANCIO)
        dati = dati.rename(columns=lambda c: c if c == "strato" else f"{c} FP32")
        dati[f"|a| polo piu' veloce {nome}"] = veloce_r
        dati[f"|a| mediana {nome}"] = tipico_r
        dati[f"|a| polo piu' lento {nome}"] = lento_r
        tutti_lenti, tutti_veloci = lento + lento_r, veloce + veloce_r
    g.asse_strati(fig, len(righe))
    g.scala_log(fig, "y", min(1, min(tutti_lenti)), max(16, max(tutti_veloci)))
    come_leggere = """
- **Asse y logaritmico:** ogni tacca è un fattore 10. Si riporta $|a|$ perché la scala logaritmica
  richiede valori positivi; il polo vero è $a = -|a|$.
- **In basso = memoria lunga.** Un punto a $10^{-5}$ è un modo con $\\tau = 10^{5}$.
- **La fascia grigia** è dove stavano tutti i poli prima dell'addestramento (S4D-Real: $a_n=-(n+1)$, $n=0\\dots15$).
  Quanto la fascia blu esce da quella grigia misura quanto l'addestramento ha allargato lo spettro.
- **Esempio di lettura:** se in uno strato il bordo inferiore scende a $10^{-6}$ e quello superiore sale a $10^{8}$,
  quello strato contiene sia modi che ricordano "per sempre" sia modi che dimenticano a ogni passo.
"""
    if righe_r:
        come_leggere += f"""- **Blu = FP32, arancione = {nome}.** La fascia e la linea blu sono il modello FP32; le linee arancioni
  sono gli stessi poli con $A_{{\\log}}$ arrotondato a {nome}: tratteggiata il più veloce, punteggiata il più lento,
  continua la mediana. Se l'arancione copre il blu, l'arrotondamento non sposta lo spettro in modo visibile: su
  questa scala anche un errore del 3% (BF16) è largo appena un centesimo di decade. Le differenze vere si leggono
  nei grafici dell'errore relativo, più sotto.
"""
    return Grafico(
        "Spettro dei poli continui per strato", fig,
        "Per ogni strato, la fascia va dal polo più lento (in basso) al più veloce (in alto); "
        "la linea è il polo tipico." + (f" In arancione lo stesso spettro con A_log in {nome}." if righe_r else ""),
        come_leggere,
        dati,
    )


# Istogramma di log10|a| a quarti di decade, su tutti gli strati; con log_abs_r, il formato ridotto sovrapposto
def _grafico_distribuzione(log_abs, log_abs_r=None, nome=None):
    tutti = np.concatenate(log_abs)
    estremi = np.concatenate([tutti, np.concatenate(log_abs_r)]) if log_abs_r else tutti
    passo = 0.25
    lo, hi = math.floor(estremi.min() / passo) * passo, math.ceil(estremi.max() / passo) * passo + passo
    conteggi, bordi = np.histogram(tutti, bins=np.arange(lo, hi + passo / 2, passo))
    centri = (bordi[:-1] + bordi[1:]) / 2
    quota = conteggi / conteggi.sum()
    fig = g.nuova_figura("|a|  (decadi)", "Quota dei poli", hover_unificato=False)
    g.zona(fig, "x", 0, math.log10(16), "inizializzazione")
    fig.add_trace(go.Bar(x=centri, y=quota, width=passo * 0.9, marker=dict(color=g.BLU, line_width=0),
                         customdata=np.stack([10 ** bordi[:-1], 10 ** bordi[1:], conteggi], axis=1),
                         hovertemplate="|a| tra %{customdata[0]:.2e} e %{customdata[1]:.2e}<br>"
                                       "%{customdata[2]:,} poli (%{y:.2%})<extra></extra>",
                         name="poli FP32" if log_abs_r else "poli"))
    dati = pd.DataFrame({"log10|a| da": bordi[:-1], "log10|a| a": bordi[1:],
                         "numero di poli": conteggi, "quota": quota})
    if log_abs_r:
        conteggi_r, _ = np.histogram(np.concatenate(log_abs_r), bins=bordi)
        quota_r = conteggi_r / conteggi_r.sum()
        fig.add_trace(go.Bar(x=centri, y=quota_r, width=passo * 0.45, marker=dict(color=g.ARANCIO, line_width=0),
                             customdata=conteggi_r, name=f"poli {nome}",
                             hovertemplate=f"%{{customdata:,}} poli in {nome} (%{{y:.2%}})<extra></extra>"))
        fig.update_layout(barmode="overlay")
        dati = dati.rename(columns={"numero di poli": "numero di poli FP32", "quota": "quota FP32"})
        dati[f"numero di poli {nome}"] = conteggi_r
        dati[f"quota {nome}"] = quota_r
    fig.update_yaxes(tickformat=".1%")
    g.asse_decadi(fig, "x", lo, hi)
    come_leggere = """
- **Ogni barra** raccoglie i poli con $|a|$ in un intervallo di un quarto di decade (fattore $10^{0.25}\\approx 1.8$).
- **Altezza** = quota sul totale dei poli del modello. Passando con il mouse si legge anche il numero esatto.
- **Le code** sono la parte interessante: a sinistra i modi lentissimi (memoria lunga, margine piccolo),
  a destra quelli che dimenticano subito.
- **Esempio di lettura:** una barra alta 5% tra $10^{-1}$ e $10^{-0.75}$ significa che un polo su venti ha
  costante di tempo tra circa 5.6 e 10.
"""
    if log_abs_r:
        come_leggere += f"""- **Barre blu = FP32, barre arancioni più strette = {nome}**, sugli stessi intervalli. Una barra arancione
  più bassa o più alta di quella blu vuol dire che l'arrotondamento ha spostato dei poli nell'intervallo accanto:
  succede solo ai poli vicini a un bordo, perché l'errore (al più qualche %) è molto più piccolo di un quarto di decade (78%).
"""
    return Grafico(
        "Distribuzione dei poli continui (tutti gli strati)", fig,
        "Quanti poli cadono in ogni intervallo di |a|, largo un quarto di decade."
        + (f" In arancione lo stesso conteggio con A_log in {nome}." if log_abs_r else ""),
        come_leggere,
        dati,
    )


# Errori dell'arrotondamento di A_log: il calcolo della vecchia Fase 3a

# Metà della spaziatura del formato attorno a x: il massimo errore di arrotondamento (anche nei subnormali)
def _mezza_spaziatura(x, tipo):
    fi = torch.finfo(tipo)
    esponente = torch.floor(torch.log2(x.abs().clamp_min(fi.smallest_normal)))
    return fi.eps * torch.exp2(esponente) / 2


def _errori(ctx, nome):
    tipo = FORMATI[nome]
    strati = ctx.model.backbone.layers
    n = len(strati)
    righe, tutti_abs, tutti_err = [], [], []
    fuori_limite = 0
    with torch.no_grad():
        for i, layer in enumerate(strati):
            A = layer.mixer.A_log.detach().double()
            A_r = layer.mixer.A_log.detach().float().to(tipo).double()
            err_A = (A_r - A).abs()
            # |a_r - a| / |a| = |exp(A_r - A) - 1|, esatto in FP64 con expm1
            err = torch.expm1(A_r - A).abs()
            fuori_limite += int((err_A > _mezza_spaziatura(A, tipo).double()).sum().item())
            k = int(err.argmax().item())
            canale, stato = divmod(k, A.shape[1])
            oltre = int((err > SOGLIA).sum().item())
            if err.max().item() == 0:
                k = None
            righe.append(dict(strato=i, err_max=err.max().item(), err_mediana=err.median().item(),
                              err_A_max=err_A.max().item(), oltre=oltre, perc_oltre=oltre / err.numel(),
                              a_peggiore=None if k is None else -math.exp(A.flatten()[k].item()),
                              canale=canale, stato=stato,
                              A_log_peggiore=None if k is None else A.flatten()[k].item(), totale=err.numel()))
            tutti_abs.append(torch.exp(A).flatten().cpu().numpy())
            tutti_err.append(err.flatten().cpu().numpy())
            ctx.avanzamento((i + 1) / n, f"Strato {i + 1} di {n}")

    abs_a, err = np.concatenate(tutti_abs), np.concatenate(tutti_err)
    totale, oltre = len(err), sum(valori(righe, "oltre"))
    peggiore = max(righe, key=lambda r: r["err_max"])
    piu_oltre = max(righe, key=lambda r: r["oltre"])
    controllo = tipo == torch.float32

    indicatori = [
        Indicatore("Poli continui esaminati", fmt_int(totale),
                   f"{n} strati × {ctx.info_modello['canali']} canali × {ctx.info_modello['stati']} stati"),
        Indicatore("Errore relativo massimo su a", fmt_perc(peggiore["err_max"]),
                   f"strato {peggiore['strato']}, canale {peggiore['canale']}, stato {peggiore['stato']}: "
                   f"a = {fmt_sci(peggiore['a_peggiore'])}, A_log = {peggiore['A_log_peggiore']:.3f}"
                   if peggiore["err_max"] > 0 else "nessun polo si sposta",
                   ("ok" if peggiore["err_max"] == 0 else None) if controllo else None),
        Indicatore("Errore relativo tipico su a", fmt_sci(float(np.median(err))), "mediana su tutti i poli"),
        Indicatore("Poli spostati più dell'1%", fmt_int(oltre),
                   f"{fmt_perc(oltre / totale)} dei poli"
                   + (f"; il massimo nello strato {piu_oltre['strato']} ({fmt_int(piu_oltre['oltre'])})" if oltre else ""),
                   "attenzione" if oltre else "ok"),
        Indicatore("Errore entro il limite teorico", "sì, per tutti i poli" if fuori_limite == 0
                   else f"no: {fmt_int(fuori_limite)} poli",
                   "|A_log arrotondato − A_log| ≤ metà della spaziatura del formato",
                   "ok" if fuori_limite == 0 else "attenzione"),
    ]
    if controllo:
        indicatori.append(Indicatore("Controllo FP32", "superato" if peggiore["err_max"] == 0 else "NON superato",
                                     "senza arrotondamento gli errori devono essere esattamente nulli",
                                     "ok" if peggiore["err_max"] == 0 else "attenzione"))

    evidenzia = {peggiore["strato"]: "errore massimo"}
    if oltre and piu_oltre["strato"] != peggiore["strato"]:
        evidenzia[piu_oltre["strato"]] = "più poli oltre l'1%"

    return Risultato(
        codice="1",
        titolo=f"Fase 1 - Poli continui con A_log arrotondato a {nome}",
        descrizione=DESCRIZIONE_CONFRONTO,
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("err_max", "Errore massimo su a", "relativo", "sci"),
                 Colonna("err_mediana", "Errore tipico su a", "mediana", "sci"),
                 Colonna("err_A_max", "Errore massimo su A_log", "assoluto", "sci"),
                 Colonna("a_peggiore", "Polo più perturbato", "a", "sci"),
                 Colonna("oltre", "Poli oltre l'1%", "numero", "int"),
                 Colonna("perc_oltre", "Poli oltre l'1%", "% dello strato", "perc")],
        righe=righe,
        indicatori=indicatori,
        legenda=[("Errore massimo su a", "Il più grande |a arrotondato − a| / |a| dello strato."),
                 ("Errore tipico su a", "La mediana dell'errore relativo sui poli dello strato."),
                 ("Errore massimo su A_log", "Il più grande |A_log arrotondato − A_log|: per poli piccoli è circa "
                                             "uguale all'errore relativo su a."),
                 ("Polo più perturbato", "Il valore di a (in FP32) del polo con l'errore relativo massimo."),
                 ("Poli oltre l'1%", "Quanti poli dello strato cambiano di più dell'1%."),
                 ("% dello strato", "Quota dei poli dello strato (canali × stati) oltre la soglia.")],
        grafici=[_grafico_errore_vs_a(abs_a, err, tipo), _grafico_strati(righe)],
        evidenzia=evidenzia,
        note_formato=descrivi_formato(nome, ctx.dispositivo).replace(", ricorrenza in FP32", "")
                     + "; a in FP32, errori in FP64",
    )


# Errore relativo in funzione di |a|: massimo e mediana per quarto di decade, con il limite teorico
def _grafico_errore_vs_a(abs_a, err, tipo):
    lg = np.log10(abs_a)
    lo, hi = math.floor(lg.min() / PASSO) * PASSO, math.ceil(lg.max() / PASSO) * PASSO + PASSO
    bordi = np.arange(lo, hi + PASSO / 2, PASSO)
    indice = np.clip(np.digitize(lg, bordi) - 1, 0, len(bordi) - 2)
    da, centri, massimi, mediane, conteggi = [], [], [], [], []
    for b in range(len(bordi) - 1):
        nel_bin = err[indice == b]
        if len(nel_bin):
            da.append(bordi[b])
            centri.append((bordi[b] + bordi[b + 1]) / 2)
            massimi.append(nel_bin.max())
            mediane.append(np.median(nel_bin))
            conteggi.append(len(nel_bin))
    x = 10 ** np.array(centri)
    # Il limite cresce con |A_log|: nel bin il valore più alto è su uno dei due bordi
    bordi_bin = torch.from_numpy(np.log(10.0) * np.stack([np.array(da), np.array(da) + PASSO]))
    teorico = np.expm1(_mezza_spaziatura(bordi_bin, tipo).max(dim=0).values.numpy())

    fig = g.nuova_figura("|a|  (scala logaritmica)", "Errore relativo su a  (scala logaritmica)", log_y=True)
    fig.update_xaxes(type="log", exponentformat="power", dtick=1)
    positivi = [v for v in massimi + mediane if v > 0]
    if positivi:
        fig.add_trace(go.Scatter(x=x, y=teorico, mode="lines", line_shape="hvh",
                                 name="limite teorico (½ spaziatura di A_log)",
                                 line=dict(color=g.TENUE, width=1.5, dash="dot"),
                                 hovertemplate="limite %{y:.2e}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=np.where(np.array(mediane) > 0, mediane, np.nan), mode="lines",
                                 name="errore tipico (mediana)", line=dict(color=g.BLU_CHIARO, width=2),
                                 hovertemplate="mediana %{y:.2e}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=np.where(np.array(massimi) > 0, massimi, np.nan), mode="lines+markers",
                                 name="errore massimo", line=dict(color=g.BLU, width=2.5), marker=g.marcatore(),
                                 customdata=conteggi,
                                 hovertemplate="massimo %{y:.2e} su %{customdata:,} poli<extra></extra>"))
        g.linea_soglia(fig, "y", SOGLIA, "1%")
        g.scala_log(fig, "y", min(positivi), max(max(positivi), max(teorico), SOGLIA))
    else:
        fig.add_annotation(text="Nessun errore: A_log non viene arrotondato (controllo FP32)", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    dati = pd.DataFrame({"log10|a| da": da, "log10|a| a": np.array(da) + PASSO, "numero di poli": conteggi, "errore massimo": massimi,
                         "errore mediano": mediane, "limite teorico": teorico})
    return Grafico(
        "Errore relativo su a in funzione di |a|", fig,
        "Per ogni quarto di decade di |a|: l'errore più grande e quello tipico dovuti all'arrotondamento di A_log.",
        """
- **Entrambi gli assi logaritmici.** A sinistra i poli lenti (memoria lunga), a destra quelli veloci.
- **La curva a V** è attesa: l'errore relativo su $a$ è circa l'errore assoluto su $A_{\\log} = \\ln|a|$,
  che cresce con $|A_{\\log}|$. Il fondo della V è attorno a $|a| = 1$, dove $A_{\\log} \\approx 0$.
- **Linea punteggiata** = limite teorico, metà della spaziatura del formato in $A_{\\log}$: il massimo non può superarla.
  I gradini sono le potenze di 2 in cui la spaziatura raddoppia.
- **Linea rossa** = 1%: i poli sopra cambiano la loro costante di tempo di più di un centesimo.
- **Esempio di lettura:** un massimo a $3\\cdot10^{-2}$ attorno a $|a| = 10^{-6}$ (in BF16) vuol dire che un modo con
  memoria di circa un milione di unità di tempo la vede cambiare del 3%.
""",
        dati,
    )


def _grafico_strati(righe):
    x, massimo, tipico = (valori(righe, k) for k in ("strato", "err_max", "err_mediana"))
    fig = g.nuova_figura("Strato", "Errore relativo su a  (scala logaritmica)", log_y=True)
    positivi = [v for v in massimo + tipico if v > 0]
    if positivi:
        fig.add_trace(go.Scatter(x=x, y=[v if v > 0 else None for v in tipico], mode="lines", name="errore tipico (mediana)",
                                 line=dict(color=g.BLU_CHIARO, width=2), hovertemplate="mediana %{y:.2e}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=[v if v > 0 else None for v in massimo], mode="lines+markers",
                                 name="errore massimo", line=dict(color=g.BLU, width=2.5), marker=g.marcatore(8),
                                 hovertemplate="massimo %{y:.2e}<extra></extra>"))
        g.linea_soglia(fig, "y", SOGLIA, "1%")
        g.scala_log(fig, "y", min(positivi), max(max(positivi), SOGLIA))
    else:
        fig.add_annotation(text="Nessun errore in nessuno strato (controllo FP32)", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    g.asse_strati(fig, len(righe))
    dati = pd.DataFrame({"strato": x, "errore massimo": massimo, "errore mediano": tipico,
                         "poli oltre 1%": valori(righe, "oltre")})
    return Grafico(
        "Errore relativo su a per strato", fig,
        "Per ogni strato, l'errore relativo più grande e quello tipico sui poli continui.",
        """
- **Linea scura** = il polo più perturbato dello strato; **linea chiara** = l'errore tipico (mediana).
- Il massimo dipende da quanto è esteso lo spettro dello strato: gli strati con poli molto lenti o molto veloci
  (vedi lo spettro FP32 qui sopra) hanno $|A_{\\log}|$ più grande e quindi errori più grandi.
- **Linea rossa** = 1%.
- **Esempio di lettura:** massimo a $4\\cdot10^{-3}$ e mediana a $10^{-4}$ significa che il polo peggiore dello
  strato si sposta dello 0.4%, mentre quello tipico di un decimillesimo.
""",
        dati,
    )


FASE = Fase(
    codice="1",
    nome="Poli continui",
    breve="a = -exp(A_log), dai pesi, anche con A_log in FP16/BF16 (non usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=False,
    esegui=esegui,
    parametri=[cf.PARAMETRO],
)
