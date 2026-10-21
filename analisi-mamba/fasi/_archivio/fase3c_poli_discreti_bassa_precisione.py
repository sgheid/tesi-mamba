import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import FORMATI, delta_baseline, descrivi_formato, esegui_in_formato, indicatore_formato
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Parametro, Risultato, valori
from nucleo.sonde import delta_modello

# Istogramma dell'errore relativo sul margine: log10 da -12 a 2 a centesimi di decade (per la mediana)
ERR_MIN, ERR_MAX, ERR_PASSO = -12.0, 2.0, 0.01
N_ERR = int(round((ERR_MAX - ERR_MIN) / ERR_PASSO))
# Istogramma dei margini della baseline vicino alla soglia FP32: log10(1 - abar) da -10 a -5
VIC_MIN, VIC_MAX, VIC_PASSO = -10.0, -5.0, 0.05
N_VIC = int(round((VIC_MAX - VIC_MIN) / VIC_PASSO))

SPIEGAZIONE = r"""
**Cosa misura.** Quanti poli discreti $\bar a = e^{\Delta a}$ cambiano classe quando il modello lavora con i pesi
in FP16 o BF16. Per ogni token, canale e stato si confronta il polo del modello FP32 con quello del modello a
bassa precisione. In entrambi $\bar a$ è calcolato **in FP32, come fa il modello**, ma nel secondo da $\Delta$ e
$a$ del formato ridotto (Fasi 3a e 3b). I margini $1-\bar a$ sono calcolati esattamente in FP64 con
$-\mathrm{expm1}(\Delta a)$, a partire dal $\Delta$ di ciascun modello.

**Perché conta.** Un polo con margine sotto $2^{-25} \approx 3\cdot10^{-8}$ viene arrotondato a **esattamente 1**
in FP32 (Fase 2b). Se $\Delta$ o $a$ si spostano, un polo vicino alla soglia può attraversarla in un senso o
nell'altro. In FP16 c'è anche una seconda causa, diversa: con $\Delta = 0$ vale $\bar a = 1$ qualunque sia $a$,
e il token non entra nello stato. Le due cause vengono contate separatamente.

**Esempio.** Un polo con margine $4\cdot10^{-8}$ in FP32 vale $\bar a < 1$. Se in FP16 il suo $\Delta$ diminuisce
del 30%, il margine scende a $2.8\cdot10^{-8}$, sotto la soglia, e $\bar a$ diventa 1: è un polo **nuovo** su 1.
Al contrario, un polo con margine $2.5\cdot10^{-8}$ (su 1 in FP32) il cui $\Delta$ cresce del 25% sale a
$3.1\cdot10^{-8}$ e torna sotto 1: è un polo **perso**.
"""


# Mediana dall'istogramma: errori nulli contati a parte, poi la cumulata dei bin (centro del bin, in log)
def _mediana(zeri, isto):
    totale = zeri + isto.sum()
    if totale == 0:
        return float("nan")
    if zeri >= totale / 2:
        return 0.0
    k = int(np.searchsorted(np.cumsum(isto), totale / 2 - zeri))
    return 10 ** (ERR_MIN + (k + 0.5) * ERR_PASSO)


def esegui(ctx):
    nome = ctx.parametri.get("formato", "FP16")
    tipo = FORMATI[nome]
    n = len(ctx.model.backbone.layers)
    A_log_base = {i: l.mixer.A_log.detach().float().cpu() for i, l in enumerate(ctx.model.backbone.layers)}
    base = delta_baseline(ctx, lambda f, m: ctx.avanzamento(0.4 * f, m))
    ctx.avanzamento(0.45, f"carico la copia in {nome}")
    dati, isto_err, zeri_err, vicini_base, vicini_su_1 = {}, {}, {}, {}, {}

    # Per ogni strato, a blocchi di token: abar in FP32 come il modello, margini esatti in FP64
    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            d_r = delta_modello(mixer, uscita)
            dev = d_r.device
            d_b = base.pop(i).to(dev)
            A_b = A_log_base.pop(i).to(dev)
            a_b32, a_b64 = -torch.exp(A_b), -torch.exp(A_b.double())
            a_r32, a_r64 = -torch.exp(mixer.A_log.float()), -torch.exp(mixer.A_log.double())
            L, D = d_r.shape
            passo = max(1, 4_000_000 // (D * a_r32.shape[1]))
            c = dict(su_1_base=0, su_1_rid=0, nuovi_zero=0, nuovi_soglia=0, persi=0, non_finiti=0, err_nulli=0)
            isto = torch.zeros(N_ERR, dtype=torch.float64, device=dev)
            vb = torch.zeros(N_VIC, dtype=torch.float64, device=dev)
            v1 = torch.zeros(N_VIC, dtype=torch.float64, device=dev)
            for s in range(0, L, passo):
                db, dr = d_b[s:s + passo], d_r[s:s + passo]
                abar_b = torch.exp(db[:, :, None] * a_b32)
                abar_r = torch.exp(dr.float()[:, :, None] * a_r32)
                m_b = -torch.expm1(db.double()[:, :, None] * a_b64)
                m_r = -torch.expm1(dr.double()[:, :, None] * a_r64)
                finito = torch.isfinite(abar_r)
                uno_b, uno_r = abar_b == 1.0, abar_r == 1.0
                zero_r = (dr == 0)[:, :, None]
                nuovi = uno_r & ~uno_b
                c["su_1_base"] += int(uno_b.sum().item())
                c["su_1_rid"] += int(uno_r.sum().item())
                c["nuovi_zero"] += int((nuovi & zero_r).sum().item())
                c["nuovi_soglia"] += int((nuovi & ~zero_r).sum().item())
                c["persi"] += int((uno_b & ~uno_r & finito).sum().item())
                c["non_finiti"] += int((~finito).sum().item())
                err = ((m_r - m_b).abs() / m_b)[finito & torch.isfinite(m_r) & (m_b > 0)]
                c["err_nulli"] += int((err == 0).sum().item())
                lg = torch.log10(err[err > 0]).clamp(ERR_MIN, ERR_MAX - 1e-9)
                isto += torch.histc(lg.float(), bins=N_ERR, min=ERR_MIN, max=ERR_MAX).double()
                lb = torch.log10(m_b.clamp_min(1e-300)).clamp_min(VIC_MIN).float()
                vb += torch.histc(lb, bins=N_VIC, min=VIC_MIN, max=VIC_MAX).double()
                v1 += torch.histc(lb[uno_r], bins=N_VIC, min=VIC_MIN, max=VIC_MAX).double()
            isto_err[i] = isto.cpu().numpy()
            zeri_err[i] = c.pop("err_nulli")
            vicini_base[i], vicini_su_1[i] = vb.cpu().numpy(), v1.cpu().numpy()
            totale = L * a_r32.numel()
            dati[i] = dict(strato=i, **c, nuovi=c["nuovi_zero"] + c["nuovi_soglia"],
                           err_mediana=_mediana(zeri_err[i], isto_err[i]), totale=totale)
            ctx.avanzamento(0.45 + 0.55 * (i + 1) / n, f"{nome}, strato {i + 1} di {n}")
        return hook

    esito = esegui_in_formato(ctx, tipo, crea_hook)
    righe = [dati[i] for i in range(n)]
    somma = {k: sum(valori(righe, k)) for k in ("su_1_base", "su_1_rid", "nuovi", "nuovi_zero", "nuovi_soglia",
                                                "persi", "non_finiti", "totale")}
    err_tipico = _mediana(sum(zeri_err.values()), sum(isto_err.values()))
    controllo = tipo == torch.float32
    tutto_zero = (somma["nuovi"] == 0 and somma["persi"] == 0 and somma["non_finiti"] == 0
                  and all(r["err_mediana"] == 0 for r in righe) and all(not v.any() for v in isto_err.values()))

    indicatori = [
        indicatore_formato(nome, ctx.dispositivo, esito),
        Indicatore("Poli discreti confrontati", fmt_int(somma["totale"]),
                   f"{ctx.n_token} token × {ctx.info_modello['canali']} canali × "
                   f"{ctx.info_modello['stati']} stati × {n} strati"),
        Indicatore("Poli su 1, FP32 → formato ridotto", f"{fmt_int(somma['su_1_base'])} → {fmt_int(somma['su_1_rid'])}",
                   f"{fmt_perc(somma['su_1_base'] / somma['totale'])} → {fmt_perc(somma['su_1_rid'] / somma['totale'])} dei poli"),
        Indicatore("Poli diventati 1 (nuovi)", fmt_int(somma["nuovi"]),
                   f"per Δ = 0: {fmt_int(somma['nuovi_zero'])} · per margine sotto 2⁻²⁵: {fmt_int(somma['nuovi_soglia'])}",
                   "attenzione" if somma["nuovi"] else "ok"),
        Indicatore("Poli non più su 1 (persi)", fmt_int(somma["persi"]),
                   "su 1 in FP32, sotto 1 nel formato ridotto", "attenzione" if somma["persi"] else "ok"),
        Indicatore("Errore tipico sul margine 1 − ā", fmt_sci(err_tipico),
                   "mediana su tutti i poli, risoluzione di un centesimo di decade"),
        Indicatore("Poli non finiti (inf/NaN)", fmt_int(somma["non_finiti"]),
                   "nessun overflow" if somma["non_finiti"] == 0 else "esclusi da conteggi ed errori",
                   "ok" if somma["non_finiti"] == 0 else "attenzione"),
    ]
    if controllo:
        indicatori.append(Indicatore("Controllo FP32", "superato" if tutto_zero else "NON superato",
                                     "due passaggi in FP32 devono dare gli stessi poli", "ok" if tutto_zero else "attenzione"))

    evidenzia = {}
    piu_nuovi = max(righe, key=lambda r: r["nuovi"])
    piu_persi = max(righe, key=lambda r: r["persi"])
    if piu_nuovi["nuovi"]:
        evidenzia[piu_nuovi["strato"]] = "più poli nuovi su 1"
    if piu_persi["persi"]:
        evidenzia[piu_persi["strato"]] = (evidenzia.get(piu_persi["strato"], "") + " · " if piu_persi["strato"] in evidenzia
                                          else "") + "più poli persi"

    return Risultato(
        codice="3c",
        titolo=f"Fase 3c - Poli discreti e margini con i pesi in {nome}",
        descrizione="abar in FP32 come nel modello, nella baseline FP32 e nel formato ridotto; margini esatti in FP64.",
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("su_1_base", "Poli su 1", "baseline FP32", "int"),
                 Colonna("su_1_rid", "Poli su 1", "formato ridotto", "int"),
                 Colonna("nuovi_zero", "Nuovi su 1", "per Delta = 0", "int"),
                 Colonna("nuovi_soglia", "Nuovi su 1", "per margine < 2^-25", "int"),
                 Colonna("persi", "Persi", "non più su 1", "int"),
                 Colonna("err_mediana", "Errore tipico sul margine", "mediana, relativo", "sci"),
                 Colonna("non_finiti", "Non finiti", "inf/NaN", "int")],
        righe=righe,
        indicatori=indicatori,
        legenda=[("Poli su 1", "Poli (token × canali × stati) che valgono esattamente 1 in FP32."),
                 ("baseline FP32", "Nel modello con i pesi in FP32 (come nella Fase 2b, ma con il Delta del modello)."),
                 ("formato ridotto", "Nel modello con i pesi in FP16/BF16."),
                 ("per Delta = 0", "Nuovi su 1 perché il Delta del formato ridotto vale 0: il token non entra nello stato."),
                 ("per margine < 2^-25", "Nuovi su 1 con Delta > 0: il margine è sceso sotto la soglia di FP32."),
                 ("Persi", "Su 1 nella baseline, sotto 1 nel formato ridotto: il margine è salito sopra la soglia."),
                 ("Errore tipico sul margine", "Mediana di |m ridotto − m FP32| / m FP32, con m = 1 − abar esatto; "
                                               "calcolata da un istogramma a centesimi di decade."),
                 ("Non finiti", "Poli inf o NaN nel formato ridotto: contati ed esclusi.")],
        grafici=[_grafico_nuovi_persi(righe), _grafico_errore(righe), _grafico_vicini(vicini_base, vicini_su_1, n)],
        evidenzia=evidenzia,
        note_formato=descrivi_formato(nome, ctx.dispositivo) + "; abar in FP32, margini in FP64",
    )


def _grafico_nuovi_persi(righe):
    x = valori(righe, "strato")
    zero, soglia, persi = (valori(righe, k) for k in ("nuovi_zero", "nuovi_soglia", "persi"))
    fig = g.nuova_figura("Strato", "Poli  (nuovi sopra lo zero, persi sotto)", hover_unificato=False)
    fig.add_trace(go.Bar(x=x, y=zero, name="nuovi su 1: Δ = 0", marker=dict(color=g.BLU, line_width=0),
                         hovertemplate="strato %{x}: %{y:,} nuovi per Δ = 0<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=soglia, name="nuovi su 1: margine sotto 2⁻²⁵", marker=dict(color=g.BLU_CHIARO, line_width=0),
                         hovertemplate="strato %{x}: %{y:,} nuovi per margine sotto soglia<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=[-p for p in persi], name="persi (non più su 1)", marker=dict(color=g.TENUE, line_width=0),
                         customdata=persi, hovertemplate="strato %{x}: %{customdata:,} persi<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line=dict(color=g.ASSE, width=1))
    g.asse_strati(fig, len(righe))
    if not any(zero) and not any(soglia) and not any(persi):
        fig.add_annotation(text="Nessun polo cambia classe in questo formato", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    dati = pd.DataFrame({"strato": x, "nuovi su 1 per Delta = 0": zero, "nuovi su 1 per margine < 2^-25": soglia,
                         "persi": persi, "su 1 baseline": valori(righe, "su_1_base"),
                         "su 1 formato ridotto": valori(righe, "su_1_rid")})
    return Grafico(
        "Poli che cambiano classe, per strato", fig,
        "Poli diventati esattamente 1 (sopra lo zero, divisi per causa) e poli non più su 1 (sotto lo zero).",
        """
- **Sopra lo zero** i poli **nuovi**: $\\bar a < 1$ in FP32, $\\bar a = 1$ nel formato ridotto.
  **Blu scuro** = per $\\Delta = 0$ (solo FP16): il token non entra nello stato e tutti i 16 poli del canale valgono 1.
  **Blu chiaro** = con $\\Delta > 0$: il margine è sceso sotto $2^{-25}$.
- **Sotto lo zero** (grigio) i poli **persi**: valevano 1 in FP32 e nel formato ridotto tornano sotto 1.
- Le due direzioni non si compensano: un polo nuovo e uno perso sono modi diversi, a token diversi.
- **Esempio di lettura:** una barra con 3 000 in blu scuro e 200 in grigio dice che in quello strato FP16 azzera
  Δ per circa 190 coppie token × canale (3 000 / 16) e fa tornare sotto 1 duecento poli che FP32 arrotondava a 1.
""",
        dati,
    )


def _grafico_errore(righe):
    x, y = valori(righe, "strato"), valori(righe, "err_mediana")
    fig = g.nuova_figura("Strato", "Errore relativo sul margine 1 − ā  (scala logaritmica)", log_y=True)
    positivi = [v for v in y if v > 0]
    if positivi:
        fig.add_trace(go.Scatter(x=x, y=[v if v > 0 else None for v in y], mode="lines+markers",
                                 name="errore tipico (mediana)", line=dict(color=g.BLU, width=2.5), marker=g.marcatore(8),
                                 hovertemplate="mediana %{y:.2e}<extra></extra>"))
        g.scala_log(fig, "y", min(positivi), max(positivi))
    else:
        fig.add_annotation(text="Nessuna differenza dalla baseline (controllo FP32)", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    fig.update_layout(showlegend=False)
    g.asse_strati(fig, len(righe))
    dati = pd.DataFrame({"strato": x, "errore mediano sul margine": y})
    return Grafico(
        "Errore tipico sul margine di stabilità, per strato", fig,
        "Mediana dell'errore relativo sul margine esatto 1 − ā tra il formato ridotto e la baseline FP32.",
        """
- Per poli lontani da 0, $1-\\bar a \\approx \\Delta\\,|a|$: l'errore sul margine è circa la somma degli errori su $\\Delta$
  (Fase 3b) e su $a$ (Fase 3a), quindi segue lo stesso andamento con la profondità.
- La mediana è letta da un istogramma a centesimi di decade: è precisa a circa il 2%.
- **Esempio di lettura:** $3\\cdot10^{-3}$ vuol dire che il margine tipico, e quindi la memoria $1/(1-\\bar a)$, cambia dello 0.3%.
""",
        dati,
    )


def _grafico_vicini(vicini_base, vicini_su_1, n):
    base = sum(vicini_base[i] for i in range(n))
    su_1 = sum(vicini_su_1[i] for i in range(n))
    bordi = np.linspace(VIC_MIN, VIC_MAX, N_VIC + 1)
    centri = (bordi[:-1] + bordi[1:]) / 2
    soglia = math.log10(g.SOGLIA_FP32)
    fig = g.nuova_figura("Margine 1 − ā nella baseline FP32  (decadi)", "Numero di poli  (scala logaritmica)",
                         log_y=True, hover_unificato=False)
    g.zona(fig, "x", VIC_MIN, soglia, "ā = 1 in FP32", colore=g.ROSSO_TENUE)
    g.linea_soglia(fig, "x", soglia, "soglia FP32 ≈ 3·10⁻⁸")
    visibili = base > 0
    if visibili.any():
        fig.add_trace(go.Bar(x=centri[visibili], y=base[visibili], width=VIC_PASSO * 0.9, name="poli della baseline",
                             marker=dict(color=g.BLU_CHIARO, line_width=0),
                             customdata=np.stack([10 ** bordi[:-1], 10 ** bordi[1:]], axis=1)[visibili],
                             hovertemplate="margine tra %{customdata[0]:.2e} e %{customdata[1]:.2e}<br>"
                                           "%{y:,.0f} poli<extra>baseline</extra>"))
        uno = su_1 > 0
        fig.add_trace(go.Bar(x=centri[uno], y=su_1[uno], width=VIC_PASSO * 0.5, name="di questi, su 1 nel formato ridotto",
                             marker=dict(color=g.BLU, line_width=0),
                             hovertemplate="%{y:,.0f} poli su 1 nel formato ridotto<extra></extra>"))
        fig.update_layout(barmode="overlay")
        g.scala_log(fig, "y", 1, base.max())
    else:
        fig.add_annotation(text="Nessun polo della baseline ha margine sotto 10⁻⁵ su questo testo", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    g.asse_decadi(fig, "x", VIC_MIN, VIC_MAX)
    fig.update_xaxes(range=[VIC_MIN - 0.1, VIC_MAX + 0.1])
    dati = pd.DataFrame({"log10(1-abar) da": bordi[:-1], "log10(1-abar) a": bordi[1:],
                         "poli baseline": base.astype(np.int64), "di questi su 1 nel formato ridotto": su_1.astype(np.int64)})
    for i in range(n):
        dati[f"strato {i} baseline"] = vicini_base[i].astype(np.int64)
    return Grafico(
        "Margini della baseline vicino alla soglia FP32", fig,
        "Quanti poli della baseline hanno margine 1 − ā tra 10⁻¹⁰ e 10⁻⁵, e quanti di questi valgono 1 nel formato ridotto.",
        """
- **Barre chiare** = poli della baseline FP32 per margine, a passi di 1/20 di decade; la barra più a sinistra raccoglie
  anche i margini sotto $10^{-10}$.
- **Barre scure, più strette** = di quei poli, quanti valgono esattamente 1 nel formato ridotto.
- **Nella zona rossa** i poli valgono 1 già in FP32: dove la barra scura è più bassa di quella chiara ci sono poli **persi**.
  **A destra della soglia** ogni barra scura è fatta di poli **nuovi**.
- Più poli stanno appena a destra della soglia, più ne possono attraversare con un piccolo errore su $\\Delta$ o su $a$.
- **Esempio di lettura:** 5 000 poli tra $3\\cdot10^{-8}$ e $3.4\\cdot10^{-8}$ con 800 su 1 nel formato ridotto: un polo su
  sei, appena sopra la soglia, viene spinto a 1 dagli errori del formato. I poli resi 1 da $\\Delta = 0$ compaiono
  dove la baseline aveva il loro margine, anche lontano dalla soglia.
- Il foglio Excel contiene anche la distribuzione della baseline **separata per strato**.
""",
        dati,
    )


FASE = Fase(
    codice="3c",
    nome="Poli discreti con pesi FP16/BF16",
    breve="abar e margini a bassa precisione contro la baseline FP32 (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[Parametro("formato", "Formato dei pesi", list(FORMATI), "FP16",
                         "Formato in cui viene caricata la seconda copia. FP32 (controllo) deve dare zero poli cambiati.")],
)
