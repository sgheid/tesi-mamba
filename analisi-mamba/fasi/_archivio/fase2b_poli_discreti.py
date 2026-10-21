import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Risultato, valori
from nucleo.sonde import delta_fp32, delta_fp64, esegui_con_sonde, ingresso_delta

# Istogramma dei margini: log10(1 - abar) da -16 a 0, a quarti di decade
LOG_MIN, LOG_MAX, PASSO = -16.0, 0.0, 0.25
N_BIN = int((LOG_MAX - LOG_MIN) / PASSO)

SPIEGAZIONE = r"""
**Cosa misura.** Il polo discreto di ogni modo a ogni token, $\bar a = e^{\Delta a}$, cioè il fattore
per cui il modo moltiplica il proprio stato a ogni passo: $h_{t} = \bar a\, h_{t-1} + \dots$
Poiché $a<0$ e $\Delta>0$, vale sempre $0<\bar a<1$: il polo sta dentro il cerchio unitario.

**Perché conta.** Il **margine di stabilità** è $1-\bar a$, la distanza dal bordo del cerchio.
Se il margine è più piccolo della risoluzione del formato numerico, $\bar a$ viene arrotondato a
**esattamente 1**: il modo passa da asintoticamente stabile a marginalmente stabile (un integratore).
In FP32 succede quando $1-\bar a < 2^{-25} \approx 3\cdot10^{-8}$.

**Esempio.** Con $1-\bar a = 10^{-3}$ lo stato si riduce a circa il 37% dopo $1/10^{-3} = 1000$ token:
è la "memoria" del modo. Con $1-\bar a = 10^{-9}$ la memoria sarebbe di un miliardo di token, ma in
FP32 quel $\bar a$ vale già 1 e il modo non dimentica più nulla.

Il calcolo di $\bar a$ è fatto in FP32, come nel modello; il margine esatto in FP64 con
`expm1`, per non perdere cifre nella sottrazione $1-\bar a$.
"""


def esegui(ctx):
    n = len(ctx.model.backbone.layers)
    dati, istogrammi = {}, {}

    # Per ogni strato: abar in FP32 come il modello, margine 1 - abar esatto in FP64, a blocchi di token
    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            ts = ingresso_delta(mixer, uscita)
            d32, d64 = delta_fp32(mixer, ts), delta_fp64(mixer, ts)
            a32, a64 = -torch.exp(mixer.A_log.float()), -torch.exp(mixer.A_log.double())
            L, D = d32.shape
            passo = max(1, 4_000_000 // (D * a32.shape[1]))
            margine_min, su_1, in_0, fuori = float("inf"), 0, 0, 0
            isto = torch.zeros(N_BIN, dtype=torch.float64, device=d32.device)
            for s in range(0, L, passo):
                abar32 = torch.exp(d32[s:s + passo, :, None] * a32)
                margine64 = -torch.expm1(d64[s:s + passo, :, None] * a64)
                margine_min = min(margine_min, margine64.min().item())
                su_1 += int((abar32 == 1.0).sum().item())
                in_0 += int((abar32 == 0.0).sum().item())
                fuori += int((abar32 > 1.0).sum().item())
                lg = torch.log10(margine64.clamp_min(1e-300)).clamp(LOG_MIN, LOG_MAX - 1e-9)
                isto += torch.histc(lg.float(), bins=N_BIN, min=LOG_MIN, max=LOG_MAX).double()
            totale = L * a32.numel()
            dati[i] = dict(strato=i, margine=margine_min,
                           memoria=1.0 / margine_min if margine_min > 0 else float("inf"),
                           su_1=su_1, perc_1=su_1 / totale, in_0=in_0, fuori=fuori, totale=totale)
            istogrammi[i] = isto.cpu().numpy()
            ctx.avanzamento((i + 1) / n, f"Strato {i + 1} di {n}")
        return hook

    esegui_con_sonde(ctx.model, ctx.ids, crea_hook)
    righe = [dati[i] for i in range(n)]
    totale, su_1, in_0, fuori = (sum(valori(righe, k)) for k in ("totale", "su_1", "in_0", "fuori"))
    piu_vicino = min(righe, key=lambda r: r["margine"])
    peggiori = [r for r in sorted(righe, key=lambda r: -r["su_1"]) if r["su_1"] > 0][:3]

    indicatori = [
        Indicatore("Poli discreti esaminati", fmt_int(totale),
                   f"{ctx.n_token} token × {ctx.info_modello['canali']} canali × "
                   f"{ctx.info_modello['stati']} stati × {n} strati"),
        Indicatore("Fuori dal cerchio (ā > 1)", fmt_int(fuori),
                   "nessun polo instabile" if fuori == 0 else "esistono poli instabili",
                   "ok" if fuori == 0 else "attenzione"),
        Indicatore("Arrotondati a 1 in FP32", fmt_int(su_1),
                   f"{fmt_perc(su_1 / totale)} dei poli: marginalmente stabili",
                   "attenzione" if su_1 > 0 else "ok"),
        Indicatore("Polo più vicino a 1", fmt_sci(piu_vicino["margine"]),
                   f"strato {piu_vicino['strato']}, memoria ≈ {fmt_sci(piu_vicino['memoria'])} token"),
        Indicatore("Strati con più poli su 1",
                   ", ".join(str(r["strato"]) for r in peggiori) if peggiori else "nessuno",
                   " · ".join(f"{r['strato']}: {fmt_int(r['su_1'])}" for r in peggiori)),
        Indicatore("Arrotondati a 0 in FP32", fmt_int(in_0),
                   f"{fmt_perc(in_0 / totale)}: senza memoria, innocui per la stabilità"),
    ]

    return Risultato(
        codice="2b",
        titolo="Fase 2b - Poli discreti  abar = exp(Delta * a)",
        descrizione="Calcolati in FP32 come nel modello; il margine esatto 1 - abar è calcolato in FP64.",
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("margine", "Polo più vicino a 1", "margine 1 - abar", "sci"),
                 Colonna("memoria", "Memoria massima", "token ≈ 1/(1-abar)", "sci"),
                 Colonna("su_1", "Poli su 1 in FP32", "numero", "int"),
                 Colonna("perc_1", "Poli su 1 in FP32", "% dello strato", "perc"),
                 Colonna("in_0", "Poli in 0 in FP32", "numero", "int")],
        righe=righe,
        indicatori=indicatori,
        legenda=[("Polo più vicino a 1", "Il margine di stabilità più piccolo dello strato, calcolato esattamente (FP64)."),
                 ("Memoria massima", "Quanti token servono a quel modo per dimenticare: circa 1/(1 - abar)."),
                 ("Poli su 1 in FP32", "Poli che FP32 arrotonda esattamente a 1: da asintoticamente a marginalmente stabili."),
                 ("% dello strato", "Quota dei poli dello strato (token x canali x stati) arrotondati a 1."),
                 ("Poli in 0 in FP32", "Modi che si azzerano a ogni token: nessuna memoria, nessun rischio per la stabilità.")],
        grafici=[_grafico_margine(righe), _grafico_su_uno(righe), _grafico_distribuzione(istogrammi, n)],
        evidenzia={r["strato"]: f"n. {k} per poli su 1" for k, r in enumerate(peggiori, start=1)},
        note_formato="FP32, con riferimento esatto in FP64 per il margine 1 - abar",
    )


def _grafico_margine(righe):
    x, y, memoria = (valori(righe, k) for k in ("strato", "margine", "memoria"))
    fig = g.nuova_figura("Strato", "Margine minimo 1 − ā  (scala logaritmica)", log_y=True)
    basso = min(min(y) / 10, g.SOGLIA_FP32 / 10)
    g.zona(fig, "y", basso, g.SOGLIA_FP32, "zona in cui FP32 arrotonda ā a 1", colore=g.ROSSO_TENUE)
    g.linea_soglia(fig, "y", g.SOGLIA_FP32, "soglia FP32 ≈ 3·10⁻⁸")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="margine minimo",
        line=dict(color=g.BLU, width=2.5), marker=g.marcatore(8), customdata=memoria,
        hovertemplate="margine %{y:.2e}<br>memoria ≈ %{customdata:.2e} token<extra></extra>"))
    g.asse_strati(fig, len(righe))
    g.scala_log(fig, "y", min(min(y), g.SOGLIA_FP32), max(max(y), g.SOGLIA_FP32 * 10))
    fig.update_layout(showlegend=False)
    dati = pd.DataFrame({"strato": x, "margine minimo 1-abar": y, "memoria massima (token)": memoria})
    return Grafico(
        "Margine di stabilità minimo per strato", fig,
        "Il polo discreto più vicino al cerchio unitario in ogni strato, su tutto il testo.",
        """
- **Asse y logaritmico.** Più il punto è in basso, più il polo è vicino a 1 e più lunga è la sua memoria.
- **Linea rossa tratteggiata** = soglia di FP32. Un punto **sotto** la linea vuol dire che in quello strato
  almeno un polo viene arrotondato a esattamente 1 (vedi il grafico successivo per quanti).
- **Memoria**: passando con il mouse si legge anche $1/(1-\\bar a)$ in token.
- **Esempio di lettura:** un punto a $10^{-12}$ è un modo che, in aritmetica esatta, dimenticherebbe dopo circa
  $10^{12}$ token; in FP32 invece non dimentica mai, perché il suo $\\bar a$ è diventato 1.
""",
        dati,
    )


def _grafico_su_uno(righe):
    x, y, conteggi = (valori(righe, k) for k in ("strato", "perc_1", "su_1"))
    fig = g.nuova_figura("Strato", "Poli arrotondati a 1  (% dello strato)", hover_unificato=False)
    fig.add_trace(go.Bar(
        x=x, y=y, name="poli su 1", marker=dict(color=g.BLU, line_width=0), customdata=conteggi,
        hovertemplate="strato %{x}<br>%{customdata:,} poli su 1 (%{y:.4%})<extra></extra>"))
    fig.update_yaxes(tickformat=".2%", rangemode="tozero")
    g.asse_strati(fig, len(righe))
    if not any(y):
        fig.add_annotation(text="Nessun polo arrotondato a 1 su questo testo", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    dati = pd.DataFrame({"strato": x, "poli su 1": conteggi, "quota dello strato": y})
    return Grafico(
        "Poli arrotondati a 1 in FP32, per strato", fig,
        "Quota dei poli discreti di ogni strato (token × canali × stati) che FP32 rende marginalmente stabili.",
        """
- **Altezza della barra** = percentuale dei poli dello strato che valgono esattamente 1 in FP32.
- **Barre concentrate in pochi strati** indicano dove il modello tiene la memoria più lunga.
- Il numero dipende dal testo: testi diversi producono $\\Delta$ diversi, quindi poli diversi.
- **Esempio di lettura:** una barra a 2% in uno strato da 1536 canali × 16 stati × 200 token significa circa
  98 000 poli che, in quello strato, hanno smesso di dimenticare.
""",
        dati,
    )


def _grafico_distribuzione(istogrammi, n):
    conteggi = sum(istogrammi[i] for i in range(n))
    bordi = np.linspace(LOG_MIN, LOG_MAX, N_BIN + 1)
    centri = (bordi[:-1] + bordi[1:]) / 2
    quota = conteggi / conteggi.sum()
    visibili = conteggi > 0
    fig = g.nuova_figura("Margine 1 − ā  (decadi)", "Quota dei poli  (scala logaritmica)",
                         log_y=True, hover_unificato=False)
    g.zona(fig, "x", LOG_MIN, math.log10(g.SOGLIA_FP32), "ā = 1 in FP32", colore=g.ROSSO_TENUE)
    g.linea_soglia(fig, "x", math.log10(g.SOGLIA_FP32), "soglia FP32")
    fig.add_trace(go.Bar(
        x=centri[visibili], y=quota[visibili], width=PASSO * 0.9, name="poli",
        marker=dict(color=g.BLU, line_width=0),
        customdata=np.stack([10 ** bordi[:-1], 10 ** bordi[1:], conteggi], axis=1)[visibili],
        hovertemplate="1 − ā tra %{customdata[0]:.1e} e %{customdata[1]:.1e}<br>"
                      "%{customdata[2]:,.0f} poli (%{y:.2e} del totale)<extra></extra>"))
    g.asse_decadi(fig, "x", LOG_MIN, LOG_MAX)
    fig.update_xaxes(range=[LOG_MIN - 0.3, LOG_MAX + 0.3])
    g.scala_log(fig, "y", quota[visibili].min(), quota[visibili].max())
    dati = pd.DataFrame({"log10(1-abar) da": bordi[:-1], "log10(1-abar) a": bordi[1:],
                         "numero di poli": conteggi.astype(np.int64), "quota": quota})
    for i in range(n):
        dati[f"strato {i}"] = istogrammi[i].astype(np.int64)
    return Grafico(
        "Distribuzione dei margini di stabilità (tutti gli strati e i token)", fig,
        "Quanti poli discreti hanno margine 1 − ā in ogni intervallo di un quarto di decade.",
        """
- **Asse x in decadi:** verso sinistra i poli più vicini a 1 (margine piccolo, memoria lunga),
  a destra quelli vicini a 0 (memoria nulla).
- **Asse y logaritmico**, per vedere anche le code con pochissimi poli.
- **Tutto ciò che sta nella zona rossa** viene arrotondato a 1 in FP32. La barra più a sinistra raccoglie
  anche i margini sotto $10^{-16}$.
- **Esempio di lettura:** una barra alta $10^{-4}$ attorno a $10^{-9}$ significa che un polo su diecimila ha
  margine di circa un miliardesimo, cioè è già "su 1" in FP32.
- Il foglio Excel contiene anche la stessa distribuzione **separata per strato**.
""",
        dati,
    )


FASE = Fase(
    codice="2b",
    nome="Poli discreti ā e margini",
    breve="abar = exp(Δ·a), margini e poli su 1 (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
)
