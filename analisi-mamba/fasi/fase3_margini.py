import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import confronto as cf
from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import FORMATI, delta_baseline, descrivi_formato, indicatore_formato, passaggio_ridotto
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Risultato, valori
from nucleo.sonde import delta_fp32, delta_fp64, esegui_con_sonde, ingresso_delta

# Istogramma dei margini: log10(1 - abar) da -16 a 0, a quarti di decade
LOG_MIN, LOG_MAX, PASSO = -16.0, 0.0, 0.25
N_BIN = int((LOG_MAX - LOG_MIN) / PASSO)
# Istogramma dell'errore relativo sul margine: log10 da -12 a 2 a centesimi di decade (per la mediana)
ERR_MIN, ERR_MAX, ERR_PASSO = -12.0, 2.0, 0.01
N_ERR = int(round((ERR_MAX - ERR_MIN) / ERR_PASSO))
# Istogramma dei margini della baseline vicino alla soglia FP32: log10(1 - abar) da -10 a -5
VIC_MIN, VIC_MAX, VIC_PASSO = -10.0, -5.0, 0.05
N_VIC = int(round((VIC_MAX - VIC_MIN) / VIC_PASSO))

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

**Confronto con FP16/BF16.** Con un confronto attivo, per ogni token, canale e stato si confronta il polo del modello
FP32 con quello della copia con i pesi in FP16 o BF16. In entrambi $\bar a$ è calcolato in FP32, come fa il modello,
ma nella copia a partire da $\Delta$ e $a$ del formato ridotto (Fasi 2 e 1); i margini sono calcolati esattamente in
FP64 con $-\mathrm{expm1}(\Delta a)$ dal $\Delta$ di ciascun modello. Se $\Delta$ o $a$ si spostano, un polo vicino
alla soglia $2^{-25}$ può attraversarla in un senso o nell'altro: un polo con margine $4\cdot10^{-8}$ il cui $\Delta$
in FP16 diminuisce del 30% scende a $2.8\cdot10^{-8}$ e diventa 1 (polo **nuovo**); uno con margine $2.5\cdot10^{-8}$
il cui $\Delta$ cresce del 25% sale a $3.1\cdot10^{-8}$ e torna sotto 1 (polo **perso**). In FP16 c'è anche una seconda
causa, diversa: con $\Delta = 0$ vale $\bar a = 1$ qualunque sia $a$. Le due cause vengono contate separatamente.
Separare $\Delta$ (Fase 2) da $\bar a$ serve proprio a capire se un polo su 1 dipende da $a$ oppure da $\Delta$.
"""

COLONNE = [Colonna("strato", "Strato", "", "int"),
           Colonna("margine", "Polo più vicino a 1", "margine 1 - abar", "sci"),
           Colonna("memoria", "Memoria massima", "token ≈ 1/(1-abar)", "sci"),
           Colonna("su_1", "Poli su 1 in FP32", "numero", "int"),
           Colonna("perc_1", "Poli su 1 in FP32", "% dello strato", "perc"),
           Colonna("in_0", "Poli in 0 in FP32", "numero", "int")]

LEGENDA = [("Polo più vicino a 1", "Il margine di stabilità più piccolo dello strato, calcolato esattamente (FP64)."),
           ("Memoria massima", "Quanti token servono a quel modo per dimenticare: circa 1/(1 - abar)."),
           ("Poli su 1 in FP32", "Poli che FP32 arrotonda esattamente a 1: da asintoticamente a marginalmente stabili."),
           ("% dello strato", "Quota dei poli dello strato (token x canali x stati) arrotondati a 1."),
           ("Poli in 0 in FP32", "Modi che si azzerano a ogni token: nessuna memoria, nessun rischio per la stabilità.")]


# Per ogni strato del modello FP32: abar in FP32 come il modello, margine 1 - abar esatto in FP64, a blocchi di token
def _margini(ctx):
    n = len(ctx.model.backbone.layers)
    dati, istogrammi = {}, {}

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
    return [dati[i] for i in range(n)], istogrammi


def _sintesi(righe):
    totale, su_1, in_0, fuori = (sum(valori(righe, k)) for k in ("totale", "su_1", "in_0", "fuori"))
    piu_vicino = min(righe, key=lambda r: r["margine"])
    peggiori = [r for r in sorted(righe, key=lambda r: -r["su_1"]) if r["su_1"] > 0][:3]
    return totale, su_1, in_0, fuori, piu_vicino, peggiori


def _strati_peggiori(peggiori):
    return ", ".join(str(r["strato"]) for r in peggiori) if peggiori else "nessuno"


def esegui(ctx):
    nome = cf.formato_confronto(ctx)
    return _solo_fp32(ctx) if nome is None else _con_confronto(ctx, nome)


# Solo FP32: il calcolo della vecchia Fase 2b, senza seconda copia
def _solo_fp32(ctx):
    righe, istogrammi = _margini(ctx)
    n = len(righe)
    totale, su_1, in_0, fuori, piu_vicino, peggiori = _sintesi(righe)

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
        Indicatore("Strati con più poli su 1", _strati_peggiori(peggiori),
                   " · ".join(f"{r['strato']}: {fmt_int(r['su_1'])}" for r in peggiori)),
        Indicatore("Arrotondati a 0 in FP32", fmt_int(in_0),
                   f"{fmt_perc(in_0 / totale)}: senza memoria, innocui per la stabilità"),
    ]

    return Risultato(
        codice="3",
        titolo="Fase 3 - Poli discreti  abar = exp(Delta * a)",
        descrizione="Calcolati in FP32 come nel modello; il margine esatto 1 - abar è calcolato in FP64.",
        colonne=COLONNE,
        righe=righe,
        indicatori=indicatori,
        legenda=LEGENDA,
        grafici=[_grafico_margine(righe), _grafico_su_uno(righe), _grafico_distribuzione(istogrammi, n)],
        evidenzia={r["strato"]: f"n. {k} per poli su 1" for k, r in enumerate(peggiori, start=1)},
        note_formato="FP32, con riferimento esatto in FP64 per il margine 1 - abar",
    )


# FP32 e formato ridotto affiancati, più i poli nuovi e persi della vecchia Fase 3c
def _con_confronto(ctx, nome):
    righe, istogrammi = _margini(cf.tratto(ctx, 0.0, 0.2))
    n = len(righe)
    passaggio = passaggio_ridotto(ctx, nome, cf.tratto(ctx, 0.2, 0.6).avanzamento)
    errori, righe_r, istogrammi_r = _errori(cf.tratto(ctx, 0.6, 1.0), nome, passaggio)
    totale, su_1, in_0, fuori, piu_vicino, peggiori = _sintesi(righe)
    _, su_1_r, in_0_r, fuori_r, piu_vicino_r, peggiori_r = _sintesi(righe_r)

    indicatori = [
        errori.indicatori[0],
        Indicatore("Poli discreti esaminati", fmt_int(totale),
                   f"{ctx.n_token} token × {ctx.info_modello['canali']} canali × "
                   f"{ctx.info_modello['stati']} stati × {n} strati"),
        cf.indicatore("Fuori dal cerchio (ā > 1)", fuori, fuori_r, fmt_int, nome,
                      "nessun polo instabile" if fuori == fuori_r == 0 else "",
                      "ok" if fuori == fuori_r == 0 else "attenzione"),
        cf.indicatore("Arrotondati a 1 in FP32", su_1, su_1_r, fmt_int, nome,
                      f"{fmt_perc(su_1 / totale)} → {fmt_perc(su_1_r / totale)} dei poli: marginalmente stabili",
                      "attenzione" if su_1 or su_1_r else "ok"),
        cf.indicatore("Polo più vicino a 1", piu_vicino["margine"], piu_vicino_r["margine"], fmt_sci, nome,
                      f"strato {piu_vicino['strato']} → {piu_vicino_r['strato']}"
                      + (": margine 0, cioè Δ = 0" if piu_vicino_r["margine"] == 0 else "")),
        Indicatore("Strati con più poli su 1", f"{_strati_peggiori(peggiori)} → {_strati_peggiori(peggiori_r)}",
                   f"FP32 → {nome} · " + " · ".join(f"{r['strato']}: {fmt_int(r['su_1'])}" for r in peggiori)
                   + " → " + " · ".join(f"{r['strato']}: {fmt_int(r['su_1'])}" for r in peggiori_r)),
        cf.indicatore("Arrotondati a 0 in FP32", in_0, in_0_r, fmt_int, nome,
                      "senza memoria, innocui per la stabilità"),
    ] + errori.indicatori[1:]

    return Risultato(
        codice="3",
        titolo=f"Fase 3 - Poli discreti  abar = exp(Delta * a), FP32 contro {nome}",
        descrizione="Calcolati in FP32 come nel modello; il margine esatto 1 - abar è calcolato in FP64. "
                    + errori.descrizione,
        colonne=cf.colonne_affiancate(COLONNE, nome) + errori.colonne[1:],
        righe=cf.righe_unite(righe, righe_r, errori.righe),
        indicatori=indicatori,
        legenda=LEGENDA + errori.legenda
                + [(f"FP32 e {nome}", f"Le grandezze della Solo FP32 compaiono due volte: nel modello FP32 e nella copia "
                                      f"con i pesi in {nome}, dove abar è calcolato in FP32 dal Delta e da a della copia.")],
        grafici=[_grafico_margine(righe, righe_r, nome), _grafico_su_uno(righe, righe_r, nome),
                 _grafico_distribuzione(istogrammi, n, istogrammi_r, nome)] + errori.grafici,
        evidenzia=cf.evidenze_unite({r["strato"]: f"n. {k} per poli su 1" for k, r in enumerate(peggiori, start=1)},
                                    errori.evidenzia),
        note_formato="FP32 e " + errori.note_formato,
    )


def _grafico_margine(righe, righe_r=None, nome=None):
    x, y, memoria = (valori(righe, k) for k in ("strato", "margine", "memoria"))
    fig = g.nuova_figura("Strato", "Margine minimo 1 − ā  (scala logaritmica)", log_y=True)
    tutti = y + ([v for v in valori(righe_r, "margine") if v > 0] if righe_r else [])
    basso = min(min(tutti) / 10, g.SOGLIA_FP32 / 10)
    g.zona(fig, "y", basso, g.SOGLIA_FP32, "zona in cui FP32 arrotonda ā a 1", colore=g.ROSSO_TENUE)
    g.linea_soglia(fig, "y", g.SOGLIA_FP32, "soglia FP32 ≈ 3·10⁻⁸")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="margine minimo FP32" if righe_r else "margine minimo",
        line=dict(color=g.BLU, width=2.5), marker=g.marcatore(8), customdata=memoria,
        hovertemplate="margine %{y:.2e}<br>memoria ≈ %{customdata:.2e} token<extra></extra>"))
    dati = pd.DataFrame({"strato": x, "margine minimo 1-abar": y, "memoria massima (token)": memoria})
    if righe_r:
        y_r = valori(righe_r, "margine")
        fig.add_trace(go.Scatter(
            x=x, y=[v if v > 0 else None for v in y_r], mode="lines+markers", name=f"margine minimo {nome}",
            line=dict(color=g.ARANCIO, width=2), marker=g.marcatore(7, g.ARANCIO), customdata=valori(righe_r, "memoria"),
            hovertemplate=f"{nome}: margine %{{y:.2e}}<br>memoria ≈ %{{customdata:.2e}} token<extra></extra>"))
        dati = dati.rename(columns=lambda c: c if c == "strato" else f"{c} FP32")
        dati[f"margine minimo 1-abar {nome}"] = y_r
        dati[f"memoria massima (token) {nome}"] = valori(righe_r, "memoria")
    g.asse_strati(fig, len(righe))
    g.scala_log(fig, "y", min(min(tutti), g.SOGLIA_FP32), max(max(tutti), g.SOGLIA_FP32 * 10))
    fig.update_layout(showlegend=bool(righe_r))
    come_leggere = """
- **Asse y logaritmico.** Più il punto è in basso, più il polo è vicino a 1 e più lunga è la sua memoria.
- **Linea rossa tratteggiata** = soglia di FP32. Un punto **sotto** la linea vuol dire che in quello strato
  almeno un polo viene arrotondato a esattamente 1 (vedi il grafico successivo per quanti).
- **Memoria**: passando con il mouse si legge anche $1/(1-\\bar a)$ in token.
- **Esempio di lettura:** un punto a $10^{-12}$ è un modo che, in aritmetica esatta, dimenticherebbe dopo circa
  $10^{12}$ token; in FP32 invece non dimentica mai, perché il suo $\\bar a$ è diventato 1.
"""
    if righe_r:
        come_leggere += f"""- **Blu = FP32, arancione = {nome}**: il margine minimo dello stesso strato nella copia in {nome}. Dove manca il
  punto arancione il margine minimo vale **0**: in quello strato il formato ha azzerato almeno un Δ (solo FP16),
  e il polo vale 1 indipendentemente da $a$. Un punto arancione più in basso di quello blu = il formato ha avvicinato
  a 1 il polo più lento dello strato.
"""
    return Grafico(
        "Margine di stabilità minimo per strato", fig,
        "Il polo discreto più vicino al cerchio unitario in ogni strato, su tutto il testo."
        + (f" In arancione la copia in {nome}." if righe_r else ""),
        come_leggere,
        dati,
    )


def _grafico_su_uno(righe, righe_r=None, nome=None):
    x, y, conteggi = (valori(righe, k) for k in ("strato", "perc_1", "su_1"))
    fig = g.nuova_figura("Strato", "Poli arrotondati a 1  (% dello strato)", hover_unificato=False)
    fig.add_trace(go.Bar(
        x=x, y=y, name="poli su 1 FP32" if righe_r else "poli su 1", marker=dict(color=g.BLU, line_width=0),
        customdata=conteggi, hovertemplate="strato %{x}<br>%{customdata:,} poli su 1 (%{y:.4%})<extra></extra>"))
    dati = pd.DataFrame({"strato": x, "poli su 1": conteggi, "quota dello strato": y})
    vuoto = not any(y)
    if righe_r:
        y_r, conteggi_r = valori(righe_r, "perc_1"), valori(righe_r, "su_1")
        fig.add_trace(go.Bar(
            x=x, y=y_r, name=f"poli su 1 {nome}", marker=dict(color=g.ARANCIO, line_width=0), customdata=conteggi_r,
            hovertemplate=f"strato %{{x}}<br>%{{customdata:,}} poli su 1 in {nome} (%{{y:.4%}})<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
        dati = dati.rename(columns=lambda c: c if c == "strato" else f"{c} FP32")
        dati[f"poli su 1 {nome}"], dati[f"quota dello strato {nome}"] = conteggi_r, y_r
        vuoto = vuoto and not any(y_r)
    fig.update_yaxes(tickformat=".2%", rangemode="tozero")
    g.asse_strati(fig, len(righe))
    if vuoto:
        fig.add_annotation(text="Nessun polo arrotondato a 1 su questo testo", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    come_leggere = """
- **Altezza della barra** = percentuale dei poli dello strato che valgono esattamente 1 in FP32.
- **Barre concentrate in pochi strati** indicano dove il modello tiene la memoria più lunga.
- Il numero dipende dal testo: testi diversi producono $\\Delta$ diversi, quindi poli diversi.
- **Esempio di lettura:** una barra a 2% in uno strato da 1536 canali × 16 stati × 200 token significa circa
  98 000 poli che, in quello strato, hanno smesso di dimenticare.
"""
    if righe_r:
        come_leggere += f"""- **Barra blu = FP32, barra arancione accanto = {nome}** nello stesso strato. Un'arancione molto più alta della blu
  vuol dire che il formato ha aggiunto poli su 1: quanti per $\\Delta = 0$ e quanti per margine sotto soglia lo dice il
  grafico "Poli che cambiano classe" più sotto. Per esempio sul 130M in FP16 i poli su 1 passano da circa 51 000 a 181 000.
"""
    return Grafico(
        "Poli arrotondati a 1 in FP32, per strato", fig,
        "Quota dei poli discreti di ogni strato (token × canali × stati) che FP32 rende marginalmente stabili."
        + (f" In arancione la copia in {nome}." if righe_r else ""),
        come_leggere,
        dati,
    )


def _grafico_distribuzione(istogrammi, n, istogrammi_r=None, nome=None):
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
        x=centri[visibili], y=quota[visibili], width=PASSO * 0.9, name="poli FP32" if istogrammi_r else "poli",
        marker=dict(color=g.BLU, line_width=0),
        customdata=np.stack([10 ** bordi[:-1], 10 ** bordi[1:], conteggi], axis=1)[visibili],
        hovertemplate="1 − ā tra %{customdata[0]:.1e} e %{customdata[1]:.1e}<br>"
                      "%{customdata[2]:,.0f} poli (%{y:.2e} del totale)<extra></extra>"))
    dati = pd.DataFrame({"log10(1-abar) da": bordi[:-1], "log10(1-abar) a": bordi[1:],
                         "numero di poli": conteggi.astype(np.int64), "quota": quota})
    quote_visibili = quota[visibili]
    if istogrammi_r:
        conteggi_r = sum(istogrammi_r[i] for i in range(n))
        quota_r = conteggi_r / conteggi_r.sum()
        vis_r = conteggi_r > 0
        fig.add_trace(go.Bar(
            x=centri[vis_r], y=quota_r[vis_r], width=PASSO * 0.45, name=f"poli {nome}",
            marker=dict(color=g.ARANCIO, line_width=0), customdata=conteggi_r[vis_r],
            hovertemplate=f"%{{customdata:,.0f}} poli in {nome} (%{{y:.2e}} del totale)<extra></extra>"))
        fig.update_layout(barmode="overlay")
        dati[f"numero di poli {nome}"], dati[f"quota {nome}"] = conteggi_r.astype(np.int64), quota_r
        quote_visibili = np.concatenate([quote_visibili, quota_r[vis_r]])
    g.asse_decadi(fig, "x", LOG_MIN, LOG_MAX)
    fig.update_xaxes(range=[LOG_MIN - 0.3, LOG_MAX + 0.3])
    g.scala_log(fig, "y", quote_visibili.min(), quote_visibili.max())
    for i in range(n):
        dati[f"strato {i}"] = istogrammi[i].astype(np.int64)
    come_leggere = """
- **Asse x in decadi:** verso sinistra i poli più vicini a 1 (margine piccolo, memoria lunga),
  a destra quelli vicini a 0 (memoria nulla).
- **Asse y logaritmico**, per vedere anche le code con pochissimi poli.
- **Tutto ciò che sta nella zona rossa** viene arrotondato a 1 in FP32. La barra più a sinistra raccoglie
  anche i margini sotto $10^{-16}$.
- **Esempio di lettura:** una barra alta $10^{-4}$ attorno a $10^{-9}$ significa che un polo su diecimila ha
  margine di circa un miliardesimo, cioè è già "su 1" in FP32.
- Il foglio Excel contiene anche la stessa distribuzione **separata per strato** (modello FP32).
"""
    if istogrammi_r:
        come_leggere += f"""- **Barre blu = FP32, barre arancioni più strette = {nome}**, sugli stessi intervalli. La barra arancione più a sinistra
  in FP16 raccoglie anche i margini esattamente 0 (Δ = 0): è lì che compaiono i poli azzerati dal formato. Altrove le
  due distribuzioni quasi coincidono, perché l'errore sul margine (qualche millesimo) è molto più piccolo di un quarto
  di decade.
"""
    return Grafico(
        "Distribuzione dei margini di stabilità (tutti gli strati e i token)", fig,
        "Quanti poli discreti hanno margine 1 − ā in ogni intervallo di un quarto di decade."
        + (f" In arancione la copia in {nome}." if istogrammi_r else ""),
        come_leggere,
        dati,
    )


# Poli nuovi e persi a bassa precisione: il calcolo della vecchia Fase 3c, sul passaggio ridotto condiviso.
# In più, nello stesso ciclo, le grandezze della Solo FP32 per la copia (margine minimo, ā = 0, ā > 1, istogramma)

# Mediana dall'istogramma: errori nulli contati a parte, poi la cumulata dei bin (centro del bin, in log)
def _mediana(zeri, isto):
    totale = zeri + isto.sum()
    if totale == 0:
        return float("nan")
    if zeri >= totale / 2:
        return 0.0
    k = int(np.searchsorted(np.cumsum(isto), totale / 2 - zeri))
    return 10 ** (ERR_MIN + (k + 0.5) * ERR_PASSO)


def _errori(ctx, nome, passaggio):
    tipo = FORMATI[nome]
    n = len(ctx.model.backbone.layers)
    A_log_base = {i: l.mixer.A_log.detach().float().cpu() for i, l in enumerate(ctx.model.backbone.layers)}
    base = delta_baseline(ctx, ctx.avanzamento)
    dati, isto_err, zeri_err, vicini_base, vicini_su_1 = {}, {}, {}, {}, {}
    righe_r, istogrammi_r = [], {}

    # Per ogni strato, a blocchi di token: abar in FP32 come il modello, margini esatti in FP64
    for i, strato in enumerate(passaggio["strati"]):
        d_r = strato["delta"].to(ctx.dispositivo)
        dev = d_r.device
        d_b = base.pop(i).to(dev)
        A_b = A_log_base.pop(i).to(dev)
        A_r = strato["A_log"].to(dev)
        a_b32, a_b64 = -torch.exp(A_b), -torch.exp(A_b.double())
        a_r32, a_r64 = -torch.exp(A_r.float()), -torch.exp(A_r.double())
        L, D = d_r.shape
        passo = max(1, 4_000_000 // (D * a_r32.shape[1]))
        c = dict(su_1_base=0, su_1_rid=0, nuovi_zero=0, nuovi_soglia=0, persi=0, non_finiti=0, err_nulli=0)
        isto = torch.zeros(N_ERR, dtype=torch.float64, device=dev)
        vb = torch.zeros(N_VIC, dtype=torch.float64, device=dev)
        v1 = torch.zeros(N_VIC, dtype=torch.float64, device=dev)
        margine_min_r, in_0_r, fuori_r = float("inf"), 0, 0
        isto_r = torch.zeros(N_BIN, dtype=torch.float64, device=dev)
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
            # Grandezze della Solo FP32 per la copia, con lo stesso istogramma dei margini
            m_r_finiti = m_r[torch.isfinite(m_r)]
            if m_r_finiti.numel():
                margine_min_r = min(margine_min_r, m_r_finiti.min().item())
            in_0_r += int((abar_r == 0.0).sum().item())
            fuori_r += int((abar_r > 1.0).sum().item())
            lg_r = torch.log10(m_r_finiti.clamp_min(1e-300)).clamp(LOG_MIN, LOG_MAX - 1e-9)
            isto_r += torch.histc(lg_r.float(), bins=N_BIN, min=LOG_MIN, max=LOG_MAX).double()
        isto_err[i] = isto.cpu().numpy()
        zeri_err[i] = c.pop("err_nulli")
        vicini_base[i], vicini_su_1[i] = vb.cpu().numpy(), v1.cpu().numpy()
        totale = L * a_r32.numel()
        dati[i] = dict(strato=i, **c, nuovi=c["nuovi_zero"] + c["nuovi_soglia"],
                       err_mediana=_mediana(zeri_err[i], isto_err[i]), totale=totale)
        righe_r.append(dict(strato=i, margine=margine_min_r,
                            memoria=1.0 / margine_min_r if margine_min_r > 0 else float("inf"),
                            su_1=c["su_1_rid"], perc_1=c["su_1_rid"] / totale, in_0=in_0_r, fuori=fuori_r,
                            totale=totale))
        istogrammi_r[i] = isto_r.cpu().numpy()
        ctx.avanzamento((i + 1) / n, f"{nome}, strato {i + 1} di {n}")

    esito = passaggio["esito"]
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

    ris = Risultato(
        codice="3",
        titolo=f"Fase 3 - Poli discreti e margini con i pesi in {nome}",
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
                 ("baseline FP32", "Nel modello con i pesi in FP32, con il Delta del modello (le operazioni di "
                                   "transformers) come riferimento del confronto. La colonna 'Poli su 1 in FP32 · FP32' "
                                   "usa il Delta ricalcolato della Solo FP32: sul 130M coincidono; una differenza "
                                   "verrebbe solo dal diverso calcolo di Delta."),
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
    return ris, righe_r, istogrammi_r


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
  (Fase 2) e su $a$ (Fase 1), quindi segue lo stesso andamento con la profondità.
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
    codice="3",
    nome="Margini di stabilità",
    breve="abar = exp(Δ·a), margini e poli su 1, anche nella copia FP16/BF16 (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[cf.PARAMETRO],
)
