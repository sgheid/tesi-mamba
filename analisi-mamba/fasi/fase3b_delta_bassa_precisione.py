import math

import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import (FORMATI, SOGLIA_FP16_ZERO, delta_baseline, descrivi_formato,
                               esegui_in_formato, indicatore_formato)
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Parametro, Risultato, valori
from nucleo.sonde import delta_modello

SPIEGAZIONE = r"""
**Cosa misura.** Quanto cambia il passo $\Delta$ quando il modello lavora con i pesi in FP16 o BF16.
Lo stesso testo passa due volte nel modello: prima in FP32 (la baseline), poi nella copia a bassa
precisione. I due $\Delta$ vengono confrontati elemento per elemento, token per token e canale per canale.

**Perché conta.** In transformers $\Delta = \mathrm{softplus}(W_\Delta\,u_t + b_\Delta)$ viene calcolato
**nel formato dei pesi**: prodotto, somma del bias e softplus sono in FP16/BF16. In più l'ingresso $u_t$
ha già attraversato tutti gli strati precedenti in bassa precisione, quindi l'errore si accumula con la
profondità. In FP16 c'è un effetto in più: ogni $\Delta$ sotto $2^{-25} \approx 3\cdot10^{-8}$ (metà del
più piccolo subnormale) diventa **esattamente 0**, e con $\Delta = 0$ il token non entra nello stato
e $\bar a = 1$.

**Esempio.** Un canale con $\Delta = 0.0123$ in FP32 dà in FP16 un valore tra i numeri rappresentabili più
vicini, distanti $2^{-17} \approx 7.6\cdot10^{-6}$: l'errore relativo per il solo arrotondamento finale è
fino a $3\cdot10^{-4}$, e in BF16 (spaziatura $2^{-14}$) fino a $2.5\cdot10^{-3}$. Un $\Delta$ FP32 di
$10^{-8}$ in FP16 vale 0: errore relativo del 100%.
"""


# Quantile esatto con kthvalue (torch.quantile ha un limite di 16M elementi)
def _quantile(x, q):
    if x.numel() == 0:
        return float("nan")
    return x.kthvalue(max(1, math.ceil(q * x.numel()))).values.item()


def esegui(ctx):
    nome = ctx.parametri.get("formato", "FP16")
    tipo = FORMATI[nome]
    n = len(ctx.model.backbone.layers)
    base = delta_baseline(ctx, lambda f, m: ctx.avanzamento(0.45 * f, m))
    ctx.avanzamento(0.5, f"carico la copia in {nome}")
    dati = {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            d_r = delta_modello(mixer, uscita)
            d_b = base.pop(i).to(d_r.device)
            finiti = torch.isfinite(d_r)
            valido = finiti & (d_b > 0)
            err = ((d_r.float() - d_b).abs() / d_b)[valido]
            zeri = d_r == 0
            dati[i] = dict(strato=i, err_mediana=_quantile(err, 0.5), err_p99=_quantile(err, 0.99),
                           err_max=err.max().item() if err.numel() else float("nan"),
                           zeri=int(zeri.sum().item()),
                           zeri_attesi=int((zeri & (d_b < SOGLIA_FP16_ZERO)).sum().item()),
                           d_min_base=d_b.min().item(),
                           non_finiti=int((~finiti).sum().item()),
                           non_finiti_ingresso=int((~torch.isfinite(ingresso[0])).sum().item()),
                           valori=d_r.numel())
            ctx.avanzamento(0.5 + 0.5 * (i + 1) / n, f"{nome}, strato {i + 1} di {n}")
        return hook

    esito = esegui_in_formato(ctx, tipo, crea_hook)
    righe = [dati[i] for i in range(n)]
    zeri, attesi, non_finiti, totale = (sum(valori(righe, k)) for k in ("zeri", "zeri_attesi", "non_finiti", "valori"))
    validi = [r for r in righe if not math.isnan(r["err_p99"])]
    peggiore = max(validi, key=lambda r: r["err_p99"]) if validi else righe[0]
    basso = min(validi or righe, key=lambda r: r["err_mediana"])
    alto = max(validi or righe, key=lambda r: r["err_mediana"])
    controllo = tipo == torch.float32
    tutto_zero = all(r["err_max"] == 0 for r in righe) and zeri == 0 and non_finiti == 0

    indicatori = [
        indicatore_formato(nome, ctx.dispositivo, esito),
        Indicatore("Valori di Δ confrontati", fmt_int(totale),
                   f"{ctx.n_token} token × {ctx.info_modello['canali']} canali × {n} strati"),
        Indicatore("Errore tipico su Δ, da / a", f"{fmt_sci(basso['err_mediana'])} / {fmt_sci(alto['err_mediana'])}",
                   f"mediana per strato: la più bassa nello strato {basso['strato']}, la più alta nello strato "
                   f"{alto['strato']} (strato 0: {fmt_sci(righe[0]['err_mediana'])}, ultimo: {fmt_sci(righe[-1]['err_mediana'])})"
                   if alto["err_mediana"] > 0 else "nessuna differenza in nessuno strato"),
        Indicatore("Errore al 99° percentile più alto", fmt_sci(peggiore["err_p99"]),
                   f"strato {peggiore['strato']}: l'1% dei Δ sbaglia di più"),
        Indicatore("Δ diventati 0", fmt_int(zeri),
                   (f"{fmt_perc(zeri / totale)} dei Δ; {fmt_int(attesi)} avevano già in FP32 Δ < 2⁻²⁵"
                    if zeri else "nessun token escluso dallo stato"),
                   "attenzione" if zeri else "ok"),
        Indicatore("Valori non finiti (inf/NaN)", fmt_int(non_finiti),
                   "nessun overflow" if non_finiti == 0 else "esclusi dagli errori: le attivazioni superano il formato",
                   "ok" if non_finiti == 0 else "attenzione"),
    ]
    if controllo:
        indicatori.append(Indicatore("Controllo FP32", "superato" if tutto_zero else "NON superato",
                                     "due passaggi in FP32 devono dare Δ identici", "ok" if tutto_zero else "attenzione"))

    evidenzia = {peggiore["strato"]: "P99 più alto"}
    piu_zeri = max(righe, key=lambda r: r["zeri"])
    if zeri and piu_zeri["strato"] != peggiore["strato"]:
        evidenzia[piu_zeri["strato"]] = "più Δ diventati 0"

    return Risultato(
        codice="3b",
        titolo=f"Fase 3b - Passo Delta con i pesi in {nome}",
        descrizione="Delta del formato ridotto confrontato elemento per elemento (token × canale) con la baseline FP32.",
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("err_mediana", "Errore tipico su Delta", "mediana, relativo", "sci"),
                 Colonna("err_p99", "Errore al 99° percentile", "relativo", "sci"),
                 Colonna("err_max", "Errore massimo su Delta", "relativo", "sci"),
                 Colonna("d_min_base", "Delta minimo FP32", "baseline", "sci"),
                 Colonna("zeri", "Delta diventati 0", "numero", "int"),
                 Colonna("zeri_attesi", "Delta diventati 0", "con Delta FP32 < 2^-25", "int"),
                 Colonna("non_finiti", "Non finiti", "inf/NaN in Delta", "int")],
        righe=righe,
        indicatori=indicatori,
        legenda=[("Errore tipico su Delta", "Mediana di |Delta ridotto − Delta FP32| / Delta FP32 su token e canali."),
                 ("Errore al 99° percentile", "Solo l'1% dei Delta dello strato ha un errore relativo più grande."),
                 ("Errore massimo su Delta", "Il caso peggiore dello strato: vale 1 (100%) se un Delta diventa 0."),
                 ("Delta minimo FP32", "Il Delta più piccolo della baseline: sotto 2^-25 FP16 lo azzera."),
                 ("Delta diventati 0", "Token × canali con Delta = 0 nel formato ridotto: il token non entra nello stato."),
                 ("con Delta FP32 < 2^-25", "Di questi, quanti erano già sotto la soglia di FP16 nella baseline."),
                 ("Non finiti", "Delta inf o NaN nel formato ridotto (overflow): contati ed esclusi dagli errori.")],
        grafici=[_grafico_profondita(righe), _grafico_zeri(righe)],
        evidenzia=evidenzia,
        note_formato=descrivi_formato(nome, ctx.dispositivo) + "; baseline FP32",
    )


def _grafico_profondita(righe):
    x = valori(righe, "strato")
    serie = [("err_max", "errore massimo", dict(color=g.BLU_CHIARO, width=1.5, dash="dot"), "lines"),
             ("err_mediana", "errore tipico (mediana)", dict(color=g.BLU_CHIARO, width=2), "lines"),
             ("err_p99", "99° percentile", dict(color=g.BLU, width=2.5), "lines+markers")]
    fig = g.nuova_figura("Strato", "Errore relativo su Δ  (scala logaritmica)", log_y=True)
    positivi = [r[k] for r in righe for k, *_ in serie if r[k] > 0]
    if positivi:
        for chiave, nome, linea, modo in serie:
            fig.add_trace(go.Scatter(x=x, y=[r[chiave] if r[chiave] > 0 else None for r in righe], mode=modo,
                                     name=nome, line=linea, marker=g.marcatore(8),
                                     hovertemplate=f"{nome} %{{y:.2e}}<extra></extra>"))
        g.scala_log(fig, "y", min(positivi), max(positivi))
    else:
        fig.add_annotation(text="Nessuna differenza dalla baseline (controllo FP32)", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    g.asse_strati(fig, len(righe))
    dati = pd.DataFrame({"strato": x, "errore mediano": valori(righe, "err_mediana"),
                         "errore 99 percentile": valori(righe, "err_p99"), "errore massimo": valori(righe, "err_max")})
    return Grafico(
        "Errore su Δ in funzione della profondità", fig,
        "Per ogni strato, l'errore relativo tipico, al 99° percentile e massimo tra il Δ a bassa precisione e quello FP32.",
        """
- **Asse y logaritmico.** Una retta che sale vuol dire un errore che cresce di un fattore costante a ogni strato.
- **Linea chiara** = errore tipico; **linea scura** = 99° percentile, cioè i Δ più colpiti escluso l'1% estremo;
  **punteggiata** = massimo, che vale 1 quando un Δ diventa 0.
- Allo strato 0 l'errore viene solo da $W_\\Delta$, $b_\\Delta$ e dalle operazioni in bassa precisione; negli strati
  successivi si aggiunge quello già accumulato nell'ingresso.
- **Esempio di lettura:** mediana da $10^{-3}$ allo strato 0 a $10^{-2}$ all'ultimo vuol dire che, in fondo al
  modello, il Δ tipico è sbagliato dell'1% invece che dello 0.1%.
""",
        dati,
    )


def _grafico_zeri(righe):
    x = valori(righe, "strato")
    attesi = valori(righe, "zeri_attesi")
    altri = [r["zeri"] - r["zeri_attesi"] for r in righe]
    fig = g.nuova_figura("Strato", "Δ diventati 0  (numero)", hover_unificato=False)
    fig.add_trace(go.Bar(x=x, y=attesi, name="già sotto 2⁻²⁵ in FP32", marker=dict(color=g.BLU, line_width=0),
                         hovertemplate="strato %{x}: %{y:,} già sotto la soglia<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=altri, name="spinti sotto la soglia dall'errore", marker=dict(color=g.BLU_CHIARO, line_width=0),
                         hovertemplate="strato %{x}: %{y:,} spinti sotto la soglia<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(rangemode="tozero")
    g.asse_strati(fig, len(righe))
    if not any(valori(righe, "zeri")):
        fig.add_annotation(text="Nessun Δ diventato 0 in questo formato", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color=g.TENUE))
    dati = pd.DataFrame({"strato": x, "Delta diventati 0": valori(righe, "zeri"),
                         "gia' sotto 2^-25 in FP32": attesi, "spinti sotto la soglia": altri,
                         "Delta minimo FP32": valori(righe, "d_min_base")})
    return Grafico(
        "Δ diventati 0, per strato", fig,
        "Quanti valori di Δ (token × canali) il formato ridotto arrotonda a esattamente 0.",
        """
- **Altezza della barra** = numero di coppie token × canale con $\\Delta = 0$: per quel token il canale non riceve
  ingresso e i suoi 16 poli valgono esattamente $\\bar a = 1$.
- **Blu scuro** = Δ che già in FP32 erano sotto $2^{-25}$: li azzera il solo formato. **Blu chiaro** = Δ che in FP32
  erano sopra la soglia e ci sono finiti per l'errore accumulato.
- Succede solo in FP16: BF16 ha lo stesso intervallo di esponenti di FP32 e non azzera valori così piccoli.
- **Esempio di lettura:** una barra a 300 in uno strato da 1536 canali × 211 token vuol dire circa un valore su mille
  che smette di far entrare il token nello stato.
""",
        dati,
    )


FASE = Fase(
    codice="3b",
    nome="Passo Δ con pesi FP16/BF16",
    breve="Delta a bassa precisione contro la baseline FP32 (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[Parametro("formato", "Formato dei pesi", list(FORMATI), "FP16",
                         "Formato in cui viene caricata la seconda copia. FP32 (controllo) deve dare errori nulli.")],
)
