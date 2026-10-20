import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_sci
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Risultato, valori

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
Quanti *token* corrispondano a $\tau$ lo dice il passo $\Delta$ della Fase 2a.
"""


# Poli continui letti dai pesi, strato per strato
def esegui(ctx):
    strati = ctx.model.backbone.layers
    n = len(strati)
    righe, log_abs = [], []
    with torch.no_grad():
        for i, layer in enumerate(strati):
            a = -torch.exp(layer.mixer.A_log.float())
            a_max = a.max().item()
            righe.append(dict(strato=i, a_min=a.min().item(), a_mediana=a.median().item(),
                              a_max=a_max, tau=1.0 / abs(a_max) if a_max != 0 else float("inf"),
                              instabili=int((a >= 0).sum().item()), totale=a.numel()))
            log_abs.append(torch.log10((-a).clamp_min(1e-45)).flatten().cpu().numpy())
            ctx.avanzamento((i + 1) / n, f"Strato {i + 1} di {n}")

    lento = max(righe, key=lambda r: r["a_max"])
    veloce = min(righe, key=lambda r: r["a_min"])
    instabili = sum(valori(righe, "instabili"))
    totale = sum(valori(righe, "totale"))
    decadi = math.log10(veloce["a_min"] / lento["a_max"])

    indicatori = [
        Indicatore("Poli continui esaminati", fmt_int(totale),
                   f"{n} strati × {ctx.info_modello['canali']} canali × {ctx.info_modello['stati']} stati"),
        Indicatore("Poli instabili (a ≥ 0)", fmt_int(instabili),
                   "tutti i modi sono asintoticamente stabili" if instabili == 0
                   else "esistono modi non stabili", "ok" if instabili == 0 else "attenzione"),
        Indicatore("Polo più lento", fmt_sci(lento["a_max"]),
                   f"strato {lento['strato']}, τ = {fmt_sci(lento['tau'])}"),
        Indicatore("Polo più veloce", fmt_sci(veloce["a_min"]), f"strato {veloce['strato']}"),
        Indicatore("Ampiezza dello spettro", f"{decadi:.1f} decadi",
                   "all'inizializzazione: 1.2 decadi (da −1 a −16)"),
    ]
    if lento["strato"] == veloce["strato"]:
        evidenzia = {lento["strato"]: "polo più lento e più veloce"}
    else:
        evidenzia = {lento["strato"]: "polo più lento", veloce["strato"]: "polo più veloce"}

    return Risultato(
        codice="1",
        titolo="Fase 1 - Poli continui  a = -exp(A_log)",
        descrizione="Letti dai pesi del modello: non dipendono dal testo.",
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("a_min", "Polo più veloce", "a_min", "sci"),
                 Colonna("a_mediana", "Polo tipico", "mediana", "sci"),
                 Colonna("a_max", "Polo più lento", "a_max", "sci"),
                 Colonna("tau", "Costante di tempo", "max = 1/|a_max|", "sci"),
                 Colonna("instabili", "Poli instabili", "a ≥ 0", "int")],
        righe=righe,
        indicatori=indicatori,
        legenda=[("Polo più veloce", "Il polo più negativo dello strato: il modo che si estingue più in fretta."),
                 ("Polo tipico", "La mediana dei poli dello strato: metà sono più veloci, metà più lenti."),
                 ("Polo più lento", "Il polo più vicino a zero: il modo con la memoria più lunga."),
                 ("Costante di tempo", "1/|a_max| in unità di tempo continuo; in token dipende da Delta (Fase 2a)."),
                 ("Poli instabili", "Poli con a ≥ 0. Per costruzione (a = -exp) devono essere zero.")],
        grafici=[_grafico_spettro(righe), _grafico_distribuzione(log_abs)],
        evidenzia=evidenzia,
    )


def _grafico_spettro(righe):
    x = valori(righe, "strato")
    veloce, tipico, lento = (valori(righe, k, abs) for k in ("a_min", "a_mediana", "a_max"))
    fig = g.nuova_figura("Strato", "|a|  (scala logaritmica)", log_y=True)
    g.zona(fig, "y", 1, 16, "inizializzazione S4D-Real: |a| da 1 a 16")
    g.fascia_min_mediana_max(fig, x, lento, tipico, veloce,
                             ("polo più veloce", "polo tipico (mediana)", "polo più lento"))
    g.asse_strati(fig, len(righe))
    g.scala_log(fig, "y", min(1, min(lento)), max(16, max(veloce)))
    dati = pd.DataFrame({"strato": x, "|a| polo piu' veloce": veloce,
                         "|a| mediana": tipico, "|a| polo piu' lento": lento})
    return Grafico(
        "Spettro dei poli continui per strato", fig,
        "Per ogni strato, la fascia va dal polo più lento (in basso) al più veloce (in alto); "
        "la linea è il polo tipico.",
        """
- **Asse y logaritmico:** ogni tacca è un fattore 10. Si riporta $|a|$ perché la scala logaritmica
  richiede valori positivi; il polo vero è $a = -|a|$.
- **In basso = memoria lunga.** Un punto a $10^{-5}$ è un modo con $\\tau = 10^{5}$.
- **La fascia grigia** è dove stavano tutti i poli prima dell'addestramento (S4D-Real: $a_n=-(n+1)$, $n=0\\dots15$).
  Quanto la fascia blu esce da quella grigia misura quanto l'addestramento ha allargato lo spettro.
- **Esempio di lettura:** se in uno strato il bordo inferiore scende a $10^{-6}$ e quello superiore sale a $10^{8}$,
  quello strato contiene sia modi che ricordano "per sempre" sia modi che dimenticano a ogni passo.
""",
        dati,
    )


# Istogramma di log10|a| a quarti di decade, su tutti gli strati
def _grafico_distribuzione(log_abs):
    tutti = np.concatenate(log_abs)
    passo = 0.25
    lo, hi = math.floor(tutti.min() / passo) * passo, math.ceil(tutti.max() / passo) * passo + passo
    conteggi, bordi = np.histogram(tutti, bins=np.arange(lo, hi + passo / 2, passo))
    centri = (bordi[:-1] + bordi[1:]) / 2
    quota = conteggi / conteggi.sum()
    fig = g.nuova_figura("|a|  (decadi)", "Quota dei poli", hover_unificato=False)
    g.zona(fig, "x", 0, math.log10(16), "inizializzazione")
    fig.add_trace(go.Bar(x=centri, y=quota, width=passo * 0.9, marker=dict(color=g.BLU, line_width=0),
                         customdata=np.stack([10 ** bordi[:-1], 10 ** bordi[1:], conteggi], axis=1),
                         hovertemplate="|a| tra %{customdata[0]:.2e} e %{customdata[1]:.2e}<br>"
                                       "%{customdata[2]:,} poli (%{y:.2%})<extra></extra>",
                         name="poli"))
    fig.update_yaxes(tickformat=".1%")
    g.asse_decadi(fig, "x", lo, hi)
    dati = pd.DataFrame({"log10|a| da": bordi[:-1], "log10|a| a": bordi[1:],
                         "numero di poli": conteggi, "quota": quota})
    return Grafico(
        "Distribuzione dei poli continui (tutti gli strati)", fig,
        "Quanti poli cadono in ogni intervallo di |a|, largo un quarto di decade.",
        """
- **Ogni barra** raccoglie i poli con $|a|$ in un intervallo di un quarto di decade (fattore $10^{0.25}\\approx 1.8$).
- **Altezza** = quota sul totale dei poli del modello. Passando con il mouse si legge anche il numero esatto.
- **Le code** sono la parte interessante: a sinistra i modi lentissimi (memoria lunga, margine piccolo),
  a destra quelli che dimenticano subito.
- **Esempio di lettura:** una barra alta 5% tra $10^{-1}$ e $10^{-0.75}$ significa che un polo su venti ha
  costante di tempo tra circa 5.6 e 10.
""",
        dati,
    )


FASE = Fase(
    codice="1",
    nome="Poli continui",
    breve="a = -exp(A_log), dai pesi (non usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=False,
    esegui=esegui,
)
