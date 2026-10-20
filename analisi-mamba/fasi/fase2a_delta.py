import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_sci
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Risultato, valori
from nucleo.sonde import delta_fp32, esegui_con_sonde, ingresso_delta

SPIEGAZIONE = r"""
**Cosa misura.** Mamba trasforma ogni modo continuo in un modo discreto campionandolo con un
passo $\Delta$. Il passo non è fisso: il modello lo calcola **a ogni token e per ogni canale**,
$\Delta = \mathrm{softplus}(W_\Delta\,u_t + b_\Delta)$. È ciò che rende il sistema tempo-variante (LTV).

**Perché conta.** $\Delta$ decide quanto "tempo continuo" passa a ogni token. Con $\Delta$ grande lo
stato si svuota e il canale riparte da capo; con $\Delta$ piccolo lo stato resta quasi invariato e il
token viene di fatto ignorato.

**Esempio.** Un modo con $a=-2$ campionato con $\Delta = 0.001$ dà $\bar a = e^{-0.002} \approx 0.998$:
dopo un token resta il 99.8% dello stato. Con $\Delta = 3$ dà $\bar a = e^{-6} \approx 0.0025$: lo stato
è praticamente azzerato. Stesso modo, comportamento opposto, deciso dal testo.
"""


# Delta di ogni strato, catturato dalle sonde mentre il modello legge il testo
def esegui(ctx):
    n = len(ctx.model.backbone.layers)
    dati, mediana_per_token = {}, {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            d = delta_fp32(mixer, ingresso_delta(mixer, uscita))
            d_min, d_max = d.min().item(), d.max().item()
            dati[i] = dict(strato=i, d_min=d_min, d_mediana=d.median().item(), d_max=d_max,
                           decadi=math.log10(d_max / d_min) if d_min > 0 else float("inf"),
                           valori=d.numel())
            mediana_per_token[i] = d.median(dim=1).values.cpu().numpy()
            ctx.avanzamento((i + 1) / n, f"Strato {i + 1} di {n}")
        return hook

    esegui_con_sonde(ctx.model, ctx.ids, crea_hook)
    righe = [dati[i] for i in range(n)]
    lento = min(righe, key=lambda r: r["d_mediana"])
    rapido = max(righe, key=lambda r: r["d_mediana"])
    canali = righe[0]["valori"] // ctx.n_token

    indicatori = [
        Indicatore("Valori di Δ per strato", fmt_int(righe[0]["valori"]),
                   f"{ctx.n_token} token × {canali} canali"),
        Indicatore("Passo tipico più piccolo", fmt_sci(lento["d_mediana"]),
                   f"strato {lento['strato']}: memoria più lunga"),
        Indicatore("Passo tipico più grande", fmt_sci(rapido["d_mediana"]),
                   f"strato {rapido['strato']}: si resetta di più"),
        Indicatore("Δ minimo / massimo del modello",
                   f"{fmt_sci(min(valori(righe, 'd_min')))} / {fmt_sci(max(valori(righe, 'd_max')))}"),
    ]
    return Risultato(
        codice="2a",
        titolo="Fase 2a - Passo di campionamento Delta",
        descrizione="Calcolato dal modello token per token: dipende dal testo letto.",
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("d_min", "Delta minimo", "(ignora il token)", "sci"),
                 Colonna("d_mediana", "Delta tipico", "mediana", "sci"),
                 Colonna("d_max", "Delta massimo", "(si resetta)", "sci"),
                 Colonna("decadi", "Selettività", "decadi max/min", "dec")],
        righe=righe,
        indicatori=indicatori,
        legenda=[("Delta", "Periodo di campionamento del canale a quel token: abar = exp(Delta * a)."),
                 ("Delta minimo", "Il token che lo strato 'salta' di più: lo stato resta quasi invariato."),
                 ("Delta tipico", "La mediana su tutti i token e i canali. Inizializzazione: tra 0.001 e 0.1."),
                 ("Delta massimo", "Il token su cui lo strato 'riparte da zero': lo stato si svuota."),
                 ("Selettività", "log10(Delta max / Delta min): di quanti ordini di grandezza varia il passo.")],
        grafici=[_grafico_fascia(righe), _grafico_mappa(mediana_per_token, n, ctx.token)],
        evidenzia={lento["strato"]: "passo tipico più piccolo", rapido["strato"]: "passo tipico più grande"},
    )


def _grafico_fascia(righe):
    x = valori(righe, "strato")
    d_min, d_med, d_max = (valori(righe, k) for k in ("d_min", "d_mediana", "d_max"))
    fig = g.nuova_figura("Strato", "Δ  (scala logaritmica)", log_y=True)
    g.zona(fig, "y", 0.001, 0.1, "inizializzazione: Δ da 0.001 a 0.1")
    g.fascia_min_mediana_max(fig, x, d_min, d_med, d_max, ("Δ massimo", "Δ tipico (mediana)", "Δ minimo"))
    g.asse_strati(fig, len(righe))
    g.scala_log(fig, "y", min(0.001, min(d_min)), max(0.1, max(d_max)))
    dati = pd.DataFrame({"strato": x, "Delta minimo": d_min, "Delta mediana": d_med, "Delta massimo": d_max})
    return Grafico(
        "Passo Δ per strato", fig,
        "Per ogni strato, la fascia va dal Δ più piccolo al più grande prodotto sul testo; la linea è il Δ tipico.",
        """
- **Asse y logaritmico**, una tacca per ogni fattore 10.
- **La fascia grigia** è l'intervallo in cui Mamba inizializza $\\Delta$ prima dell'addestramento.
- **Linea in basso = strato con memoria lunga**: a ogni token passa poco tempo, i modi si consumano lentamente.
- **Fascia molto alta = strato molto selettivo**: alcuni token lo fanno ripartire da zero, altri vengono saltati.
- **Esempio di lettura:** se uno strato ha mediana $10^{-3}$ e massimo $10^{1}$, di solito conserva lo stato
  ma su qualche token (spesso punteggiatura o inizio frase) lo azzera.
""",
        dati,
    )


# Mappa strato × token del Delta mediano sui canali
def _grafico_mappa(mediana_per_token, n_strati, token):
    z = np.log10(np.stack([mediana_per_token[i] for i in range(n_strati)]).clip(min=1e-30))
    L = z.shape[1]
    etichette = [t.replace("\n", "↵") for t in token[:L]] if token else [str(k) for k in range(L)]
    tacche = list(range(math.ceil(z.min()), math.floor(z.max()) + 1))
    if len(tacche) >= 2:
        etichette_tacche = [f"10<sup>{k}</sup>" for k in tacche]
    else:
        tacche = list(np.linspace(z.min(), z.max(), 4))
        etichette_tacche = [fmt_sci(10 ** v) for v in tacche]
    fig = g.nuova_figura("Token (posizione nel testo)", "Strato", altezza=max(380, 14 * n_strati + 120),
                         hover_unificato=False)
    fig.add_trace(go.Heatmap(
        z=z, x=list(range(L)), y=list(range(n_strati)), colorscale=g.SCALA_BLU,
        customdata=np.array([etichette] * n_strati),
        hovertemplate="strato %{y}, token %{x}: «%{customdata}»<br>Δ mediano = 10<sup>%{z:.2f}</sup><extra></extra>",
        colorbar=dict(title=dict(text="Δ mediano", side="right"), thickness=12, outlinewidth=0,
                      tickvals=tacche, ticktext=etichette_tacche),
    ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickmode="linear", dtick=1 if n_strati <= 24 else 4)
    dati = pd.DataFrame(10 ** z.T, columns=[f"strato {i}" for i in range(n_strati)])
    dati.insert(0, "token", etichette)
    dati.insert(0, "posizione", list(range(L)))
    return Grafico(
        "Mappa del passo Δ: strato × token", fig,
        "Ogni cella è il Δ mediano (sui canali) di uno strato mentre legge un token. Passa sopra una cella per vedere il token.",
        """
- **Righe = strati** (lo 0 in alto), **colonne = token** nell'ordine del testo.
- **Scuro = Δ grande**: lo strato su quel token cancella molta memoria. **Chiaro = Δ piccolo**: conserva.
- **Colonne scure verticali** = token che fanno "ripartire" molti strati insieme (spesso punti e inizi di frase).
- **Righe uniformemente chiare** = strati che conservano memoria su tutto il testo.
- **Esempio di lettura:** una colonna scura in corrispondenza di «.» indica che a fine frase molti strati
  svuotano lo stato: il modello tratta il punto come un confine.
""",
        dati,
    )


FASE = Fase(
    codice="2a",
    nome="Passo di campionamento Δ",
    breve="Delta token per token (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
)
