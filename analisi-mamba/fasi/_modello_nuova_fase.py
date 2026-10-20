# Modello per una nuova fase (ignorato perché il nome inizia con "_").
# Copialo in fasi/fase3a_<nome>.py, cambia i campi di FASE, scrivi il calcolo in esegui()
# e premi "Ricarica le fasi" nell'app: la fase compare nel menu.
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_sci
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Parametro, Risultato, valori
from nucleo.sonde import delta_fp32, esegui_con_sonde, ingresso_delta

SPIEGAZIONE = r"""
**Cosa misura.** ...

**Perché conta.** ...

**Esempio.** ... (un caso numerico concreto, con i numeri)
"""

FORMATI = {"FP16": torch.float16, "BF16": torch.bfloat16}


# Esempio di calcolo: quanto cambia A_log arrotondandolo al formato scelto
def esegui(ctx):
    formato = FORMATI[ctx.parametri.get("formato", "FP16")]
    strati = ctx.model.backbone.layers
    righe = []
    with torch.no_grad():
        for i, layer in enumerate(strati):
            originale = layer.mixer.A_log.float()
            errore = (originale.to(formato).float() - originale).abs()
            righe.append(dict(strato=i, errore_max=errore.max().item(), errore_mediano=errore.median().item()))
            ctx.avanzamento((i + 1) / len(strati), f"Strato {i + 1} di {len(strati)}")

    peggiore = max(righe, key=lambda r: r["errore_max"])
    fig = g.nuova_figura("Strato", "Errore assoluto su A_log", log_y=True)
    fig.add_trace(go.Scatter(x=valori(righe, "strato"), y=valori(righe, "errore_max"),
                             mode="lines+markers", name="errore massimo", line=dict(color=g.BLU, width=2)))
    g.asse_strati(fig, len(righe))

    return Risultato(
        codice="9",
        titolo="Fase 9 - Titolo della fase",
        descrizione="Una riga su cosa contiene la tabella.",
        colonne=[Colonna("strato", "Strato", "", "int"),
                 Colonna("errore_max", "Errore massimo", "|A_log arrotondato - A_log|", "sci"),
                 Colonna("errore_mediano", "Errore tipico", "mediana", "sci")],
        righe=righe,
        indicatori=[Indicatore("Strato con l'errore massimo", str(peggiore["strato"]),
                               fmt_sci(peggiore["errore_max"]))],
        legenda=[("Errore massimo", "Spiegazione in una riga."),
                 ("Errore tipico", "Spiegazione in una riga.")],
        grafici=[Grafico("Titolo del grafico", fig, "Una frase sotto il titolo.",
                         "- punti per la lettura\n- **Esempio di lettura:** ...",
                         pd.DataFrame(righe))],
        evidenzia={peggiore["strato"]: "errore massimo"},
        note_formato=f"pesi in {ctx.parametri.get('formato', 'FP16')}, calcolo in FP32",
    )


FASE = Fase(
    codice="9",
    nome="Nome breve",
    breve="una riga per il menu",
    spiegazione=SPIEGAZIONE,
    usa_testo=False,
    esegui=esegui,
    parametri=[Parametro("formato", "Formato dei pesi", ["FP16", "BF16"], "FP16",
                         "Formato in cui arrotondare i pesi prima del calcolo.")],
)
