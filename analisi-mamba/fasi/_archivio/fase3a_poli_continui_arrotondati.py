import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import FORMATI, descrivi_formato
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Parametro, Risultato, valori

SOGLIA = 0.01  # un polo è "molto perturbato" se |a| cambia di più dell'1%
PASSO = 0.25   # binning di |a| a quarti di decade

SPIEGAZIONE = r"""
**Cosa misura.** Quanto si spostano i poli continui $a = -\exp(A_{\log})$ quando il modello viene
caricato con i pesi in FP16 o BF16. Il modello calcola $a$ in FP32, ma a partire da $A_{\log}$ già
arrotondato al formato ridotto: l'errore nasce tutto lì. Non usa il testo.

**Perché conta.** Se $A_{\log}$ si sposta di $\delta$, il polo diventa $a\,e^{\delta}$: l'errore relativo
su $a$ è circa l'errore **assoluto** su $A_{\log}$. La spaziatura dei numeri in virgola mobile cresce con
il valore, quindi l'errore cresce con $|A_{\log}|$: i poli più colpiti sono i più lenti ($|a|\ll 1$,
$A_{\log}$ molto negativo) e i più veloci ($|a|\gg 1$). I poli lenti sono proprio quelli con la memoria più
lunga, la cui costante di tempo $1/|a|$ cambia della stessa percentuale.

**Esempio.** Con $A_{\log} \approx -13.1$ ($|a| \approx 2\cdot10^{-6}$, $\tau \approx 5\cdot10^{5}$):
tra 8 e 16 FP16 ha spaziatura $2^{-7} \approx 0.0078$, quindi $A_{\log}$ si sposta al più di metà,
$0.0039$, e $a$ cambia fino a $\pm 0.4\%$. BF16 ha spaziatura $2^{-4} = 0.0625$: $A_{\log}$ si sposta
fino a $0.031$ e $a$ fino a $\pm 3\%$. Con "FP32 (controllo)" non c'è arrotondamento e gli errori sono zero.
"""


# Metà della spaziatura del formato attorno a x: il massimo errore di arrotondamento (anche nei subnormali)
def _mezza_spaziatura(x, tipo):
    fi = torch.finfo(tipo)
    esponente = torch.floor(torch.log2(x.abs().clamp_min(fi.smallest_normal)))
    return fi.eps * torch.exp2(esponente) / 2


def esegui(ctx):
    nome = ctx.parametri.get("formato", "FP16")
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
        codice="3a",
        titolo=f"Fase 3a - Poli continui con A_log arrotondato a {nome}",
        descrizione="a = -exp(A_log) con A_log arrotondato al formato dei pesi; errori calcolati in FP64.",
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
  (vedi la Fase 1) hanno $|A_{\\log}|$ più grande e quindi errori più grandi.
- **Linea rossa** = 1%.
- **Esempio di lettura:** massimo a $4\\cdot10^{-3}$ e mediana a $10^{-4}$ significa che il polo peggiore dello
  strato si sposta dello 0.4%, mentre quello tipico di un decimillesimo.
""",
        dati,
    )


FASE = Fase(
    codice="3a",
    nome="Poli continui con pesi FP16/BF16",
    breve="a = -exp(A_log arrotondato), dai pesi (non usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=False,
    esegui=esegui,
    parametri=[Parametro("formato", "Formato dei pesi", list(FORMATI), "FP16",
                         "Formato in cui vengono caricati i pesi. FP32 (controllo) deve dare errori nulli.")],
)
