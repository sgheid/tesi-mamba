import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from nucleo import confronto as cf
from nucleo import grafica as g
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import FORMATI, SOGLIA_FP16_ZERO, delta_baseline, descrivi_formato, indicatore_formato, passaggio_ridotto
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

**Confronto con FP16/BF16.** Con un confronto attivo lo stesso testo passa anche in una copia del modello con i
pesi in FP16 o BF16, e i due $\Delta$ vengono confrontati elemento per elemento, token per token e canale per
canale. In transformers $\Delta$ viene calcolato **nel formato dei pesi**: prodotto, somma del bias e softplus sono
in FP16/BF16, e l'ingresso $u_t$ ha già attraversato tutti gli strati precedenti in bassa precisione, quindi
l'errore si accumula con la profondità. In FP16 c'è un effetto in più: ogni $\Delta$ sotto
$2^{-25} \approx 3\cdot10^{-8}$ (metà del più piccolo subnormale) diventa **esattamente 0**, e con $\Delta = 0$ il
token non entra nello stato e $\bar a = 1$. Per esempio, un $\Delta = 0.0123$ in FP32 cade in FP16 tra numeri
distanti $2^{-17} \approx 7.6\cdot10^{-6}$: il solo arrotondamento finale sbaglia fino a $3\cdot10^{-4}$ in
relativo, in BF16 (spaziatura $2^{-14}$) fino a $2.5\cdot10^{-3}$; un $\Delta$ FP32 di $10^{-8}$ in FP16 vale 0.
Il riferimento del confronto è il $\Delta$ del modello FP32 con le stesse operazioni di transformers.
"""

COLONNE = [Colonna("strato", "Strato", "", "int"),
           Colonna("d_min", "Delta minimo", "(ignora il token)", "sci"),
           Colonna("d_mediana", "Delta tipico", "mediana", "sci"),
           Colonna("d_max", "Delta massimo", "(si resetta)", "sci"),
           Colonna("decadi", "Selettività", "decadi max/min", "dec")]

LEGENDA = [("Delta", "Periodo di campionamento del canale a quel token: abar = exp(Delta * a)."),
           ("Delta minimo", "Il token che lo strato 'salta' di più: lo stato resta quasi invariato."),
           ("Delta tipico", "La mediana su tutti i token e i canali. Inizializzazione: tra 0.001 e 0.1."),
           ("Delta massimo", "Il token su cui lo strato 'riparte da zero': lo stato si svuota."),
           ("Selettività", "log10(Delta max / Delta min): di quanti ordini di grandezza varia il passo.")]


def _statistiche(i, d):
    d_min, d_max = d.min().item(), d.max().item()
    return dict(strato=i, d_min=d_min, d_mediana=d.median().item(), d_max=d_max,
                decadi=math.log10(d_max / d_min) if d_min > 0 else float("inf"),
                valori=d.numel())


# Delta di ogni strato, catturato dalle sonde mentre il modello FP32 legge il testo
def _delta(ctx):
    n = len(ctx.model.backbone.layers)
    dati, mediana_per_token = {}, {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            d = delta_fp32(mixer, ingresso_delta(mixer, uscita))
            dati[i] = _statistiche(i, d)
            mediana_per_token[i] = d.median(dim=1).values.cpu().numpy()
            ctx.avanzamento((i + 1) / n, f"Strato {i + 1} di {n}")
        return hook

    esegui_con_sonde(ctx.model, ctx.ids, crea_hook)
    return [dati[i] for i in range(n)], mediana_per_token


def _sintesi(righe):
    lento = min(righe, key=lambda r: r["d_mediana"])
    rapido = max(righe, key=lambda r: r["d_mediana"])
    return lento, rapido, min(valori(righe, "d_min")), max(valori(righe, "d_max"))


def esegui(ctx):
    nome = cf.formato_confronto(ctx)
    return _solo_fp32(ctx) if nome is None else _con_confronto(ctx, nome)


# Solo FP32: il calcolo della vecchia Fase 2a, senza seconda copia
def _solo_fp32(ctx):
    righe, mediana_per_token = _delta(ctx)
    lento, rapido, d_min, d_max = _sintesi(righe)
    canali = righe[0]["valori"] // ctx.n_token

    indicatori = [
        Indicatore("Valori di Δ per strato", fmt_int(righe[0]["valori"]),
                   f"{ctx.n_token} token × {canali} canali"),
        Indicatore("Passo tipico più piccolo", fmt_sci(lento["d_mediana"]),
                   f"strato {lento['strato']}: memoria più lunga"),
        Indicatore("Passo tipico più grande", fmt_sci(rapido["d_mediana"]),
                   f"strato {rapido['strato']}: si resetta di più"),
        Indicatore("Δ minimo / massimo del modello", f"{fmt_sci(d_min)} / {fmt_sci(d_max)}"),
    ]
    return Risultato(
        codice="2",
        titolo="Fase 2 - Passo di campionamento Delta",
        descrizione="Calcolato dal modello token per token: dipende dal testo letto.",
        colonne=COLONNE,
        righe=righe,
        indicatori=indicatori,
        legenda=LEGENDA,
        grafici=[_grafico_fascia(righe), _grafico_mappa(mediana_per_token, len(righe), ctx.token)],
        evidenzia={lento["strato"]: "passo tipico più piccolo", rapido["strato"]: "passo tipico più grande"},
    )


# FP32 e formato ridotto affiancati, più gli errori della vecchia Fase 3b
def _con_confronto(ctx, nome):
    righe, mediana_per_token = _delta(cf.tratto(ctx, 0.0, 0.2))
    passaggio = passaggio_ridotto(ctx, nome, cf.tratto(ctx, 0.2, 0.7).avanzamento)
    righe_r = [_statistiche(i, s["delta"].float()) for i, s in enumerate(passaggio["strati"])]
    errori = _errori(cf.tratto(ctx, 0.7, 1.0), nome, passaggio)
    lento, rapido, d_min, d_max = _sintesi(righe)
    lento_r, rapido_r, d_min_r, d_max_r = _sintesi(righe_r)
    canali = righe[0]["valori"] // ctx.n_token
    min_max = lambda t: f"{fmt_sci(t[0])} / {fmt_sci(t[1])}"

    indicatori = [
        errori.indicatori[0],
        Indicatore("Valori di Δ per strato", fmt_int(righe[0]["valori"]),
                   f"{ctx.n_token} token × {canali} canali"),
        cf.indicatore("Passo tipico più piccolo", lento["d_mediana"], lento_r["d_mediana"], fmt_sci, nome,
                      f"strato {lento['strato']} → {lento_r['strato']}"),
        cf.indicatore("Passo tipico più grande", rapido["d_mediana"], rapido_r["d_mediana"], fmt_sci, nome,
                      f"strato {rapido['strato']} → {rapido_r['strato']}"),
        cf.indicatore("Δ minimo / massimo del modello", (d_min, d_max), (d_min_r, d_max_r), min_max, nome,
                      "un minimo 0 vuol dire Δ azzerati dal formato" if d_min_r == 0 else ""),
    ] + errori.indicatori[1:]

    return Risultato(
        codice="2",
        titolo=f"Fase 2 - Passo di campionamento Delta, FP32 contro {nome}",
        descrizione="Calcolato dal modello token per token: dipende dal testo letto. " + errori.descrizione,
        colonne=cf.colonne_affiancate(COLONNE, nome) + errori.colonne[1:],
        righe=cf.righe_unite(righe, righe_r, errori.righe),
        indicatori=indicatori,
        legenda=LEGENDA + errori.legenda
                + [(f"FP32 e {nome}", f"Le grandezze della Fase 2 compaiono due volte: Delta del modello FP32 e "
                                      f"Delta della copia con i pesi in {nome}, sullo stesso testo.")],
        grafici=[_grafico_fascia(righe, righe_r, nome), _grafico_mappa(mediana_per_token, len(righe), ctx.token)]
                + errori.grafici,
        evidenzia=cf.evidenze_unite({lento["strato"]: "passo tipico più piccolo",
                                     rapido["strato"]: "passo tipico più grande"}, errori.evidenzia),
        note_formato="FP32 e " + errori.note_formato,
    )


def _positivi(v):
    return [x if x > 0 else None for x in v]


def _grafico_fascia(righe, righe_r=None, nome=None):
    x = valori(righe, "strato")
    d_min, d_med, d_max = (valori(righe, k) for k in ("d_min", "d_mediana", "d_max"))
    suffisso = " FP32" if righe_r else ""
    fig = g.nuova_figura("Strato", "Δ  (scala logaritmica)", log_y=True)
    g.zona(fig, "y", 0.001, 0.1, "inizializzazione: Δ da 0.001 a 0.1")
    g.fascia_min_mediana_max(fig, x, d_min, d_med, d_max,
                             tuple(f"{s}{suffisso}" for s in ("Δ massimo", "Δ tipico (mediana)", "Δ minimo")))
    dati = pd.DataFrame({"strato": x, "Delta minimo": d_min, "Delta mediana": d_med, "Delta massimo": d_max})
    tutti_min, tutti_max = d_min, d_max
    if righe_r:
        r_min, r_med, r_max = (valori(righe_r, k) for k in ("d_min", "d_mediana", "d_max"))
        g.fascia_min_mediana_max(fig, x, _positivi(r_min), _positivi(r_med), _positivi(r_max),
                                 (f"Δ massimo {nome}", f"Δ tipico {nome}", f"Δ minimo {nome}"), colore=g.ARANCIO)
        dati = dati.rename(columns=lambda c: c if c == "strato" else f"{c} FP32")
        dati[f"Delta minimo {nome}"], dati[f"Delta mediana {nome}"], dati[f"Delta massimo {nome}"] = r_min, r_med, r_max
        tutti_min = d_min + [v for v in r_min if v > 0]
        tutti_max = d_max + [v for v in r_max if math.isfinite(v)]
    g.asse_strati(fig, len(righe))
    g.scala_log(fig, "y", min(0.001, min(tutti_min)), max(0.1, max(tutti_max)))
    come_leggere = """
- **Asse y logaritmico**, una tacca per ogni fattore 10.
- **La fascia grigia** è l'intervallo in cui Mamba inizializza $\\Delta$ prima dell'addestramento.
- **Linea in basso = strato con memoria lunga**: a ogni token passa poco tempo, i modi si consumano lentamente.
- **Fascia molto alta = strato molto selettivo**: alcuni token lo fanno ripartire da zero, altri vengono saltati.
- **Esempio di lettura:** se uno strato ha mediana $10^{-3}$ e massimo $10^{1}$, di solito conserva lo stato
  ma su qualche token (spesso punteggiatura o inizio frase) lo azzera.
"""
    if righe_r:
        come_leggere += f"""- **Blu = FP32, arancione = {nome}.** La fascia e la linea blu sono il Δ del modello FP32; le linee arancioni
  lo stesso testo nella copia in {nome}: tratteggiata il massimo, punteggiata il minimo, continua la mediana.
  Se la linea punteggiata arancione **manca** in uno strato, lì il minimo vale 0: il formato ha azzerato qualche Δ
  (solo FP16, vedi "Δ diventati 0" più sotto). Le differenze sulla mediana sono di qualche millesimo e su questa scala
  non si vedono: si leggono nel grafico dell'errore in funzione della profondità.
"""
    return Grafico(
        "Passo Δ per strato", fig,
        "Per ogni strato, la fascia va dal Δ più piccolo al più grande prodotto sul testo; la linea è il Δ tipico."
        + (f" In arancione lo stesso testo nella copia in {nome}." if righe_r else ""),
        come_leggere,
        dati,
    )


# Mappa strato × token del Delta mediano sui canali (solo FP32)
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
        "Ogni cella è il Δ mediano (sui canali) di uno strato del modello FP32 mentre legge un token. "
        "Passa sopra una cella per vedere il token.",
        """
- **Righe = strati** (lo 0 in alto), **colonne = token** nell'ordine del testo.
- **Chiaro = Δ grande**: lo strato su quel token cancella molta memoria. **Scuro = Δ piccolo**: conserva.
- **Colonne chiare verticali** = token che fanno "ripartire" molti strati insieme (spesso punti e inizi di frase).
- **Righe uniformemente scure** = strati che conservano memoria su tutto il testo.
- **Esempio di lettura:** una colonna chiara in corrispondenza di «.» indica che a fine frase molti strati
  svuotano lo stato: il modello tratta il punto come un confine.
- La mappa è sempre quella del modello FP32: le differenze del formato ridotto sono troppo piccole per vedersi a colori.
""",
        dati,
    )


# Errori del Delta a bassa precisione: il calcolo della vecchia Fase 3b, sul passaggio ridotto condiviso

# Quantile esatto con kthvalue (torch.quantile ha un limite di 16M elementi)
def _quantile(x, q):
    if x.numel() == 0:
        return float("nan")
    return x.kthvalue(max(1, math.ceil(q * x.numel()))).values.item()


def _errori(ctx, nome, passaggio):
    tipo = FORMATI[nome]
    n = len(ctx.model.backbone.layers)
    base = delta_baseline(ctx, ctx.avanzamento)
    dati = {}

    for i, strato in enumerate(passaggio["strati"]):
        d_r = strato["delta"].to(ctx.dispositivo)
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
                       non_finiti_ingresso=strato["non_finiti_ingresso"],
                       valori=d_r.numel())

    esito = passaggio["esito"]
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
        codice="2",
        titolo=f"Fase 2 - Passo Delta con i pesi in {nome}",
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
                 ("Delta minimo FP32", "Il Delta più piccolo della baseline: sotto 2^-25 FP16 lo azzera. Baseline = "
                                       "Delta del modello FP32 con le operazioni di transformers."),
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
  ingresso e i suoi 16 poli valgono esattamente $\\bar a = 1$ (vedi la Fase 3).
- **Blu scuro** = Δ che già in FP32 erano sotto $2^{-25}$: li azzera il solo formato. **Blu chiaro** = Δ che in FP32
  erano sopra la soglia e ci sono finiti per l'errore accumulato.
- Succede solo in FP16: BF16 ha lo stesso intervallo di esponenti di FP32 e non azzera valori così piccoli.
- **Esempio di lettura:** una barra a 300 in uno strato da 1536 canali × 211 token vuol dire circa un valore su mille
  che smette di far entrare il token nello stato.
""",
        dati,
    )


FASE = Fase(
    codice="2",
    nome="Passo di campionamento Δ",
    breve="Delta token per token, anche nella copia FP16/BF16 (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[cf.PARAMETRO],
)
