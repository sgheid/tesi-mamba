import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from nucleo import grafica as g
from nucleo.precisione import da_cache

# Posizioni per blocco nel calcolo dei logit: 64 × 50 280 ≈ 3,2 milioni di valori per tensore FP64 (25 MB)
BLOCCO_CALCOLO = 64
# Posizioni per riga della tabella e finestra della media mobile nei grafici
BLOCCO_TABELLA = 64
FINESTRA = 32
# Istogramma della KL: log10 da -16 a 2 a quarti di decade
KL_MIN, KL_MAX, KL_PASSO = -16.0, 2.0, 0.25


# Stato finale del modello FP32 condiviso (uscita di norm_f, token × d) su CPU: una volta per modello e testo
def stato_finale_fp32(ctx):
    def calcola():
        with torch.no_grad():
            return ctx.model.backbone(ctx.ids, use_cache=False).last_hidden_state[0].cpu()
    return da_cache(ctx, "uscita FP32", calcola)


# Log-probabilità in FP64 di un blocco: logit nel formato del peso, come lm_head, poi convertiti
def _log_prob(h, W):
    logit = F.linear(h.to(W.device, W.dtype), W).float().double()
    return F.log_softmax(logit, dim=-1), torch.isfinite(logit).all(dim=-1)


# Metriche per posizione t (predizione di ids[t+1]) a blocchi di posizioni: in memoria non c'è mai un tensore
# token × vocabolario intero. Senza h_r solo la NLL del riferimento. Posizioni con logit non finiti: NaN, contate a parte
def metriche(h32, W32, ids, h_r=None, W_r=None, avanzamento=lambda frazione, messaggio="": None):
    n = h32.shape[0] - 1
    bersagli = ids[0, 1:].to(W32.device)
    m = dict(nll=np.full(n, np.nan), finito=np.zeros(n, dtype=bool))
    if h_r is not None:
        m.update(nll_r=np.full(n, np.nan), kl=np.full(n, np.nan), top1=np.zeros(n, dtype=bool))
    with torch.no_grad():
        for s in range(0, n, BLOCCO_CALCOLO):
            e = min(s + BLOCCO_CALCOLO, n)
            t = bersagli[s:e, None]
            lp, finito = _log_prob(h32[s:e], W32)
            nll = -lp.gather(1, t)[:, 0]
            if h_r is not None:
                lr, finito_r = _log_prob(h_r[s:e], W_r.to(W32.device))
                finito &= finito_r
                nll_r = -lr.gather(1, t)[:, 0]
                # KL(P32 || Pr) in FP64; i valori negativi sono solo arrotondamento e diventano 0
                kl = (lp.exp() * (lp - lr)).sum(dim=-1).clamp_min(0)
                m["nll_r"][s:e] = torch.where(finito, nll_r, torch.nan).cpu().numpy()
                m["kl"][s:e] = torch.where(finito, kl, torch.nan).cpu().numpy()
                m["top1"][s:e] = (lp.argmax(dim=-1) == lr.argmax(dim=-1)).cpu().numpy()
            m["nll"][s:e] = torch.where(finito, nll, torch.nan).cpu().numpy()
            m["finito"][s:e] = finito.cpu().numpy()
            avanzamento(e / n, f"logit e KL, posizione {e} di {n}")
    return m


def perplexity(nll):
    return math.exp(np.nanmean(nll)) if np.isfinite(nll).any() else float("nan")


# Sintesi del confronto sulle posizioni finite
def sintesi(m):
    f = m["finito"]
    s = dict(posizioni=len(f), non_finite=int((~f).sum()), ppl=perplexity(m["nll"]), nll=np.nanmean(m["nll"]))
    if "kl" in m and f.any():
        kl = m["kl"][f]
        s.update(ppl_r=perplexity(m["nll_r"]), nll_r=np.nanmean(m["nll_r"]),
                 kl_media=float(kl.mean()), kl_mediana=float(np.median(kl)), kl_p99=float(np.percentile(kl, 99)),
                 kl_max=float(kl.max()), pos_kl_max=int(np.flatnonzero(f)[kl.argmax()]),
                 top1=float(m["top1"][f].mean()), dnll=float((m["nll_r"][f] - m["nll"][f]).mean()),
                 identiche=bool(np.array_equal(m["nll"], m["nll_r"], equal_nan=True)))
    return s


# Una riga per blocco di posizioni; la chiave "strato" contiene l'etichetta, come le righe per strato delle altre fasi
def righe_per_blocchi(m):
    righe = []
    for s in range(0, len(m["nll"]), BLOCCO_TABELLA):
        e = min(s + BLOCCO_TABELLA, len(m["nll"]))
        f = m["finito"][s:e]
        r = dict(strato=f"{s}–{e - 1}", da=s, a=e - 1, nll=_media(m["nll"][s:e]), non_finite=int((~f).sum()))
        if "kl" in m:
            r.update(nll_r=_media(m["nll_r"][s:e]), dnll=_media(m["nll_r"][s:e] - m["nll"][s:e]),
                     kl=_media(m["kl"][s:e]), top1=float(m["top1"][s:e][f].mean()) if f.any() else float("nan"))
        righe.append(r)
    return righe


def _media(v):
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def media_mobile(v, finestra=FINESTRA):
    return pd.Series(v).rolling(finestra, min_periods=1).mean().to_numpy()


# Valori per posizione (punti tenui) e media mobile (linea); serie = [(nome, valori, colore), ...]
def figura_posizioni(asse_y, serie, log_y=False):
    fig = g.nuova_figura("Posizione nel testo  (token)", asse_y, log_y=log_y)
    for nome, y, colore in serie:
        x = np.arange(len(y))
        punti = np.where(y > 0, y, np.nan) if log_y else y
        fig.add_trace(go.Scatter(x=x, y=punti, mode="markers", name=f"{nome}, per posizione", opacity=0.3,
                                 marker=dict(size=3, color=colore), hoverinfo="skip", showlegend=False))
        mm = media_mobile(y)
        fig.add_trace(go.Scatter(x=x, y=np.where(mm > 0, mm, np.nan) if log_y else mm, mode="lines",
                                 name=f"{nome}, media mobile su {FINESTRA}", line=dict(color=colore, width=2.2),
                                 hovertemplate=f"{nome}: %{{y:.3g}}<extra></extra>"))
    if log_y:
        positivi = np.concatenate([y[np.isfinite(y) & (y > 0)] for _, y, _ in serie])
        if positivi.size:
            g.scala_log(fig, "y", positivi.min(), positivi.max())
    return fig


def dati_posizioni(m, nomi):
    dati = pd.DataFrame({"posizione": np.arange(len(m["nll"])), "logit finiti": m["finito"]})
    for chiave, nome in nomi.items():
        dati[nome] = m[chiave]
        dati[f"{nome}, media mobile su {FINESTRA}"] = media_mobile(m[chiave])
    return dati


# Istogramma di log10(KL) sulle posizioni con KL > 0; serie = [(nome, kl, colore), ...]
def figura_istogramma_kl(serie):
    bordi = np.arange(KL_MIN, KL_MAX + KL_PASSO / 2, KL_PASSO)
    centri = (bordi[:-1] + bordi[1:]) / 2
    fig = g.nuova_figura("KL(P_FP32 ‖ P_ridotto) per posizione  (decadi)", "Posizioni", hover_unificato=False)
    dati = pd.DataFrame({"log10(KL) da": bordi[:-1], "log10(KL) a": bordi[1:]})
    for k, (nome, kl, colore) in enumerate(serie):
        v = kl[np.isfinite(kl) & (kl > 0)]
        conteggi = np.histogram(np.clip(np.log10(v), KL_MIN, KL_MAX - 1e-9), bins=bordi)[0]
        fig.add_trace(go.Bar(x=centri, y=conteggi, width=KL_PASSO * (0.9 if k == 0 else 0.45), name=nome,
                             marker=dict(color=colore, line_width=0),
                             customdata=np.stack([10 ** bordi[:-1], 10 ** bordi[1:]], axis=1),
                             hovertemplate=f"{nome}: KL tra %{{customdata[0]:.1e}} e %{{customdata[1]:.1e}}<br>"
                                           "%{y} posizioni<extra></extra>"))
        dati[f"posizioni {nome}"] = conteggi
    fig.update_layout(barmode="overlay")
    g.asse_decadi(fig, "x", KL_MIN, KL_MAX)
    return fig, dati
