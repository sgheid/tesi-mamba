import numpy as np
import torch

from nucleo import confronto as cf
from nucleo import grafica as g
from nucleo import uscita as u
from nucleo.formato import fmt_int, fmt_perc, fmt_sci
from nucleo.precisione import FORMATI, descrivi_formato, indicatore_formato, passaggio_ridotto
from nucleo.risultato import Colonna, Fase, Grafico, Indicatore, Risultato

SPIEGAZIONE = r"""
**Cosa misura.** Quanto bene il modello predice il token successivo del testo, e quanto cambia questa predizione
quando i pesi sono in FP16 o BF16. A ogni posizione $t$ il modello dà una distribuzione $P(\cdot \mid x_{\le t})$
sul vocabolario (50 280 token): la **NLL** è $-\log P(x_{t+1})$, la **perplexity** è $e^{\text{NLL media}}$.
Con un confronto si misura anche la **KL** $\mathrm{KL}(P_{FP32}\,\|\,P_{ridotto}) = \sum_v P_{FP32}(v)\,
\log\frac{P_{FP32}(v)}{P_{ridotto}(v)}$, cioè quanto la distribuzione del formato ridotto si allontana da quella
FP32, e l'**accordo top-1**, cioè quante volte i due modelli scelgono lo stesso token più probabile.

**Perché conta.** Le Fasi 1–3 mostrano che la bassa precisione sposta $a$, $\Delta$ e i poli $\bar a$; questa fase
dice se quegli spostamenti arrivano fino all'uscita. L'andamento lungo il testo serve a vedere se l'errore
**cresce con il contesto**: se i modi a memoria lunga accumulano l'errore token dopo token, la KL alla fine del
testo è più alta che all'inizio.

**Esempio.** Se a una posizione FP32 assegna 0,40 al token giusto e FP16 gli assegna 0,38, la NLL passa da
$-\ln 0{,}40 = 0{,}916$ a $-\ln 0{,}38 = 0{,}968$: $\Delta$NLL $= +0{,}051$. Una KL media di $10^{-4}$ nat
vuol dire distribuzioni quasi indistinguibili; una KL di $0{,}1$ nat in una posizione vuol dire che lì la scelta
del token può cambiare davvero.

Tutto è calcolato a blocchi di 64 posizioni: logit nel formato del modello, come fa `lm_head`, poi convertiti in
FP64 per `log_softmax`, NLL e KL. Nel confronto la ricorrenza resta in FP32, come nei runtime reali; la
Fase 5 studia il caso in cui va anch'essa in bassa precisione.
"""

COLONNE = [Colonna("strato", "Posizioni", "token che predicono il successivo", "testo"),
           Colonna("nll", "NLL media", "nat per token", "sci"),
           Colonna("ppl", "Perplexity", "exp(NLL media) del blocco", "sci")]

COLONNE_CONFRONTO = [Colonna("dnll", "ΔNLL media", "ridotto − FP32, nat", "sci"),
                     Colonna("kl", "KL media", "KL(P_FP32 ‖ P_ridotto), nat", "sci"),
                     Colonna("top1", "Accordo top-1", "% stesso token più probabile", "perc"),
                     Colonna("non_finite", "Logit non finiti", "posizioni escluse", "int")]

LEGENDA = [("Posizioni", "Blocco di 64 posizioni: la posizione t predice il token t+1 del testo."),
           ("NLL media", "Media di −log P(token vero) sulle posizioni del blocco, in nat. Più bassa = predizione migliore."),
           ("Perplexity", "exp(NLL media): il numero di token tra cui il modello è 'indeciso' in media. "
                          "Quella del testo intero è negli indicatori.")]

LEGENDA_CONFRONTO = [("ΔNLL media", "NLL del formato ridotto meno NLL FP32, mediata sul blocco: positiva = il "
                                    "formato ridotto predice peggio il token vero."),
                     ("KL media", "KL(P_FP32 ‖ P_ridotto) in nat, calcolata in FP64 da log_softmax sull'intero "
                                  "vocabolario, mediata sul blocco. 0 = distribuzioni identiche."),
                     ("Accordo top-1", "Quota delle posizioni del blocco in cui FP32 e formato ridotto hanno lo stesso "
                                       "token più probabile (argmax dei logit)."),
                     ("Logit non finiti", "Posizioni con almeno un logit inf/NaN: contate ed escluse dalle medie.")]


def esegui(ctx):
    nome = cf.formato_confronto(ctx)
    return _solo_fp32(ctx) if nome is None else _con_confronto(ctx, nome)


def _fmt_ppl(x):
    return f"{x:.4f}"


def _token(ctx, t):
    return repr(ctx.token[t]) if 0 <= t < len(ctx.token) else "?"


def _diverse(k):
    return "nessuna posizione con un token più probabile diverso" if k == 0 else (
        f"{fmt_int(k)} {'posizione' if k == 1 else 'posizioni'} con un token più probabile diverso")


def _note(ctx, testo):
    return f"{testo} · dispositivo: {ctx.nome_dispositivo}"


def _solo_fp32(ctx):
    h32 = u.stato_finale_fp32(cf.tratto(ctx, 0.0, 0.5))
    ctx.avanzamento(0.5, "stato finale FP32 pronto")
    m = u.metriche(h32, ctx.model.lm_head.weight, ctx.ids, avanzamento=cf.tratto(ctx, 0.5, 1.0).avanzamento)
    s = u.sintesi(m)
    righe = [dict(r, ppl=np.exp(r["nll"])) for r in u.righe_per_blocchi(m)]
    peggiore = max(righe, key=lambda r: r["nll"])

    indicatori = [
        Indicatore("Perplexity", _fmt_ppl(s["ppl"]), f"exp(NLL media) su {fmt_int(s['posizioni'])} predizioni"),
        Indicatore("NLL media", f"{s['nll']:.4f} nat", "−log P(token vero), media sul testo"),
        Indicatore("Blocco più difficile", peggiore["strato"], f"NLL media {peggiore['nll']:.3f} nat"),
        Indicatore("Logit non finiti", fmt_int(s["non_finite"]),
                   "nessuno" if s["non_finite"] == 0 else "posizioni escluse dalle medie",
                   "ok" if s["non_finite"] == 0 else "attenzione"),
        Indicatore("Dispositivo", ctx.nome_dispositivo, "logit in FP32, log_softmax e NLL in FP64"),
    ]
    return Risultato(
        codice="4",
        titolo="Fase 4 - Uscita del modello: NLL e perplexity",
        descrizione="Predizione del token successivo sul testo, a blocchi di 64 posizioni; NLL in FP64 dai logit FP32.",
        colonne=COLONNE,
        righe=righe,
        indicatori=indicatori,
        legenda=LEGENDA,
        grafici=[_grafico_nll(m)],
        evidenzia={peggiore["strato"]: "NLL media più alta"},
        note_formato=_note(ctx, "FP32, NLL in FP64"),
    )


def _con_confronto(ctx, nome):
    h32 = u.stato_finale_fp32(cf.tratto(ctx, 0.0, 0.2))
    passaggio = passaggio_ridotto(ctx, nome, cf.tratto(ctx, 0.2, 0.7).avanzamento)
    m = u.metriche(h32, ctx.model.lm_head.weight, ctx.ids, passaggio["uscita"]["stato_finale"],
                   passaggio["uscita"]["lm_head"], cf.tratto(ctx, 0.7, 1.0).avanzamento)
    s = u.sintesi(m)
    righe = [dict(r, ppl=np.exp(r["nll"]), ppl_r=np.exp(r["nll_r"])) for r in u.righe_per_blocchi(m)]
    quarto = max(1, len(m["kl"]) // 4)
    kl_inizio, kl_fine = np.nanmean(m["kl"][:quarto]), np.nanmean(m["kl"][-quarto:])
    controllo = FORMATI[nome] == torch.float32

    indicatori = [
        indicatore_formato(nome, ctx.dispositivo, passaggio["esito"]),
        cf.indicatore("Perplexity", s["ppl"], s["ppl_r"], _fmt_ppl, nome,
                      f"su {fmt_int(s['posizioni'])} predizioni"),
        Indicatore("KL media", fmt_sci(s["kl_media"]),
                   f"nat per posizione · mediana {fmt_sci(s['kl_mediana'])} · P99 {fmt_sci(s['kl_p99'])}"),
        Indicatore("KL massima", fmt_sci(s["kl_max"]),
                   f"posizione {s['pos_kl_max']}, che predice {_token(ctx, s['pos_kl_max'] + 1)}"),
        Indicatore("Accordo top-1", fmt_perc(s["top1"]),
                   _diverse(round((1 - s['top1']) * (s['posizioni'] - s['non_finite']))), "ok" if s["top1"] == 1 else None),
        Indicatore("ΔNLL media", f"{s['dnll']:+.2e} nat",
                   "il formato ridotto predice peggio il token vero" if s["dnll"] > 0
                   else ("nessuna differenza" if s["dnll"] == 0 else "il formato ridotto dà in media più probabilità al token vero")),
        Indicatore("KL: primo → ultimo quarto del testo", f"{fmt_sci(kl_inizio)} → {fmt_sci(kl_fine)}",
                   cf.differenza(kl_inizio, kl_fine) or "un aumento indica errore che si accumula con il contesto"),
        Indicatore("Logit non finiti", fmt_int(s["non_finite"]),
                   "nessuno" if s["non_finite"] == 0 else "posizioni escluse da medie e KL",
                   "ok" if s["non_finite"] == 0 else "attenzione"),
        Indicatore("Dispositivo", ctx.nome_dispositivo, "logit nel formato dei pesi, log_softmax, NLL e KL in FP64"),
    ]
    if controllo:
        ok = s["identiche"] and s["kl_max"] == 0 and s["top1"] == 1
        indicatori.append(Indicatore("Controllo FP32", "superato" if ok else "NON superato",
                                     "due passaggi in FP32: KL nulla, NLL identiche bit a bit, top-1 al 100%",
                                     "ok" if ok else "attenzione"))

    evidenzia = {}
    if s["kl_max"] > 0:
        evidenzia[max(righe, key=lambda r: r["kl"])["strato"]] = "KL media più alta"
        peggiore = min(righe, key=lambda r: r["top1"])
        if peggiore["top1"] < 1:
            evidenzia = cf.evidenze_unite(evidenzia, {peggiore["strato"]: "accordo top-1 più basso"})

    return Risultato(
        codice="4",
        titolo=f"Fase 4 - Uscita del modello, FP32 contro {nome}",
        descrizione="Predizione del token successivo sul testo, a blocchi di 64 posizioni; logit nel formato dei "
                    "pesi, NLL e KL in FP64.",
        colonne=cf.colonne_affiancate(COLONNE, nome) + COLONNE_CONFRONTO,
        righe=righe,
        indicatori=indicatori,
        legenda=LEGENDA + LEGENDA_CONFRONTO,
        grafici=[_grafico_nll(m, nome), _grafico_kl(m, nome), _grafico_dnll(m, nome), _grafico_istogramma(m, nome)],
        evidenzia=evidenzia,
        note_formato=_note(ctx, descrizione_formato(nome, ctx.dispositivo)),
    )


def descrizione_formato(nome, dispositivo):
    return descrivi_formato(nome, dispositivo) + "; NLL e KL in FP64"


def _vuoto(fig, testo):
    fig.add_annotation(text=testo, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=14, color=g.TENUE))


def _grafico_nll(m, nome=None):
    serie = [("NLL FP32" if nome else "NLL", m["nll"], g.BLU)]
    nomi = {"nll": "NLL FP32" if nome else "NLL"}
    if nome:
        serie.append((f"NLL {nome}", m["nll_r"], g.ARANCIO))
        nomi["nll_r"] = f"NLL {nome}"
    fig = u.figura_posizioni("NLL  (nat)", serie)
    come_leggere = f"""
- **Punti tenui** = NLL di ogni singola predizione; **linea** = media mobile sulle ultime {u.FINESTRA} posizioni.
- Più in basso = il modello era più sicuro del token che è poi arrivato. Le prime posizioni hanno di solito NLL alta:
  il modello ha ancora poco contesto.
- **Esempio di lettura:** una media mobile che scende da 6 a 3 nat nei primi 200 token dice che il contesto aiuta:
  la perplexity locale passa da $e^6 \\approx 400$ a $e^3 \\approx 20$ token "plausibili".
"""
    if nome:
        come_leggere += f"""- **Blu = FP32, arancione = {nome}.** Di solito le due linee si sovrappongono: la differenza si legge meglio
  nei grafici della KL e della ΔNLL.
"""
    return Grafico("NLL lungo il testo", fig,
                   "−log P(token vero) a ogni posizione, con la media mobile." + (f" In arancione {nome}." if nome else ""),
                   come_leggere, u.dati_posizioni(m, nomi))


def _grafico_kl(m, nome):
    fig = u.figura_posizioni("KL(P_FP32 ‖ P_ridotto)  (nat, scala logaritmica)", [(f"KL {nome}", m["kl"], g.ARANCIO)],
                             log_y=True)
    if not np.nanmax(m["kl"], initial=0) > 0:
        _vuoto(fig, "KL nulla in ogni posizione (controllo FP32)")
    return Grafico(
        "KL tra FP32 e formato ridotto lungo il testo", fig,
        f"Distanza tra la distribuzione FP32 e quella {nome} a ogni posizione, con la media mobile.",
        f"""
- **Asse y logaritmico.** Ogni decade in più = distribuzioni dieci volte più lontane. Le posizioni con KL esattamente 0
  non compaiono.
- **Una media mobile che sale** verso destra vuol dire che l'errore si accumula con il contesto: è il segnale atteso
  se lo portano i modi a memoria lunga. **Una linea piatta** vuol dire errore locale, che non dipende da quanto testo
  è già passato.
- **Esempio di lettura:** media mobile a $10^{{-4}}$ nei primi 100 token e a $10^{{-3}}$ dopo 800 token: l'errore è
  cresciuto di dieci volte con il contesto. Un punto isolato a $10^{{-1}}$ è una singola predizione in cui {nome} ha
  cambiato in modo visibile le probabilità.
""",
        u.dati_posizioni(m, {"kl": f"KL FP32 vs {nome}"}))


def _grafico_dnll(m, nome):
    dnll = m["nll_r"] - m["nll"]
    fig = u.figura_posizioni("ΔNLL = NLL ridotta − NLL FP32  (nat)", [(f"ΔNLL {nome}", dnll, g.ARANCIO)])
    fig.add_hline(y=0, line=dict(color=g.ASSE, width=1))
    dati = u.dati_posizioni(m, {"nll": "NLL FP32", "nll_r": f"NLL {nome}"})
    dati["ΔNLL"], dati[f"ΔNLL, media mobile su {u.FINESTRA}"] = dnll, u.media_mobile(dnll)
    return Grafico(
        "ΔNLL lungo il testo", fig,
        f"Quanto {nome} predice peggio (sopra lo zero) o meglio (sotto) il token vero, a ogni posizione.",
        f"""
- A differenza della KL, la ΔNLL guarda solo il token vero e ha un segno: sopra lo zero {nome} gli dà meno
  probabilità di FP32.
- I singoli punti oscillano attorno a zero; conta la **media mobile**: se resta sopra lo zero, il formato peggiora
  la predizione in modo sistematico.
- **Esempio di lettura:** una media mobile a $+10^{{-3}}$ nat vuol dire perplexity più alta di circa lo 0,1%
  ($e^{{0{{,}}001}} \\approx 1{{,}}001$) in quel tratto di testo.
""",
        dati)


def _grafico_istogramma(m, nome):
    fig, dati = u.figura_istogramma_kl([(f"posizioni {nome}", m["kl"], g.ARANCIO)])
    if not np.nanmax(m["kl"], initial=0) > 0:
        _vuoto(fig, "KL nulla in ogni posizione (controllo FP32)")
    nulle = int((m["kl"][m["finito"]] == 0).sum())
    return Grafico(
        "Distribuzione della KL sulle posizioni", fig,
        f"Quante posizioni hanno KL in ogni quarto di decade ({fmt_int(nulle)} posizioni con KL esattamente 0 escluse).",
        f"""
- **Asse x in decadi**: a sinistra le posizioni in cui FP32 e {nome} danno distribuzioni quasi identiche, a destra
  quelle in cui differiscono di più.
- Una coda lunga verso destra vuol dire che l'errore medio è basso ma poche posizioni cambiano molto.
- **Esempio di lettura:** il grosso delle barre tra $10^{{-6}}$ e $10^{{-4}}$ e qualche barra a $10^{{-2}}$: quasi
  ovunque l'effetto è trascurabile, in poche posizioni (di solito token incerti, con due candidati vicini) è visibile.
""",
        dati)


FASE = Fase(
    codice="4",
    nome="Effetto sull'uscita",
    breve="NLL, perplexity, KL e accordo top-1 contro la copia FP16/BF16, lungo il testo (usa il testo)",
    spiegazione=SPIEGAZIONE,
    usa_testo=True,
    esegui=esegui,
    parametri=[cf.PARAMETRO],
)
