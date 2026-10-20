import hashlib
import html
import math
import os
import time
from datetime import datetime

import streamlit as st

from fasi import carica_fasi
from nucleo import grafica as g
from nucleo.excel import crea_excel, nome_file, salva_excel
from nucleo.formato import FORMATI, fmt_int
from nucleo.modelli import carica_e_verifica, cartella_cache_predefinita, dispositivi_disponibili, libera, trova_modelli
from nucleo.risultato import Contesto
from nucleo.testo import MAX_TOKEN_PREDEFINITO, TESTO_ESEMPIO, prepara

CARTELLA_APP = os.path.dirname(os.path.abspath(__file__))
CARTELLA_RISULTATI = os.path.join(CARTELLA_APP, "risultati")

st.set_page_config(page_title="Analisi modi di Mamba", page_icon="〰", layout="wide",
                   initial_sidebar_state="expanded")

# Stile della pagina: palette blu notte/blu, numeri in monospazio, griglia da carta millimetrata
st.markdown("""
<style>
  :root {--notte: #0f2747; --blu: #2a78d6; --blu-chiaro: #86b6ef; --linea: #e3e8f0; --testo-2: #4b5567;
         --tenue: #8a93a3; --mono: "Cascadia Code", "JetBrains Mono", Consolas, monospace;
         --ombra: 0 1px 2px rgba(15,39,71,0.04), 0 4px 14px rgba(15,39,71,0.06);}
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  header[data-testid="stHeader"] {background: transparent;}
  h1, h2, h3, h4 {letter-spacing: -0.01em; color: var(--notte);}

  section[data-testid="stSidebar"] {width: 360px !important;
      background-color: #eef2f8;
      background-image: linear-gradient(rgba(42,120,214,0.045) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(42,120,214,0.045) 1px, transparent 1px);
      background-size: 20px 20px;}
  .marchio {display: flex; gap: 12px; align-items: center; padding: 14px 16px; margin-bottom: 4px;
            border-radius: 14px; color: #fff; box-shadow: 0 6px 18px rgba(15,39,71,0.22);
            background-image: linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px),
                              linear-gradient(135deg, #0f2747 0%, #1d4f8f 100%);
            background-size: 14px 14px, 14px 14px, auto;}
  .marchio-nome {font-size: 1.12rem; font-weight: 700; letter-spacing: 0.01em;}
  .marchio-nome span {color: var(--blu-chiaro); margin: 0 4px;}
  .marchio-sotto {font-size: 0.74rem; opacity: 0.75; letter-spacing: 0.04em;}

  [class*="st-key-passo_"] {background: #ffffff; border: 1px solid var(--linea); border-radius: 14px;
                            padding: 14px 14px 12px 14px; box-shadow: var(--ombra);}
  .passo-testa {display: flex; align-items: center; gap: 10px; margin-bottom: 2px;}
  .passo-num {display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px;
              border-radius: 50%; background: var(--notte); color: #fff; font: 600 0.78rem var(--mono);}
  .passo-tit {font-weight: 650; color: var(--notte); font-size: 1rem; flex: 1;}
  .chip {display: inline-block; font: 600 0.7rem var(--mono); padding: 2px 8px; border-radius: 999px;
         letter-spacing: 0.02em; white-space: nowrap;}
  .chip.attesa {background: #eef1f5; color: var(--tenue);}
  .chip.ok {background: rgba(12,163,12,0.12); color: #0a6e0a;}
  .chip.errore {background: rgba(208,59,59,0.12); color: #a12a2a;}
  .chip.codice {background: rgba(42,120,214,0.12); color: #1d5fae;}
  .riga-stato {display: flex; gap: 10px; align-items: flex-start; background: #f5f8fc; border: 1px solid var(--linea);
               border-radius: 10px; padding: 8px 10px; font-size: 0.84rem; color: var(--testo-2); line-height: 1.35;}
  .riga-stato b {color: var(--notte);}
  .riga-stato .punto {width: 8px; height: 8px; border-radius: 50%; background: #0ca30c; margin-top: 5px;
                      box-shadow: 0 0 0 3px rgba(12,163,12,0.15); flex: none;}
  .piede {font-size: 0.72rem; color: var(--tenue); text-align: center; margin-top: 6px; letter-spacing: 0.03em;}

  .hero {position: relative; overflow: hidden; display: flex; justify-content: space-between; align-items: center;
         gap: 20px; color: #fff; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
         box-shadow: 0 12px 30px rgba(15,39,71,0.22);
         background-image: linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.055) 1px, transparent 1px),
                           linear-gradient(125deg, #0f2747 0%, #184f95 55%, #2a78d6 100%);
         background-size: 24px 24px, 24px 24px, auto;}
  .hero .sopra {font: 600 0.72rem var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--blu-chiaro);}
  .hero .titolo {font-size: 2.05rem; font-weight: 700; line-height: 1.15; margin-top: 4px;}
  .hero .sotto {font-size: 0.98rem; opacity: 0.88; margin-top: 8px; max-width: 760px; line-height: 1.5;}
  .hero svg {flex: none;}
  @media (max-width: 900px) {.hero svg {display: none;}}

  .capacita {display: flex; flex-wrap: wrap; gap: 8px; margin: -4px 0 20px 0;}
  .capacita span {background: #fff; border: 1px solid var(--linea); border-radius: 999px; padding: 4px 12px;
                  font-size: 0.82rem; color: var(--testo-2); box-shadow: 0 1px 2px rgba(15,39,71,0.04);}
  .capacita span::before {content: "◆"; color: var(--blu); font-size: 0.6rem; margin-right: 7px; vertical-align: 2px;}
  .passi {display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin: 4px 0 20px 0;}
  .passo {position: relative; background: #fff; border: 1px solid var(--linea); border-radius: 14px;
          padding: 20px 20px 18px 20px; box-shadow: var(--ombra); overflow: hidden;}
  .passo::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
                  background: linear-gradient(90deg, var(--notte), var(--blu), var(--blu-chiaro));}
  .passo .num {font: 600 1.6rem var(--mono); color: var(--blu-chiaro); line-height: 1;}
  .passo .tit {font-weight: 650; color: var(--notte); margin: 10px 0 4px 0; font-size: 1.02rem;}
  .passo .desc {font-size: 0.9rem; color: var(--testo-2); line-height: 1.5;}

  .scheda-tecnica {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); background: #fff;
                   border: 1px solid var(--linea); border-radius: 14px; box-shadow: var(--ombra); margin-bottom: 12px;}
  .scheda-tecnica .voce {padding: 14px 18px;}
  .scheda-tecnica .voce + .voce {border-left: 1px solid var(--linea);}
  .scheda-tecnica .et {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--tenue);}
  .scheda-tecnica .val {font: 600 1.35rem var(--mono); color: var(--notte); margin-top: 4px;}

  .testa-fase {display: flex; align-items: center; gap: 12px; margin: 10px 0 2px 0;}
  .testa-fase .codice {font: 600 0.78rem var(--mono); letter-spacing: 0.1em; color: #fff; background: var(--blu);
                       padding: 4px 10px; border-radius: 8px;}
  .testa-fase .nome {font-size: 1.6rem; font-weight: 700; color: var(--notte);}
  .breve-fase {color: var(--testo-2); font-size: 0.92rem; margin-bottom: 10px;}

  .riquadri {display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin: 6px 0 18px 0;}
  .riquadro {background: #fff; border: 1px solid var(--linea); border-left: 4px solid var(--blu-chiaro);
             border-radius: 12px; padding: 13px 15px 12px 15px; box-shadow: var(--ombra);
             transition: transform 0.15s ease, box-shadow 0.15s ease;}
  .riquadro:hover {transform: translateY(-2px); box-shadow: 0 10px 24px rgba(15,39,71,0.10);}
  .riquadro.ok {border-left-color: #0ca30c;}
  .riquadro.attenzione {border-left-color: #fab219;}
  .riquadro .et {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--tenue);
                 margin-bottom: 5px;}
  .riquadro .val {font: 600 1.38rem var(--mono); color: var(--notte); line-height: 1.25;}
  .riquadro .nota {font-size: 0.78rem; color: var(--testo-2); margin-top: 4px; line-height: 1.35;}
  .stato {display: inline-block; font: 600 0.66rem var(--mono); padding: 1px 7px; border-radius: 999px;
          margin-left: 6px; vertical-align: 1px; text-transform: none; letter-spacing: 0;}
  .stato.ok {background: rgba(12,163,12,0.12); color: #0a6e0a;}
  .stato.attenzione {background: rgba(250,178,25,0.22); color: #7a5200;}

  .contesto {display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;}
  .contesto span {background: #fff; border: 1px solid var(--linea); border-radius: 8px; padding: 3px 10px;
                  font-size: 0.8rem; color: var(--testo-2);}
  .contesto span i {font-style: normal; color: var(--tenue); margin-right: 5px; font-size: 0.72rem;
                    text-transform: uppercase; letter-spacing: 0.06em;}
  .controllo {font-size: 0.88rem; margin: 2px 0;}
  .sottotitolo-grafico {color: var(--testo-2); font-size: 0.92rem; margin-top: -6px;}

  [class*="st-key-vista_"] {margin: 4px 0 6px 0;}

  .tabella-wrap {max-height: 640px; overflow: auto; background: #fff; border: 1px solid var(--linea);
                 border-radius: 14px; box-shadow: var(--ombra); margin: 4px 0 8px 0;}
  table.tabella {width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.88rem; margin: 0;}
  table.tabella th {position: sticky; top: 0; z-index: 2; background: #f1f5fb; color: var(--notte);
                    text-align: right; font-weight: 650; padding: 11px 14px 9px 14px; vertical-align: bottom;
                    border: none; border-bottom: 2px solid #d5deeb; white-space: nowrap;}
  table.tabella th small {display: block; font-weight: 400; font-size: 0.72rem; color: var(--tenue); margin-top: 2px;}
  table.tabella td {padding: 7px 14px; text-align: right; border: none; border-bottom: 1px solid #eef1f6;
                    font-family: var(--mono); font-variant-numeric: tabular-nums; color: #16233a;}
  table.tabella .centro {text-align: center;}
  table.tabella .sinistra {text-align: left; font-family: inherit;}
  table.tabella tbody tr:nth-child(even) td {background: #fafbfd;}
  table.tabella tbody tr:hover td {background: #eef4fc;}
  table.tabella tbody tr.citata td {background: rgba(250,178,25,0.11);}
  table.tabella tbody tr.citata td:first-child {box-shadow: inset 3px 0 0 #fab219;}
  table.tabella tbody tr:last-child td {border-bottom: none;}
  .badge-strato {display: inline-block; min-width: 30px; padding: 1px 6px; border-radius: 6px;
                 background: #eef2f8; color: var(--notte); font-weight: 600;}
  .nota-chip {display: inline-block; font-size: 0.74rem; background: rgba(250,178,25,0.22); color: #7a5200;
              padding: 2px 9px; border-radius: 999px; white-space: nowrap;}
  table.tabella th .aiuto {position: relative; cursor: help; border-bottom: 1px dashed var(--tenue);}
  table.tabella th .suggerimento {visibility: hidden; opacity: 0; position: absolute; top: calc(100% + 10px); right: 0;
                                  width: 260px; white-space: normal; text-align: left; font-weight: 400;
                                  font-size: 0.78rem; line-height: 1.4; color: #fff; background: var(--notte);
                                  padding: 8px 11px; border-radius: 8px; z-index: 5; transition: opacity 0.12s ease;
                                  box-shadow: 0 8px 20px rgba(15,39,71,0.25);}
  table.tabella th.a-sinistra .suggerimento {right: auto; left: 0;}
  table.tabella th .aiuto:hover .suggerimento {visibility: visible; opacity: 1;}
</style>
""", unsafe_allow_html=True)

# Stato della sessione
DISPOSITIVI = dict(dispositivi_disponibili())
PREDEFINITI = dict(cartella=cartella_cache_predefinita(), model=None, tok=None, info=None, controlli=[],
                   dispositivo=next(iter(DISPOSITIVI)), testo=None, risultati={}, messaggio=None)
for chiave, valore in PREDEFINITI.items():
    st.session_state.setdefault(chiave, valore)


def impronta_testo(testo, max_token):
    return hashlib.sha1(f"{max_token}|{testo}".encode("utf-8")).hexdigest()[:10]


def html_sicuro(testo):
    return html.escape(str(testo))


def mostra_html(codice):
    st.markdown(codice, unsafe_allow_html=True)


# Illustrazioni SVG: cerchio unitario con poli vicini a 1 e risposta di un modo smorzato
def _croce(x, y, r, colore):
    return (f"<path d='M{x - r:.1f} {y - r:.1f}L{x + r:.1f} {y + r:.1f}M{x + r:.1f} {y - r:.1f}L{x - r:.1f} {y + r:.1f}' "
            f"stroke='{colore}' stroke-width='1.8' stroke-linecap='round'/>")


def svg_marchio():
    poli = "".join(_croce(19 + 13 * a, 19, 2.6, "#ffffff") for a in (0.3, 0.75))
    return ("<svg width='38' height='38' viewBox='0 0 38 38'>"
            "<line x1='2' y1='19' x2='36' y2='19' stroke='rgba(255,255,255,0.3)' stroke-dasharray='2 2'/>"
            "<line x1='19' y1='2' x2='19' y2='36' stroke='rgba(255,255,255,0.3)' stroke-dasharray='2 2'/>"
            "<circle cx='19' cy='19' r='13' fill='rgba(255,255,255,0.08)' stroke='#86b6ef' stroke-width='1.6'/>"
            f"{poli}{_croce(31.2, 19, 2.6, '#fab219')}</svg>")


def svg_hero():
    cx, cy, r = 70, 72, 54
    poli = "".join(_croce(cx + r * a, cy + r * b, 3.2, "#ffffff")
                   for a, b in ((0.25, 0), (0.5, 0.18), (0.5, -0.18), (0.72, 0), (0.86, 0.08), (0.86, -0.08)))
    curva = " ".join(f"{'M' if k == 0 else 'L'}{150 + k * 1.1:.1f} {cy - 46 * math.exp(-k / 34) * math.cos(k / 6.5):.1f}"
                     for k in range(101))
    inviluppo = " ".join(f"{'M' if k == 0 else 'L'}{150 + k * 1.1:.1f} {cy - 46 * math.exp(-k / 34):.1f}"
                         for k in range(0, 101, 4))
    return ("<svg width='280' height='144' viewBox='0 0 280 144'>"
            f"<line x1='8' y1='{cy}' x2='132' y2='{cy}' stroke='rgba(255,255,255,0.3)' stroke-dasharray='3 3'/>"
            f"<line x1='{cx}' y1='10' x2='{cx}' y2='134' stroke='rgba(255,255,255,0.3)' stroke-dasharray='3 3'/>"
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='rgba(255,255,255,0.07)' stroke='#86b6ef' stroke-width='1.6'/>"
            f"{poli}{_croce(cx + r * 0.985, cy, 3.6, '#fab219')}"
            f"<line x1='146' y1='{cy}' x2='272' y2='{cy}' stroke='rgba(255,255,255,0.3)'/>"
            f"<path d='{inviluppo}' fill='none' stroke='rgba(134,182,239,0.6)' stroke-width='1.2' stroke-dasharray='3 3'/>"
            f"<path d='{curva}' fill='none' stroke='#ffffff' stroke-width='2' stroke-linecap='round'/>"
            "</svg>")


# Azioni
def sfoglia_cartella():
    try:
        import tkinter as tk
        from tkinter import filedialog
        radice = tk.Tk()
        radice.withdraw()
        radice.wm_attributes("-topmost", 1)
        scelta = filedialog.askdirectory(initialdir=st.session_state.cartella,
                                         title="Cartella che contiene i modelli")
        radice.destroy()
        if scelta:
            st.session_state.cartella = os.path.normpath(scelta)
    except Exception as e:
        st.session_state.messaggio = ("warning", f"Finestra di selezione non disponibile ({e}). "
                                                 "Scrivi o incolla il percorso nella casella.")


def carica_modello(modello, dispositivo):
    if st.session_state.model is not None:
        st.session_state.update(model=None, tok=None)
        libera()
    with st.spinner(f"Carico e verifico {modello.nome}..."):
        model, tok, info, controlli = carica_e_verifica(modello, dispositivo)
    st.session_state.update(model=model, tok=tok, info=info, controlli=controlli, dispositivo=dispositivo)
    t = st.session_state.testo
    if model is not None and t is not None:
        imposta_testo(t["grezzo"], t["fonte"], t["max_token"])


def imposta_testo(grezzo, fonte, max_token):
    ids, token, testo, n_originali = prepara(st.session_state.tok, grezzo, st.session_state.dispositivo, max_token)
    st.session_state.testo = dict(grezzo=grezzo, fonte=fonte, max_token=max_token, ids=ids, token=token,
                                  testo=testo, n_originali=n_originali, impronta=impronta_testo(testo, max_token))


# Barra laterale: marchio e tre schede ① modello, ② testo, ③ fase, ognuna con un chip di stato
def titolo_passo(segnaposto, numero, titolo, stato, tono):
    segnaposto.markdown(f"<div class='passo-testa'><span class='passo-num'>{numero}</span>"
                        f"<span class='passo-tit'>{titolo}</span><span class='chip {tono}'>{html_sicuro(stato)}</span></div>",
                        unsafe_allow_html=True)


def riga_stato(titolo, sotto):
    mostra_html(f"<div class='riga-stato'><span class='punto'></span>"
                f"<div><b>{html_sicuro(titolo)}</b><br>{html_sicuro(sotto)}</div></div>")


def sezione_modello():
    testa = st.empty()
    st.text_input("Cartella dei modelli", key="cartella",
                  help="Predefinita: la cache di Hugging Face, dove from_pretrained salva i modelli "
                       "scaricati. Va bene anche una cartella con config.json e i pesi.")
    st.button("Sfoglia…", on_click=sfoglia_cartella, width="stretch")

    trovati = trova_modelli(st.session_state.cartella)
    solo_mamba = st.toggle("Mostra solo modelli Mamba-1", value=True)
    elenco = [m for m in trovati if m.is_mamba] if solo_mamba else trovati
    if not trovati:
        st.caption("Nessun modello trovato in questa cartella.")
    elif not elenco:
        st.caption(f"{len(trovati)} modelli trovati, nessuno Mamba-1. Disattiva il filtro per vederli.")

    scelto = None
    if elenco:
        scelto = st.selectbox("Modello", elenco,
                              format_func=lambda m: m.nome if m.is_mamba else f"{m.nome} [{m.tipo}]")
        st.caption(f"Pesi: {scelto.dimensione_gb:.2f} GB su disco · commit {scelto.commit}")
    disp = st.radio("Esegui su", list(DISPOSITIVI), format_func=DISPOSITIVI.get, horizontal=True,
                    index=list(DISPOSITIVI).index(st.session_state.dispositivo))
    if scelto is not None and disp == "cpu" and scelto.dimensione_gb > 2:
        st.caption(f"⚠ Modello grande su CPU: servono circa {scelto.dimensione_gb * 1.3:.1f} GB di RAM liberi.")
    if st.button("Carica e verifica", type="primary", disabled=scelto is None, width="stretch"):
        carica_modello(scelto, disp)

    info = st.session_state.info
    if info is not None:
        riga_stato(info["nome"], f"caricato su {DISPOSITIVI[st.session_state.dispositivo]} · FP32")
        titolo_passo(testa, 1, "Modello", "pronto", "ok")
    elif st.session_state.controlli:
        titolo_passo(testa, 1, "Modello", "errore", "errore")
    else:
        titolo_passo(testa, 1, "Modello", "da caricare", "attesa")


def sezione_testo():
    testa = st.empty()
    if st.session_state.model is None:
        titolo_passo(testa, 2, "Testo", "in attesa", "attesa")
        st.caption("Carica prima un modello: il testo va tokenizzato con il suo tokenizer.")
        return
    fonte = st.radio("Fonte", ["Testo di esempio", "Incolla un testo", "File .txt"], label_visibility="collapsed")
    grezzo = caricato = None
    if fonte == "Testo di esempio":
        st.caption("Breve testo sulla teoria del controllo (211 token): un riferimento fisso per confrontare "
                   "esecuzioni diverse.")
        grezzo = TESTO_ESEMPIO
    elif fonte == "Incolla un testo":
        grezzo = st.text_area("Testo", height=160, placeholder="Incolla qui il testo (in inglese, "
                              "come i dati di addestramento di Mamba).", label_visibility="collapsed")
    else:
        caricato = st.file_uploader("File di testo", type=["txt", "md"], label_visibility="collapsed")
        if caricato is not None:
            grezzo = caricato.getvalue().decode("utf-8", errors="replace")
    max_token = st.number_input("Token massimi", 16, 8192, MAX_TOKEN_PREDEFINITO, step=64,
                                help="I testi più lunghi vengono troncati. Su una GPU da 8 GB con un modello "
                                     "da circa 1.4B parametri, 1024 è un limite prudente.")
    if st.button("Usa questo testo", disabled=not (grezzo and grezzo.strip()), width="stretch"):
        nome_fonte = {"Testo di esempio": "testo di esempio", "Incolla un testo": "testo incollato"}.get(
            fonte, f"file {getattr(caricato, 'name', '')}")
        imposta_testo(grezzo.strip(), nome_fonte, int(max_token))
    t = st.session_state.testo
    if t is None:
        titolo_passo(testa, 2, "Testo", "da scegliere", "attesa")
        return
    n = t["ids"].shape[1]
    troncato = f", troncato da {t['n_originali']}" if t["n_originali"] > t["max_token"] else ""
    riga_stato(f"{n} token in uso", f"{t['fonte']}{troncato}")
    titolo_passo(testa, 2, "Testo", f"{n} token", "ok")


def sezione_fase(fasi, errori_fasi):
    testa = st.empty()
    fase = None
    if fasi:
        fase = st.radio("Fase", fasi, label_visibility="collapsed",
                        format_func=lambda f: f"Fase {f.codice} · {f.nome}", captions=[f.breve for f in fasi])
    else:
        st.error("Nessuna fase trovata nella cartella fasi/.")
    for nome, errore in errori_fasi:
        st.error(f"Errore nel file fasi/{nome}.py", icon="⚠️")
        with st.expander("Dettagli"):
            st.code(errore)
    if st.button("Ricarica le fasi", help="Rilegge i file in fasi/ dopo averne aggiunto o modificato uno.",
                 width="stretch"):
        carica_fasi(ricarica=True)
        st.rerun()
    titolo_passo(testa, 3, "Fase", fase.codice if fase else "nessuna", "codice" if fase else "errore")
    return fase


fasi, errori_fasi = carica_fasi()
with st.sidebar:
    mostra_html(f"<div class='marchio'>{svg_marchio()}<div><div class='marchio-nome'>Mamba<span>·</span>Modi</div>"
                f"<div class='marchio-sotto'>analisi di poli e stabilità</div></div></div>")
    with st.container(key="passo_modello"):
        sezione_modello()
    with st.container(key="passo_testo"):
        sezione_testo()
    with st.container(key="passo_fase"):
        fase = sezione_fase(fasi, errori_fasi)
        salva_auto = st.toggle("Salva ogni risultato anche in risultati/", value=True,
                               help=f"Cartella: {CARTELLA_RISULTATI}")
    mostra_html("<div class='piede'>Elaborazione locale · nessuna connessione di rete</div>")

if st.session_state.messaggio:
    tipo, testo_msg = st.session_state.messaggio
    getattr(st, tipo)(testo_msg)
    st.session_state.messaggio = None


# Pagina principale: blocchi di presentazione
def hero(sopra, titolo, sotto):
    mostra_html(f"<div class='hero'><div><div class='sopra'>{html_sicuro(sopra)}</div>"
                f"<div class='titolo'>{html_sicuro(titolo)}</div><div class='sotto'>{sotto}</div></div>"
                f"{svg_hero()}</div>")


def mostra_controlli(controlli):
    for c in controlli:
        dettaglio = f" — <span style='color:#8a93a3'>{html_sicuro(c.dettaglio)}</span>" if c.dettaglio else ""
        mostra_html(f"<div class='controllo'>{'✅' if c.ok else '❌'} {html_sicuro(c.descrizione)}{dettaglio}</div>")


BADGE = {"ok": "<span class='stato ok'>✓ ok</span>",
         "attenzione": "<span class='stato attenzione'>! da notare</span>"}


def riquadri(indicatori):
    parti = []
    for ind in indicatori:
        nota = f"<div class='nota'>{html_sicuro(ind.nota)}</div>" if ind.nota else ""
        parti.append(f"<div class='riquadro {ind.stato or ''}'>"
                     f"<div class='et'>{html_sicuro(ind.etichetta)}{BADGE.get(ind.stato, '')}</div>"
                     f"<div class='val'>{html_sicuro(ind.valore)}</div>{nota}</div>")
    mostra_html(f"<div class='riquadri'>{''.join(parti)}</div>")


# Titolo con sottotitolo e, a destra, un popover con la guida alla lettura
def titolo_con_legenda(titolo, spiegazione, legenda, etichetta="ⓘ Come leggere"):
    sinistra, destra = st.columns([5, 1], vertical_alignment="bottom")
    with sinistra:
        st.subheader(titolo)
        if spiegazione:
            mostra_html(f"<div class='sottotitolo-grafico'>{html_sicuro(spiegazione)}</div>")
    if legenda:
        with destra.popover(etichetta, width="stretch"):
            st.markdown(legenda)


# Tabella per strato in HTML: stessi numeri dell'Excel, legenda nei tooltip delle intestazioni, righe citate in ambra
def tabella(ris):
    legenda = dict(ris.legenda)
    intestazioni = []
    for j, c in enumerate(ris.colonne):
        aiuto = legenda.get(c.sottotitolo) or legenda.get(c.titolo)
        titolo = html_sicuro(c.titolo)
        if aiuto:
            titolo = f"<span class='aiuto'>{titolo}<span class='suggerimento'>{html_sicuro(aiuto)}</span></span>"
        classi = " ".join(filter(None, ["centro" if c.chiave == "strato" else "", "a-sinistra" if j < 2 else ""]))
        sotto = f"<small>{html_sicuro(c.sottotitolo)}</small>" if c.sottotitolo else ""
        intestazioni.append(f"<th class='{classi}'>{titolo}{sotto}</th>")
    if ris.evidenzia:
        intestazioni.append("<th class='sinistra'>Nota</th>")

    righe = []
    for r in ris.righe:
        celle = []
        for c in ris.colonne:
            valore = html_sicuro(FORMATI[c.formato][0](r.get(c.chiave)))
            if c.chiave == "strato":
                celle.append(f"<td class='centro'><span class='badge-strato'>{valore}</span></td>")
            else:
                celle.append(f"<td class='{'sinistra' if c.formato == 'testo' else ''}'>{valore}</td>")
        nota = ris.evidenzia.get(r.get("strato"))
        if ris.evidenzia:
            celle.append(f"<td class='sinistra'>{f'<span class=nota-chip>{html_sicuro(nota)}</span>' if nota else ''}</td>")
        righe.append(f"<tr class='{'citata' if nota else ''}'>{''.join(celle)}</tr>")

    mostra_html(f"<div class='tabella-wrap'><table class='tabella'><thead><tr>{''.join(intestazioni)}</tr></thead>"
                f"<tbody>{''.join(righe)}</tbody></table></div>")


PASSI = [("01", "Scegli il modello", "Nella barra a sinistra, dalla cartella dei modelli scaricati. "
          "Il programma lo carica in FP32 e controlla che sia analizzabile."),
         ("02", "Scegli il testo", "Serve alle fasi che fanno leggere il modello (2a, 2b). La Fase 1 legge solo i pesi."),
         ("03", "Esegui una fase", "Risultati in riquadri, grafici e tabella, con la guida alla lettura. "
          "Ogni fase si può scaricare in Excel.")]


def pagina_benvenuto():
    hero("Analisi spettrale di modelli state-space", "Analisi dei modi di Mamba",
         "Misura poli, passi di campionamento e margini di stabilità di un modello Mamba-1, strato per strato. "
         "Tutto gira sul tuo PC, senza connessione.")
    capacita = "".join(f"<span>{html_sicuro(f.nome)}</span>" for f in fasi) + "<span>Esportazione in Excel</span>"
    mostra_html(f"<div class='capacita'>{capacita}</div>")
    passi = "".join(f"<div class='passo'><div class='num'>{n}</div><div class='tit'>{t}</div>"
                    f"<div class='desc'>{d}</div></div>" for n, t, d in PASSI)
    mostra_html(f"<div class='passi'>{passi}</div>")
    if st.session_state.controlli:
        st.subheader("Esito dell'ultima verifica")
        mostra_controlli(st.session_state.controlli)
        st.error("Il modello non è stato caricato: vedi il controllo non superato qui sopra.")


def intestazione_modello(info):
    hero("Modello caricato", info["nome"],
         f"{html_sicuro(DISPOSITIVI[st.session_state.dispositivo])} · FP32 · commit {html_sicuro(info['commit'])}")
    voci = [("Strati", info["strati"]), ("Canali per strato", fmt_int(info["canali"])),
            ("Stati per canale", info["stati"]), ("Poli continui", fmt_int(info["poli_continui"])),
            ("Pesi su disco", f"{info['dimensione_gb']:.2f} GB")]
    mostra_html("<div class='scheda-tecnica'>" + "".join(
        f"<div class='voce'><div class='et'>{e}</div><div class='val'>{html_sicuro(v)}</div></div>" for e, v in voci)
        + "</div>")
    with st.expander("Verifica del modello superata", icon="✅"):
        mostra_controlli(st.session_state.controlli)
        st.caption(f"Cartella: `{info['percorso']}` · commit {info['commit']} · rango di Δ: {info['rank_delta']}")


# Esecuzione di una fase e memorizzazione del risultato (con salvataggio Excel opzionale)
def chiave_esecuzione(fase, parametri):
    t = st.session_state.testo
    return (st.session_state.info["percorso"], st.session_state.dispositivo,
            t["impronta"] if (fase.usa_testo and t) else None, tuple(sorted(parametri.items())))


def esegui_fase(fase, parametri):
    info = st.session_state.info
    nome_disp = DISPOSITIVI[st.session_state.dispositivo]
    t = (st.session_state.testo if fase.usa_testo else None) or {}
    barra = st.progress(0.0, text=f"Fase {fase.codice} in corso...")
    ctx = Contesto(
        model=st.session_state.model, tok=st.session_state.tok, info_modello=info,
        dispositivo=st.session_state.dispositivo, nome_dispositivo=nome_disp,
        ids=t.get("ids"), token=t.get("token", []), testo=t.get("testo"), fonte_testo=t.get("fonte"),
        parametri=parametri,
        avanzamento=lambda f, m="": barra.progress(min(max(f, 0.0), 1.0), text=f"Fase {fase.codice}: {m}"))
    t0 = time.time()
    try:
        ris = fase.esegui(ctx)
    except Exception as e:
        import torch
        barra.empty()
        if isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            st.error("Memoria della GPU esaurita: riduci i token massimi o usa un testo più corto.")
        else:
            st.error(f"Errore durante la fase: {type(e).__name__}: {e}")
            st.exception(e)
        return
    barra.empty()
    ris.info = dict(
        modello=info["nome"], percorso=info["percorso"], commit=info["commit"],
        struttura=f"{info['strati']} strati × {info['canali']} canali × {info['stati']} stati",
        dispositivo=nome_disp, data=datetime.now(), durata=time.time() - t0, parametri=parametri,
        testo=t.get("testo"), fonte_testo=t.get("fonte"),
        n_token=int(t["ids"].shape[1]) if t else None,
        chiave=chiave_esecuzione(fase, parametri))
    if salva_auto:
        try:
            ris.info["file"] = salva_excel(ris, CARTELLA_RISULTATI)
        except OSError as e:
            st.warning(f"Salvataggio in risultati/ non riuscito: {e}")
    st.session_state.risultati[fase.codice] = ris


VISTE = ["📈 Grafici", "🧮 Tabella per strato", "📝 Testo analizzato"]


# Presentazione di un risultato: contesto, riquadri, selettore della vista e download
def mostra_risultato(fase, ris, parametri):
    i = ris.info
    voci = [("modello", i["modello"]), ("dispositivo", i["dispositivo"]), ("formato", ris.note_formato),
            ("data", f"{i['data']:%d/%m/%Y %H:%M}"), ("durata", f"{i['durata']:.1f} s")]
    if i.get("n_token"):
        voci.append(("testo", f"{i['n_token']} token · {i['fonte_testo']}"))
    mostra_html("<div class='contesto'>" + "".join(f"<span><i>{e}</i>{html_sicuro(v)}</span>" for e, v in voci)
                + "</div>")
    if i.get("chiave") != chiave_esecuzione(fase, parametri):
        st.warning("Questo risultato è stato ottenuto con un modello, un testo o dei parametri diversi da "
                   "quelli selezionati ora. Premi **Esegui** per aggiornarlo.", icon="🔁")

    riquadri(ris.indicatori)

    vista = st.segmented_control("Vista", VISTE, default=VISTE[0], key=f"vista_{ris.codice}",
                                 label_visibility="collapsed") or VISTE[0]
    if vista == VISTE[0]:
        for k, graf in enumerate(ris.grafici):
            titolo_con_legenda(graf.titolo, graf.spiegazione, graf.come_leggere)
            with st.container(border=True):
                st.plotly_chart(graf.figura, width="stretch", config=g.CONFIG, theme=None,
                                key=f"grafico_{ris.codice}_{k}")
        st.caption("Suggerimento: l'icona a forma di fotocamera sopra ogni grafico salva un PNG ad alta "
                   "risoluzione, pronto per relazioni e pubblicazioni. Trascinando si ingrandisce; "
                   "doppio clic per tornare indietro.")
    elif vista == VISTE[1]:
        titolo_con_legenda(ris.titolo, ris.descrizione,
                           "\n".join(f"- **{nome}** — {spiegazione}" for nome, spiegazione in ris.legenda),
                           "ⓘ Legenda colonne")
        tabella(ris)
        st.caption("Passa il mouse sulle intestazioni sottolineate per la spiegazione della colonna."
                   + (" Le righe in ambra sono quelle citate nella colonna Nota." if ris.evidenzia else ""))
    else:
        if i.get("testo"):
            st.text_area("Testo analizzato", i["testo"], height=260, disabled=True, label_visibility="collapsed")
        else:
            st.caption("Questa fase non usa il testo: legge solo i pesi.")

    c1, c2 = st.columns([1, 2])
    c1.download_button("⬇ Scarica i risultati in Excel", data=crea_excel(ris), file_name=nome_file(ris),
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary", width="stretch")
    if i.get("file"):
        c2.caption(f"Salvato anche in `{i['file']}`")


if st.session_state.model is None or fase is None:
    pagina_benvenuto()
else:
    intestazione_modello(st.session_state.info)
    mostra_html(f"<div class='testa-fase'><span class='codice'>FASE {html_sicuro(fase.codice)}</span>"
                f"<span class='nome'>{html_sicuro(fase.nome)}</span></div>"
                f"<div class='breve-fase'>{html_sicuro(fase.breve)}</div>")
    with st.expander("Cosa misura questa fase", icon="📘", expanded=fase.codice not in st.session_state.risultati):
        st.markdown(fase.spiegazione)

    parametri = {}
    if fase.parametri:
        colonne = st.columns(min(len(fase.parametri), 4))
        for col, p in zip(colonne, fase.parametri):
            parametri[p.chiave] = col.selectbox(p.etichetta, p.scelte,
                                                index=p.scelte.index(p.predefinito) if p.predefinito in p.scelte else 0,
                                                help=p.aiuto or None, key=f"param_{fase.codice}_{p.chiave}")

    manca_testo = fase.usa_testo and st.session_state.testo is None
    if manca_testo:
        st.info("Questa fase fa leggere un testo al modello: sceglilo nella sezione ② della barra laterale "
                "(il testo di esempio dà risultati ripetibili).", icon="📄")
    if st.button(f"▶ Esegui la Fase {fase.codice}", type="primary", disabled=manca_testo):
        esegui_fase(fase, parametri)

    ris = st.session_state.risultati.get(fase.codice)
    if ris is not None:
        mostra_risultato(fase, ris, parametri)
