import html
import itertools
import math
import os
import time
import traceback
from datetime import datetime

import streamlit as st

from fasi import carica_fasi
from nucleo import corpora
from nucleo import grafica as g
from nucleo.excel import crea_excel, nome_file, nome_file_unico, salva_excel, salva_excel_unico
from nucleo.formato import FORMATI, fmt_int
from nucleo.modelli import (carica_e_verifica, cartella_cache_predefinita, dispositivi_disponibili, libera,
                            trova_modelli, versioni)
from nucleo.precisione import svuota_cache
from nucleo.risultato import Contesto
from nucleo.testo import MAX_TOKEN_PREDEFINITO, TESTO_ESEMPIO, prepara

CARTELLA_APP = os.path.dirname(os.path.abspath(__file__))
CARTELLA_RISULTATI = os.path.join(CARTELLA_APP, "risultati")

st.set_page_config(page_title="Analisi modi di Mamba", page_icon="〰", layout="wide")

# Stile della pagina: tema scuro blu notte, numeri in monospazio, griglia da carta millimetrata
st.markdown("""
<style>
  :root {--sfondo: #0b1220; --superficie: #111a2c; --superficie-2: #17223a; --linea: #22304a;
         --notte: #e6ecf5; --testo-2: #a9b5c9; --tenue: #6f7e97;
         --blu: #4c9aff; --blu-chiaro: #8cc2ff; --verde: #3ccf73; --ambra: #f5b83d; --rosso: #ff6b6b;
         --mono: "Cascadia Code", "JetBrains Mono", Consolas, monospace;
         --ombra: 0 1px 2px rgba(0,0,0,0.3), 0 6px 18px rgba(0,0,0,0.28);}
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  header[data-testid="stHeader"] {background: transparent;}
  h1, h2, h3, h4 {letter-spacing: -0.01em; color: var(--notte);}

  /* Timeline in alto, sempre visibile: Modello → Testo → Fasi, ridisegnata mentre il programma lavora */
  /* Lo sticky va sull'elemento figlio del blocco principale: il contenitore o, se c'è, il suo involucro */
  .st-key-timeline, [data-testid="stLayoutWrapper"]:has(> .st-key-timeline) {position: sticky; top: 0; z-index: 990;
                                                                              background: var(--sfondo);}
  .st-key-timeline {padding: 4px 0 10px 0;}
  .tl {display: flex; align-items: flex-start; min-height: 112px; background: var(--superficie);
       border: 1px solid var(--linea); border-radius: 14px; padding: 12px 14px; box-shadow: var(--ombra);}
  .tl-marchio {display: flex; align-items: center; gap: 10px; align-self: stretch; padding-right: 16px;
               margin-right: 8px; border-right: 1px solid var(--linea);}
  .tl-marchio b {display: block; font-size: 1rem; color: var(--notte); letter-spacing: 0.01em;}
  .tl-marchio b span {color: var(--blu-chiaro); margin: 0 3px;}
  .tl-marchio small {display: block; font-size: 0.68rem; color: var(--tenue); letter-spacing: 0.04em;}
  .tl-nodo {flex: 1 1 0; min-width: 0; padding: 6px 10px 8px 10px; border-radius: 10px;}
  .tl-nodo.fasi {flex: 1.8 1 0;}
  .tl-nodo.attivo {background: rgba(76,154,255,0.07); box-shadow: inset 0 0 0 1px rgba(76,154,255,0.4);}
  .tl-testa {display: flex; align-items: center; gap: 9px;}
  .tl-punto {display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; flex: none;
             border-radius: 50%; background: var(--superficie-2); color: var(--tenue); font: 600 0.74rem var(--mono);
             border: 1px solid #33425e;}
  .tl-nodo.fatto .tl-punto {background: rgba(60,207,115,0.16); color: var(--verde); border-color: rgba(60,207,115,0.5);}
  .tl-nodo.corso .tl-punto {background: var(--blu); color: #fff; border-color: var(--blu);
                            box-shadow: 0 0 0 4px rgba(76,154,255,0.22);}
  .tl-nodo.errore .tl-punto {background: rgba(255,107,107,0.16); color: var(--rosso); border-color: rgba(255,107,107,0.5);}
  .tl-tit {font-weight: 650; color: var(--notte); font-size: 0.95rem; flex: 1;}
  .tl-corpo {margin: 5px 0 0 33px; font-size: 0.8rem; color: var(--testo-2); line-height: 1.45;}
  .tl-corpo > div:not(.tl-tacche) {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
  .tl-corpo b {color: var(--notte); font-weight: 600;}
  .tl-linea {flex: 0 0 32px; height: 2px; margin-top: 18px; border-radius: 2px; background: var(--linea);}
  .tl-linea.fatta {background: linear-gradient(90deg, var(--verde), rgba(60,207,115,0.4));}
  .tl-tacche {display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 2px 0 4px 0;}
  .tl-gruppo {display: inline-flex; align-items: center; gap: 3px;}
  .tl-gruppo i {font: 600 0.68rem var(--mono); font-style: normal; color: var(--tenue); margin-right: 2px;}
  .tl-tacca {width: 14px; height: 10px; border-radius: 3px; background: rgba(255,255,255,0.06);
             border: 1px solid #33425e;}
  .tl-tacca.ok {background: var(--verde); border-color: var(--verde);}
  .tl-tacca.errore {background: var(--rosso); border-color: var(--rosso);}
  .tl-tacca.corso {background: var(--blu); border-color: var(--blu-chiaro); box-shadow: 0 0 0 3px rgba(76,154,255,0.3);}
  .tl-barra {height: 4px; border-radius: 2px; background: rgba(255,255,255,0.08); margin-top: 6px; overflow: hidden;}
  .tl-barra div {height: 100%; background: linear-gradient(90deg, var(--blu), var(--blu-chiaro));}
  @media (max-width: 900px) {.tl-marchio div {display: none;}}

  [class*="st-key-passo_"] {background: var(--superficie); border: 1px solid var(--linea); border-radius: 14px;
                            padding: 14px 14px 12px 14px; box-shadow: var(--ombra);}
  .passo-testa {display: flex; align-items: center; gap: 10px; margin-bottom: 2px;}
  .passo-num {display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; flex: none;
              border-radius: 50%; background: var(--superficie-2); color: var(--notte); font: 600 0.78rem var(--mono);
              border: 1px solid #33425e; box-shadow: 0 0 0 4px rgba(76,154,255,0.06);}
  .passo-num.fatto {background: rgba(60,207,115,0.16); color: var(--verde); border-color: rgba(60,207,115,0.5);
                    box-shadow: 0 0 0 4px rgba(60,207,115,0.07);}
  .passo-tit {font-weight: 650; color: var(--notte); font-size: 1rem; flex: 1; line-height: 1.15;}
  .passo-tit small {display: block; font: 600 0.62rem var(--mono); letter-spacing: 0.14em; text-transform: uppercase;
                    color: var(--tenue); margin-bottom: 2px;}
  .chip {display: inline-block; font: 600 0.7rem var(--mono); padding: 2px 8px; border-radius: 999px;
         letter-spacing: 0.02em; white-space: nowrap;}
  .chip.attesa {background: rgba(255,255,255,0.05); color: var(--tenue); border: 1px solid var(--linea);}
  .chip.ok {background: rgba(60,207,115,0.14); color: #7ee2a3;}
  .chip.errore {background: rgba(255,107,107,0.14); color: #ff9b9b;}
  .chip.codice {background: rgba(76,154,255,0.16); color: var(--blu-chiaro);}
  .piede {font-size: 0.72rem; color: var(--tenue); text-align: center; margin-top: 28px; padding-top: 12px;
          letter-spacing: 0.03em; border-top: 1px solid var(--linea);}

  .hero {position: relative; overflow: hidden; display: flex; justify-content: space-between; align-items: center;
         gap: 20px; color: #fff; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
         border: 1px solid rgba(140,194,255,0.16); box-shadow: 0 14px 34px rgba(0,0,0,0.45);
         background-image: linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px),
                           linear-gradient(125deg, #0d1f3c 0%, #163e74 55%, #2a6fc4 100%);
         background-size: 24px 24px, 24px 24px, auto;}
  .hero .sopra {font: 600 0.72rem var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--blu-chiaro);}
  .hero .titolo {font-size: 2.05rem; font-weight: 700; line-height: 1.15; margin-top: 4px;}
  .hero .sotto {font-size: 0.98rem; opacity: 0.85; margin-top: 8px; max-width: 760px; line-height: 1.5;}
  .hero svg {flex: none;}
  @media (max-width: 900px) {.hero svg {display: none;}}

  .capacita {display: flex; flex-wrap: wrap; gap: 8px; margin: -4px 0 20px 0;}
  .capacita span {background: var(--superficie); border: 1px solid var(--linea); border-radius: 999px; padding: 4px 12px;
                  font-size: 0.82rem; color: var(--testo-2);}
  .capacita span::before {content: "◆"; color: var(--blu); font-size: 0.6rem; margin-right: 7px; vertical-align: 2px;}
  /* Schede delle fasi nella pagina del testo: una per fase, con il suo pulsante */
  .scheda-fase .testa {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}
  .scheda-fase .codice {font: 600 0.74rem var(--mono); letter-spacing: 0.1em; color: var(--blu-chiaro);
                        background: rgba(76,154,255,0.14); border: 1px solid rgba(76,154,255,0.4);
                        padding: 3px 9px; border-radius: 8px;}
  .scheda-fase .nome {font-weight: 650; color: var(--notte); font-size: 1.02rem;}
  .scheda-fase .breve {font-size: 0.86rem; color: var(--testo-2); margin-top: 5px; line-height: 1.45;}
  .scheda-fase .analisi {font: 500 0.74rem var(--mono); color: var(--tenue); margin-top: 5px;}
  .gruppo-fasi {font: 600 0.74rem var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--tenue);
                margin: 14px 0 6px 0;}

  .scheda-tecnica {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); background: var(--superficie);
                   border: 1px solid var(--linea); border-radius: 14px; box-shadow: var(--ombra); margin-bottom: 12px;}
  .scheda-tecnica .voce {padding: 14px 18px;}
  .scheda-tecnica .voce + .voce {border-left: 1px solid var(--linea);}
  .scheda-tecnica .et {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--tenue);}
  .scheda-tecnica .val {font: 600 1.35rem var(--mono); color: var(--notte); margin-top: 4px;}

  .testa-fase {display: flex; align-items: center; gap: 12px; margin: 10px 0 2px 0;}
  .testa-fase .codice {font: 600 0.78rem var(--mono); letter-spacing: 0.1em; color: var(--blu-chiaro);
                       background: rgba(76,154,255,0.14); border: 1px solid rgba(76,154,255,0.4);
                       padding: 4px 10px; border-radius: 8px;}
  .testa-fase .nome {font-size: 1.6rem; font-weight: 700; color: var(--notte);}
  .breve-fase {color: var(--testo-2); font-size: 0.92rem; margin-bottom: 10px;}

  .riquadri {display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin: 6px 0 18px 0;}
  .riquadro {background: var(--superficie); border: 1px solid var(--linea); border-left: 4px solid var(--blu);
             border-radius: 12px; padding: 13px 15px 12px 15px; box-shadow: var(--ombra);
             transition: transform 0.15s ease, box-shadow 0.15s ease;}
  .riquadro:hover {transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0,0,0,0.45);}
  .riquadro.ok {border-left-color: var(--verde);}
  .riquadro.attenzione {border-left-color: var(--ambra);}
  .riquadro .et {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--tenue);
                 margin-bottom: 5px;}
  .riquadro .val {font: 600 1.38rem var(--mono); color: var(--notte); line-height: 1.25;}
  .riquadro .nota {font-size: 0.78rem; color: var(--testo-2); margin-top: 4px; line-height: 1.35;}
  .stato {display: inline-block; font: 600 0.66rem var(--mono); padding: 1px 7px; border-radius: 999px;
          margin-left: 6px; vertical-align: 1px; text-transform: none; letter-spacing: 0;}
  .stato.ok {background: rgba(60,207,115,0.14); color: #7ee2a3;}
  .stato.attenzione {background: rgba(245,184,61,0.16); color: #ffd27a;}

  .contesto {display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;}
  .contesto span {background: var(--superficie); border: 1px solid var(--linea); border-radius: 8px; padding: 3px 10px;
                  font-size: 0.8rem; color: var(--testo-2);}
  .contesto span i {font-style: normal; color: var(--tenue); margin-right: 5px; font-size: 0.72rem;
                    text-transform: uppercase; letter-spacing: 0.06em;}
  .controllo {font-size: 0.88rem; margin: 2px 0;}
  .controllo .dettaglio {color: var(--tenue);}
  .sottotitolo-grafico {color: var(--testo-2); font-size: 0.92rem; margin-top: -6px;}

  [class*="st-key-vista_"] {margin: 4px 0 6px 0;}

  .articolo {background: var(--superficie-2); border: 1px solid var(--linea); border-radius: 12px;
             padding: 12px 15px; margin: 4px 0 8px 0;}
  .articolo .tit {font-weight: 650; color: var(--notte); font-size: 1.02rem; line-height: 1.3;}
  .articolo .meta {display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 6px; font-size: 0.8rem;
                   color: var(--testo-2);}
  .articolo .meta i {font-style: normal; color: var(--tenue); margin-right: 5px; font-size: 0.72rem;
                     text-transform: uppercase; letter-spacing: 0.06em;}
  .articolo a {color: var(--blu-chiaro);}
  .visto {display: inline-block; font: 600 0.7rem var(--mono); padding: 2px 9px; border-radius: 999px;
          white-space: nowrap;}
  .visto.mai {background: rgba(60,207,115,0.14); color: #7ee2a3;}
  .visto.forse {background: rgba(245,184,61,0.16); color: #ffd27a;}

  .tabella-wrap {max-height: 640px; overflow: auto; background: var(--superficie); border: 1px solid var(--linea);
                 border-radius: 14px; box-shadow: var(--ombra); margin: 4px 0 8px 0;}
  table.tabella {width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.88rem; margin: 0;}
  table.tabella th {position: sticky; top: 0; z-index: 2; background: var(--superficie-2); color: var(--notte);
                    text-align: right; font-weight: 650; padding: 11px 14px 9px 14px; vertical-align: bottom;
                    border: none; border-bottom: 2px solid #2c3d5c; white-space: nowrap;}
  table.tabella th small {display: block; font-weight: 400; font-size: 0.72rem; color: var(--tenue); margin-top: 2px;}
  table.tabella td {padding: 7px 14px; text-align: right; border: none; border-bottom: 1px solid rgba(255,255,255,0.04);
                    font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--notte);}
  table.tabella .centro {text-align: center;}
  table.tabella .sinistra {text-align: left; font-family: inherit;}
  table.tabella tbody tr:nth-child(even) td {background: rgba(255,255,255,0.02);}
  table.tabella tbody tr:hover td {background: rgba(76,154,255,0.08);}
  table.tabella tbody tr.citata td {background: rgba(245,184,61,0.08);}
  table.tabella tbody tr.citata td:first-child {box-shadow: inset 3px 0 0 var(--ambra);}
  table.tabella tbody tr:last-child td {border-bottom: none;}
  .badge-strato {display: inline-block; min-width: 30px; padding: 1px 6px; border-radius: 6px;
                 background: rgba(76,154,255,0.12); color: var(--blu-chiaro); font-weight: 600;}
  .nota-chip {display: inline-block; font-size: 0.74rem; background: rgba(245,184,61,0.16); color: #ffd27a;
              padding: 2px 9px; border-radius: 999px; white-space: nowrap;}
  table.tabella th .aiuto {position: relative; cursor: help; border-bottom: 1px dashed var(--tenue);}
  table.tabella th .suggerimento {visibility: hidden; opacity: 0; position: absolute; top: calc(100% + 10px); right: 0;
                                  width: 260px; white-space: normal; text-align: left; font-weight: 400;
                                  font-size: 0.78rem; line-height: 1.4; color: var(--notte); background: #1c2945;
                                  border: 1px solid #33425e; padding: 8px 11px; border-radius: 8px; z-index: 5;
                                  transition: opacity 0.12s ease; box-shadow: 0 10px 24px rgba(0,0,0,0.5);}
  table.tabella th.a-sinistra .suggerimento {right: auto; left: 0;}
  table.tabella th .aiuto:hover .suggerimento {visibility: visible; opacity: 1;}
</style>
""", unsafe_allow_html=True)

# Stato della sessione: passo della procedura (modello → testo → risultati), risultati in memoria e orari
# della timeline (modello, testo, fasi_inizio, fasi_fine)
DISPOSITIVI = dict(dispositivi_disponibili())
PREDEFINITI = dict(cartella=cartella_cache_predefinita(), model=None, tok=None, info=None, controlli=[],
                   dispositivo=next(iter(DISPOSITIVI)), testo=None, passo="modello",
                   risultati={}, errori={}, esito_salvataggio=None, messaggio=None, orari={})
for chiave, valore in PREDEFINITI.items():
    st.session_state.setdefault(chiave, valore)

FONTI = ["Testo di prova", "Estrai da un corpus", "Scrivi o incolla", "File .txt"]


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


def svuota_risultati():
    st.session_state.update(risultati={}, errori={}, esito_salvataggio=None)


def vai(passo):
    st.session_state.passo = passo


def ricarica_fasi():
    carica_fasi(ricarica=True)
    svuota_risultati()
    if st.session_state.passo == "risultati":
        st.session_state.passo = "testo"


# Un modello nuovo azzera anche il testo: i suoi token vengono dal tokenizer del modello precedente
def carica_modello(modello, dispositivo):
    if st.session_state.model is not None:
        st.session_state.update(model=None, tok=None, info=None)
        libera()
    st.session_state.update(testo=None, controlli=[])
    svuota_risultati()
    svuota_cache()
    model, tok, info, controlli = carica_e_verifica(
        modello, dispositivo,
        lambda superati, prossimo: disegna_timeline(("modello", modello.nome, superati, prossimo)))
    st.session_state.update(model=model, tok=tok, info=info, controlli=controlli, dispositivo=dispositivo)
    st.session_state.orari["modello"] = datetime.now()
    disegna_timeline()


# Un testo nuovo rende inutili i passaggi a bassa precisione in cache (Fasi 2 e 3).
# dettagli: righe (etichetta, valore) sulla provenienza del testo, per l'Excel (testi estratti da un corpus)
def imposta_testo(grezzo, fonte, max_token, dettagli=None):
    svuota_cache()
    ids, token, testo, n_originali = prepara(st.session_state.tok, grezzo, st.session_state.dispositivo, max_token)
    st.session_state.testo = dict(grezzo=grezzo, fonte=fonte, max_token=max_token, ids=ids, token=token,
                                  testo=testo, n_originali=n_originali, dettagli=dettagli or [])
    st.session_state.orari["testo"] = datetime.now()
    disegna_timeline()


# Fasi principali (partono con "Esegui tutte le fasi") e fasi extra (pulsante a parte, es. la diagnostica della Fase 6)
def principali(fasi):
    return [f for f in fasi if not f.extra]


def extra(fasi):
    return [f for f in fasi if f.extra]


# Ogni fase con tutte le combinazioni dei suoi parametri (es. 3a in FP16, BF16 e FP32 controllo)
def lavori(fasi):
    elenco = []
    for fase in fasi:
        for scelte in itertools.product(*[p.scelte for p in fase.parametri]):
            elenco.append((fase, {p.chiave: v for p, v in zip(fase.parametri, scelte)}))
    return elenco


def chiave(fase, parametri):
    return fase.codice, tuple(sorted(parametri.items()))


def descrivi_lavoro(fase, parametri):
    return f"Fase {fase.codice}" + "".join(f" · {v}" for v in parametri.values())


# Una fase: restituisce (risultato, None) oppure (None, (messaggio, dettaglio))
def esegui_fase(fase, parametri, avanzamento):
    info = st.session_state.info
    nome_disp = DISPOSITIVI[st.session_state.dispositivo]
    t = (st.session_state.testo if fase.usa_testo else None) or {}
    ctx = Contesto(
        model=st.session_state.model, tok=st.session_state.tok, info_modello=info,
        dispositivo=st.session_state.dispositivo, nome_dispositivo=nome_disp,
        ids=t.get("ids"), token=t.get("token", []), testo=t.get("testo"), fonte_testo=t.get("fonte"),
        parametri=parametri, avanzamento=avanzamento)
    t0 = time.time()
    try:
        ris = fase.esegui(ctx)
    except Exception as e:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            dettaglio = str(e).split("\n")[0]
            occupata = torch.cuda.memory_allocated() / 2 ** 30
            torch.cuda.empty_cache()
            return None, (f"Memoria della GPU esaurita ({occupata:.2f} GB occupati da PyTorch al momento dell'errore). "
                          "Chiudi i programmi che usano la GPU (es. video nel browser), riduci i token massimi o usa "
                          "un testo più corto.", f"{dettaglio}\n\n{traceback.format_exc()}")
        return None, (f"{type(e).__name__}: {e}", traceback.format_exc())
    ris.info = dict(
        modello=info["nome"], percorso=info["percorso"], commit=info["commit"],
        struttura=f"{info['strati']} strati × {info['canali']} canali × {info['stati']} stati",
        dispositivo=nome_disp, data=datetime.now(), durata=time.time() - t0, parametri=parametri,
        testo=t.get("testo"), fonte_testo=t.get("fonte"), dettagli_testo=t.get("dettagli", []), versioni=versioni(),
        n_token=int(t["ids"].shape[1]) if t else None)
    return ris, None


# Le fasi scelte in fila (una sola, oppure tutte le principali), seguite nella timeline; un errore non ferma le
# successive. I risultati si accumulano sullo stesso testo: rieseguire una fase sostituisce solo le sue analisi.
# tutte: l'elenco completo delle fasi, per la posizione delle tacche nella timeline
def esegui_fasi(scelte, tutte):
    s = st.session_state
    da_fare = lavori(scelte)
    for fase, parametri in da_fare:
        s.risultati.pop(chiave(fase, parametri), None)
        s.errori.pop(chiave(fase, parametri), None)
    s.esito_salvataggio = None
    elenco = lavori(tutte)
    s.orari["fasi_inizio"] = datetime.now()
    for i, (fase, parametri) in enumerate(da_fare):
        k = next(j for j, (f, p) in enumerate(elenco) if f.codice == fase.codice and p == parametri)
        # Al più un ridisegno ogni 0,15 s: le fasi chiamano l'avanzamento anche a ogni strato
        ultimo = [0.0]

        def avanzamento(f, m="", k=k, i=i, ultimo=ultimo):
            if time.time() - ultimo[0] >= 0.15:
                ultimo[0] = time.time()
                disegna_timeline(("fasi", k, min(max(f, 0.0), 1.0), m, i, len(da_fare)))

        avanzamento(0.0)
        ris, errore = esegui_fase(fase, parametri, avanzamento)
        if ris is not None:
            s.risultati[chiave(fase, parametri)] = ris
        else:
            s.errori[chiave(fase, parametri)] = errore
    s.orari["fasi_fine"] = datetime.now()
    disegna_timeline()


def salva_una(k):
    ris = st.session_state.risultati[k]
    try:
        ris.info["file"] = salva_excel(ris, CARTELLA_RISULTATI)
        st.session_state.esito_salvataggio = ([ris.info["file"]], [])
    except OSError as e:
        st.session_state.esito_salvataggio = ([], [f"{os.path.basename(nome_file(ris))}: {e}"])


# Tutte le analisi eseguite in un solo file Excel, nell'ordine di esecuzione
def salva_tutte(fasi):
    risultati, errori = st.session_state.risultati, st.session_state.errori
    analisi = [(f, p, risultati.get(chiave(f, p)), errori.get(chiave(f, p))) for f, p in lavori(fasi)
               if chiave(f, p) in risultati or chiave(f, p) in errori]
    try:
        percorso = salva_excel_unico(analisi, CARTELLA_RISULTATI)
    except OSError as e:
        st.session_state.esito_salvataggio = ([], [f"{nome_file_unico(analisi)}: {e}"])
        return
    for ris in risultati.values():
        ris.info["file"] = percorso
    st.session_state.esito_salvataggio = ([percorso], [])


# Blocchi di presentazione
# fatto: il numero diventa una spunta verde; sopra: etichetta piccola sopra il titolo (es. "Passo 1")
def titolo_passo(segnaposto, numero, titolo, stato, tono, fatto=False, sopra=None):
    sopra = f"<small>{html_sicuro(sopra)}</small>" if sopra else ""
    segnaposto.markdown(f"<div class='passo-testa'><span class='passo-num{' fatto' if fatto else ''}'>"
                        f"{'✓' if fatto else numero}</span><span class='passo-tit'>{sopra}{titolo}</span>"
                        f"<span class='chip {tono}'>{html_sicuro(stato)}</span></div>",
                        unsafe_allow_html=True)


def bottone_ricarica(colonna, bloccato=False):
    colonna.button("↻ Ricarica le fasi", width="stretch", on_click=ricarica_fasi, disabled=bloccato,
                   help="Rilegge i file in fasi/ dopo averne aggiunto o modificato uno. "
                        "I risultati in memoria vengono cancellati.")


def hero(sopra, titolo, sotto):
    mostra_html(f"<div class='hero'><div><div class='sopra'>{html_sicuro(sopra)}</div>"
                f"<div class='titolo'>{html_sicuro(titolo)}</div><div class='sotto'>{sotto}</div></div>"
                f"{svg_hero()}</div>")


def mostra_controlli(controlli):
    for c in controlli:
        dettaglio = f" — <span class='dettaglio'>{html_sicuro(c.dettaglio)}</span>" if c.dettaglio else ""
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


# Tabella per strato in HTML: stessi numeri dell'Excel, legenda nei tooltip delle intestazioni, righe citate in ambra.
# Con colonne=None tutte le colonne del risultato
def tabella(ris, colonne=None):
    colonne = ris.colonne if colonne is None else colonne
    legenda = dict(ris.legenda)
    intestazioni = []
    for j, c in enumerate(colonne):
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
        for c in colonne:
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
        st.caption(f"Cartella: `{info['percorso']}` · commit {info['commit']} · rango di Δ: {info['rank_delta']} · "
                   + " · ".join(f"{nome} {v}" for nome, v in versioni().items()))


VISTE = ["📈 Grafici", "🧮 Tabella per strato", "📝 Testo analizzato"]


# Presentazione di un risultato: contesto, riquadri, selettore della vista e download
def mostra_risultato(ris, id_vista):
    i = ris.info
    voci = [("modello", i["modello"]), ("dispositivo", i["dispositivo"]), ("formato", ris.note_formato),
            ("data", f"{i['data']:%d/%m/%Y %H:%M}"), ("durata", f"{i['durata']:.1f} s")]
    if i.get("n_token"):
        voci.append(("testo", f"{i['n_token']} token · {i['fonte_testo']}"))
    mostra_html("<div class='contesto'>" + "".join(f"<span><i>{e}</i>{html_sicuro(v)}</span>" for e, v in voci)
                + "</div>")

    # Risultati fondamentali in primo piano, misure di contorno in un riquadro compresso
    principali = [x for x in ris.indicatori if not x.secondario]
    secondari = [x for x in ris.indicatori if x.secondario]
    riquadri(principali)
    if secondari:
        with st.expander(f"Misure secondarie ({len(secondari)})", icon="🗂"):
            riquadri(secondari)

    vista = st.segmented_control("Vista", VISTE, default=VISTE[0], key="vista",
                                 label_visibility="collapsed") or VISTE[0]
    if vista == VISTE[0]:
        def disegna(k, graf):
            titolo_con_legenda(graf.titolo, graf.spiegazione, graf.come_leggere)
            with st.container(border=True):
                st.plotly_chart(graf.figura, width="stretch", config=g.CONFIG, theme=None,
                                key=f"grafico_{id_vista}_{k}")
        secondari = [(k, x) for k, x in enumerate(ris.grafici) if x.secondario]
        for k, graf in enumerate(ris.grafici):
            if not graf.secondario:
                disegna(k, graf)
        if secondari:
            with st.expander(f"Grafici secondari ({len(secondari)})", icon="🗂"):
                for k, graf in secondari:
                    disegna(k, graf)
    elif vista == VISTE[1]:
        titolo_con_legenda(ris.titolo, ris.descrizione,
                           "\n".join(f"- **{nome}** — {spiegazione}" for nome, spiegazione in ris.legenda),
                           "ⓘ Legenda colonne")
        principali = [c for c in ris.colonne if not c.secondaria]
        tabella(ris, principali)
        st.caption("Passa il mouse sulle intestazioni sottolineate per la spiegazione della colonna."
                   + (" Le righe in ambra sono quelle citate nella colonna Nota." if ris.evidenzia else ""))
        if len(principali) < len(ris.colonne):
            with st.expander(f"Tabella completa ({len(ris.colonne)} colonne, come nell'Excel)", icon="🗂"):
                tabella(ris)
    else:
        if i.get("testo"):
            st.text_area("Testo analizzato", i["testo"], height=260, disabled=True, label_visibility="collapsed",
                         key=f"testo_{id_vista}")
        else:
            st.caption("Questa fase non usa il testo: legge solo i pesi.")

    c1, c2 = st.columns([1, 2])
    c1.download_button("⬇ Scarica questa fase in Excel", data=crea_excel(ris), file_name=nome_file(ris),
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch", key=f"scarica_{id_vista}")
    if i.get("file"):
        c2.caption(f"Salvato in `{i['file']}`")


# Timeline in alto: Modello → Testo → Fasi. Si ridisegna nel segnaposto `timeline` (creato nel corpo principale)
# anche durante il caricamento e l'esecuzione; in_corso descrive il lavoro in atto:
# ("modello", nome, controlli superati, prossimo controllo) oppure ("fasi", indice, frazione, messaggio)
N_CONTROLLI = 6
TONI = {"attesa": "attesa", "corso": "codice", "fatto": "ok", "errore": "errore"}


def _durata(secondi):
    return f"{secondi:.0f} s" if secondi < 60 else f"{int(secondi // 60)} min {secondi % 60:.0f} s"


def _ora(nome):
    ora = st.session_state.orari.get(nome)
    return f"{ora:%H:%M}" if ora else ""


def _barra(frazione):
    return f"<div class='tl-barra'><div style='width:{100 * min(max(frazione, 0.0), 1.0):.1f}%'></div></div>"


# Un nodo: righe sono frammenti HTML già sicuri; sopra e sotto vanno prima e dopo le righe (tacche, barra)
def _nodo(numero, titolo, stato, chip, righe, attivo, classe="", sopra="", sotto=""):
    punto = {"fatto": "✓", "errore": "!"}.get(stato, numero)
    corpo = "".join(f"<div>{r}</div>" for r in righe)
    return (f"<div class='tl-nodo {classe} {stato}{' attivo' if attivo else ''}'>"
            f"<div class='tl-testa'><span class='tl-punto'>{punto}</span><span class='tl-tit'>{titolo}</span>"
            f"<span class='chip {TONI[stato]}'>{html_sicuro(chip)}</span></div>"
            f"<div class='tl-corpo'>{sopra}{corpo}{sotto}</div></div>")


# Ogni _nodo_* restituisce (stato, html)
def _nodo_modello(in_corso):
    s = st.session_state
    attivo = s.passo == "modello"
    if in_corso and in_corso[0] == "modello":
        _, nome, superati, prossimo = in_corso
        return "corso", _nodo(1, "Modello", "corso", f"{superati}/{N_CONTROLLI} controlli",
                              [f"<b>{html_sicuro(nome)}</b>", f"{html_sicuro(prossimo)}…"], attivo,
                              sotto=_barra(superati / N_CONTROLLI))
    if s.info:
        return "fatto", _nodo(1, "Modello", "fatto", "pronto",
                              [f"<b>{html_sicuro(s.info['nome'].split('/')[-1])}</b>",
                               f"{html_sicuro(DISPOSITIVI[s.dispositivo])} · FP32 · {_ora('modello')}",
                               f"✓ {len(s.controlli)} controlli superati"], attivo)
    fallito = next((c for c in s.controlli if not c.ok), None)
    if fallito:
        return "errore", _nodo(1, "Modello", "errore", "non caricato",
                               [f"<b>{html_sicuro(fallito.descrizione)}</b>",
                                f"<span title='{html_sicuro(fallito.dettaglio)}'>{html_sicuro(fallito.dettaglio)}</span>"],
                               attivo)
    return "attesa", _nodo(1, "Modello", "attesa", "da caricare",
                           ["Scegli un modello Mamba-1", "e premi <b>Carica e verifica</b>"], attivo)


def _nodo_testo():
    s, t = st.session_state, st.session_state.testo
    attivo = s.passo == "testo"
    if t:
        n = int(t["ids"].shape[1])
        troncato = f" su {fmt_int(t['n_originali'])}" if t["n_originali"] > n else ""
        return "fatto", _nodo(2, "Testo", "fatto", "pronto",
                              [f"<b>{fmt_int(n)} token</b>{troncato}",
                               f"{html_sicuro(t['fonte'])} · {_ora('testo')}"], attivo)
    return "attesa", _nodo(2, "Testo", "attesa", "da scegliere", ["Testo di prova, da un corpus,",
                                                                   "scritto o da file .txt"], attivo)


# Una tacca per analisi, raggruppate per fase; il title dice esito e durata
def _tacche(elenco, corrente):
    s = st.session_state
    gruppi = {}
    for k, (fase, parametri) in enumerate(elenco):
        c, nome = chiave(fase, parametri), descrivi_lavoro(fase, parametri)
        if c in s.risultati:
            stato, nota = "ok", f"{nome} — ✓ {s.risultati[c].info['durata']:.1f} s"
        elif c in s.errori:
            stato, nota = "errore", f"{nome} — non riuscita: {s.errori[c][0]}"
        elif k == corrente:
            stato, nota = "corso", f"{nome} — in corso"
        else:
            stato, nota = "attesa", nome
        gruppi.setdefault(fase.codice, []).append(f"<b class='tl-tacca {stato}' title='{html_sicuro(nota)}'></b>")
    return "<div class='tl-tacche'>" + "".join(
        f"<span class='tl-gruppo'><i>{html_sicuro(codice)}</i>{''.join(t)}</span>" for codice, t in gruppi.items()
    ) + "</div>"


def _nodo_fasi(fasi, in_corso):
    s = st.session_state
    elenco = lavori(fasi)
    n, n_ok, n_err = len(elenco), len(s.risultati), len(s.errori)
    attivo = s.passo == "risultati"
    if in_corso and in_corso[0] == "fasi":
        _, k, frazione, messaggio, i, n_lotto = in_corso
        trascorsi = (datetime.now() - s.orari["fasi_inizio"]).total_seconds()
        righe = [f"<b>{html_sicuro(descrivi_lavoro(*elenco[k]))}</b>"
                 + (f": {html_sicuro(messaggio)}" if messaggio else ""),
                 f"questa analisi {100 * frazione:.0f} % · {_durata(trascorsi)} dall'avvio"]
        return "corso", _nodo(3, "Fasi", "corso", f"{i + 1} di {n_lotto}", righe, attivo, "fasi",
                              _tacche(elenco, k), _barra((i + frazione) / n_lotto))
    if n_ok or n_err:
        stato = "errore" if n_err else "fatto"
        inizio, fine = s.orari.get("fasi_inizio"), s.orari.get("fasi_fine")
        durata = f" in {_durata((fine - inizio).total_seconds())} · {_ora('fasi_fine')}" if inizio and fine else ""
        righe = [f"<b>{n_ok} di {n} analisi completate</b>{durata}"]
        if n_err:
            righe.append(f"{n_err} non riuscite: passa il mouse sulle tacche rosse")
        return stato, _nodo(3, "Fasi", stato, f"{n_ok} di {n}", righe, attivo, "fasi", _tacche(elenco, None))
    return "attesa", _nodo(3, "Fasi", "attesa", f"{n} analisi", ["Si scelgono dopo il testo, una alla volta o tutte"],
                           attivo, "fasi", _tacche(elenco, None))


def html_timeline(fasi, in_corso=None):
    nodi = [_nodo_modello(in_corso), _nodo_testo(), _nodo_fasi(fasi, in_corso)]
    parti = [f"<div class='tl-marchio'>{svg_marchio()}<div><b>Mamba<span>·</span>Modi</b>"
             f"<small>analisi di poli e stabilità</small></div></div>"]
    for k, (stato, codice) in enumerate(nodi):
        if k:
            # La linea verso il nodo successivo diventa verde quando il nodo precedente è fatto
            parti.append(f"<div class='tl-linea{' fatta' if nodi[k - 1][0] == 'fatto' else ''}'></div>")
        parti.append(codice)
    return f"<div class='tl'>{''.join(parti)}</div>"


# Usa le variabili del corpo principale `timeline` (segnaposto) e `fasi`
def disegna_timeline(in_corso=None):
    timeline.markdown(html_timeline(fasi, in_corso), unsafe_allow_html=True)


# Passo ①: scelta e caricamento del modello
def pagina_modello(fasi):
    hero("Analisi spettrale di modelli state-space", "Analisi dei modi di Mamba",
         "Misura poli, passi di campionamento e margini di stabilità di un modello Mamba-1, strato per strato. "
         "Tutto gira in locale, senza connessione.")
    capacita = "".join(f"<span>{html_sicuro(f.nome)}</span>" for f in fasi) + "<span>Esportazione in Excel</span>"
    mostra_html(f"<div class='capacita'>{capacita}</div>")

    with st.container(key="passo_modello"):
        titolo_passo(st.empty(), 1, "Carica il modello", "pronto" if st.session_state.info else "da caricare",
                     "ok" if st.session_state.info else "attesa")
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        c1.text_input("Cartella dei modelli", key="cartella",
                      help="Predefinita: la cache di Hugging Face, dove from_pretrained salva i modelli "
                           "scaricati. Va bene anche una cartella con config.json e i pesi.")
        c2.button("Sfoglia…", on_click=sfoglia_cartella, width="stretch")

        trovati = trova_modelli(st.session_state.cartella)
        solo_mamba = st.toggle("Mostra solo modelli Mamba-1", value=True)
        elenco = [m for m in trovati if m.is_mamba] if solo_mamba else trovati
        if not trovati:
            st.caption("Nessun modello trovato in questa cartella.")
        elif not elenco:
            st.caption(f"{len(trovati)} modelli trovati, nessuno Mamba-1. Disattiva il filtro per vederli.")

        scelto = None
        c1, c2 = st.columns([3, 2])
        if elenco:
            scelto = c1.selectbox("Modello", elenco,
                                  format_func=lambda m: m.nome if m.is_mamba else f"{m.nome} [{m.tipo}]")
            c1.caption(f"Pesi: {scelto.dimensione_gb:.2f} GB su disco · commit {scelto.commit}")
        disp = c2.radio("Esegui su", list(DISPOSITIVI), format_func=DISPOSITIVI.get, horizontal=True,
                        index=list(DISPOSITIVI).index(st.session_state.dispositivo))
        if scelto is not None and disp == "cpu" and scelto.dimensione_gb > 2:
            c2.caption(f"⚠ Modello grande su CPU: servono circa {scelto.dimensione_gb * 1.3:.1f} GB di RAM liberi.")

        info = st.session_state.info
        c1, c2 = st.columns(2)
        if c1.button("Carica e verifica", type="primary" if info is None else "secondary",
                     disabled=scelto is None, width="stretch"):
            carica_modello(scelto, disp)
            if st.session_state.model is not None:
                vai("testo")
                st.rerun()
        if info is not None:
            c2.button(f"Continua con {info['nome'].split('/')[-1]} →", type="primary", width="stretch",
                      on_click=vai, args=("testo",))

    if st.session_state.controlli and st.session_state.model is None:
        st.subheader("Esito dell'ultima verifica")
        mostra_controlli(st.session_state.controlli)
        st.error("Il modello non è stato caricato: vedi il controllo non superato qui sopra.")


# Token di ogni articolo di un corpus, contati una volta per corpus e tokenizer e tenuti per la sessione
def token_corpus(corpus):
    cache = st.session_state.setdefault("token_corpora", {})
    k = (corpus["percorso"], os.path.getmtime(corpus["percorso"]), st.session_state.info["nome"])
    if k not in cache:
        with st.spinner(f"Conto i token di {len(corpus['articoli'])} articoli…"):
            cache[k] = corpora.conta_token(st.session_state.tok, corpus)
    return cache[k]


# I pulsanti Esegui non lanciano le fasi direttamente: registrano la richiesta e la pagina si ridisegna con tutti i
# comandi bloccati; solo dopo parte l'esecuzione. Così, mentre una fase gira, non si può premere altro (un clic durante
# l'esecuzione interromperebbe lo script di Streamlit a metà)
def richiedi_fasi(codici):
    st.session_state.richiesta_fasi = list(codici)


def estrai_altro(chiave_articolo, idonei):
    st.session_state[chiave_articolo] = corpora.estrai(idonei, st.session_state.get(chiave_articolo))


# Testo estratto da un corpus scaricato con scarica_testi.py. Sono proposti solo gli articoli lunghi almeno
# max_token, così ogni testo estratto ha esattamente max_token token e i risultati sono confrontabili fra articoli.
# Restituisce (testo, nome della fonte, dettagli per l'Excel) oppure (None, None, None)
def pannello_corpus(max_token, bloccato=False):
    elenco = corpora.elenco_corpora()
    if not elenco:
        st.info("Nessun corpus scaricato. Una volta sola, con la connessione, dalla cartella `analisi-mamba` con "
                "l'ambiente attivo: `python scarica_testi.py`. Scarica in `testi/corpora/` WikiText-2 (test) e il mese "
                "più recente di notizie BBC (pochi MB in tutto); poi l'app li legge senza rete.", icon="📥")
        return None, None, None

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    per_percorso = {c["percorso"]: c for c in elenco}
    corpus = per_percorso[c1.selectbox("Corpus", list(per_percorso), key="corpus", disabled=bloccato,
                                       format_func=lambda p: per_percorso[p]["nome"])]
    c2.markdown(f"<span class='visto {'mai' if corpus.get('mai_visto') else 'forse'}'>"
                f"{'✓ mai visto dal modello' if corpus.get('mai_visto') else '! forse visto in addestramento'}</span>",
                unsafe_allow_html=True)
    st.caption(f"{corpus['breve']} Nei dati di addestramento di Mamba: {corpus['addestramento']}. "
               f"Fonte: {corpus['citazione']}, `{corpus['dataset']}`.")

    n_token = token_corpus(corpus)
    articoli = corpus["articoli"]
    idonei = [k for k, n in enumerate(n_token) if n >= max_token]
    if not idonei:
        st.warning(f"Nessun articolo di questo corpus ha almeno {max_token} token: abbassa i token massimi.")
        return None, None, None

    chiave_articolo = f"articolo_{corpus['id']}"
    if st.session_state.get(chiave_articolo) not in idonei:
        st.session_state[chiave_articolo] = corpora.estrai(idonei)
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    c1.selectbox(f"Articolo ({len(idonei)} su {len(articoli)} hanno almeno {max_token} token)", idonei,
                 key=chiave_articolo, disabled=bloccato,
                 format_func=lambda k: f"{articoli[k]['titolo']}" + (f" · {articoli[k]['data']}" if articoli[k]["data"]
                                                                    else "") + f" · {n_token[k]} token")
    c2.button("🎲 Estrai un altro", width="stretch", on_click=estrai_altro, args=(chiave_articolo, idonei),
              disabled=bloccato,
              help="Sceglie a caso un altro articolo fra quelli abbastanza lunghi.")

    k = st.session_state[chiave_articolo]
    a = articoli[k]
    meta = [("articolo", f"n. {k + 1} di {len(articoli)}"), ("lunghezza", f"{fmt_int(n_token[k])} token")]
    if a["data"]:
        meta.append(("pubblicato", a["data"]))
    link = (f"<span><i>fonte</i><a href='{html_sicuro(a['link'])}' target='_blank'>apri l'originale ↗</a></span>"
            if a["link"].startswith("http") else "")
    mostra_html(f"<div class='articolo'><div class='tit'>{html_sicuro(a['titolo'])}</div><div class='meta'>"
                + "".join(f"<span><i>{e}</i>{html_sicuro(v)}</span>" for e, v in meta) + link + "</div></div>")
    st.text_area("Anteprima dell'articolo", a["testo"], height=230, disabled=True, label_visibility="collapsed",
                 key=f"anteprima_{corpus['id']}_{k}")
    fonte = f"{corpus['nome']} · n. {k + 1} «{a['titolo']}»"
    return a["testo"], fonte, corpora.dettagli(corpus, k, n_token[k])


# Passo ②: scelta del testo e delle fasi da eseguire (una alla volta o tutte)
def pagina_testo(fasi):
    # Richiesta di esecuzione dal clic precedente: in questo passaggio tutti i comandi sono disegnati bloccati e,
    # alla fine della pagina, partono le fasi richieste
    richiesta = st.session_state.pop("richiesta_fasi", None)
    bloccato = richiesta is not None
    intestazione_modello(st.session_state.info)
    with st.container(key="passo_testo"):
        titolo_passo(st.empty(), 2, "Scegli il testo", "da scegliere", "attesa")
        fonte = st.radio("Fonte", FONTI, horizontal=True, key="fonte", label_visibility="collapsed", disabled=bloccato)
        # Il limite serve già alla scelta degli articoli del corpus: si legge dallo stato del widget, disegnato più sotto
        max_token = int(st.session_state.get("max_token", MAX_TOKEN_PREDEFINITO))
        grezzo = caricato = nome_corpus = None
        dettagli = []
        if fonte == FONTI[0]:
            st.caption("Breve testo in inglese sulla teoria del controllo: un riferimento fisso per confrontare "
                       "esecuzioni diverse.")
            grezzo = TESTO_ESEMPIO
            st.text_area("Testo di prova", TESTO_ESEMPIO, height=230, disabled=True, label_visibility="collapsed")
        elif fonte == FONTI[1]:
            grezzo, nome_corpus, dettagli = pannello_corpus(max_token, bloccato)
        elif fonte == FONTI[2]:
            grezzo = st.text_area("Testo", height=230, key="testo_incollato", label_visibility="collapsed",
                                  disabled=bloccato,
                                  placeholder="Scrivi o incolla qui il testo (in inglese, come i dati di addestramento "
                                              "di Mamba).")
        else:
            caricato = st.file_uploader("File di testo", type=["txt", "md"], label_visibility="collapsed",
                                        disabled=bloccato)
            if caricato is not None:
                grezzo = caricato.getvalue().decode("utf-8", errors="replace")
                st.text_area("Contenuto del file", grezzo, height=230, disabled=True, label_visibility="collapsed",
                             key=f"anteprima_{caricato.name}_{caricato.size}")
        max_token = st.number_input("Token massimi", 16, 8192, MAX_TOKEN_PREDEFINITO, step=64, key="max_token",
                                    disabled=bloccato,
                                    help=f"I testi più lunghi vengono troncati. Mamba è addestrato su sequenze di "
                                         f"{corpora.LUNGHEZZA_ADDESTRAMENTO} token: oltre, il modello lavora fuori dal "
                                         "regime di addestramento. Dal corpus sono proposti solo articoli lunghi "
                                         "almeno quanto questo valore. Su una GPU da 8 GB con il 1.4B, se la memoria "
                                         "non basta conviene scendere a 1024.")
        if max_token > corpora.LUNGHEZZA_ADDESTRAMENTO:
            st.caption(f"⚠ Oltre {corpora.LUNGHEZZA_ADDESTRAMENTO} token, la lunghezza di addestramento di Mamba.")
        pronto = bool(grezzo and grezzo.strip())
        if pronto:
            n = len(st.session_state.tok(grezzo.strip()).input_ids)
            st.caption(f"{n} token" + (f": verranno analizzati i primi {int(max_token)}" if n > max_token else ""))

        nome_fonte = {FONTI[0]: "testo di prova", FONTI[1]: nome_corpus, FONTI[2]: "testo scritto o incollato"}.get(
            fonte, f"file {getattr(caricato, 'name', '')}")

    # Scelta delle fasi: una scheda per fase con il suo pulsante, più "Esegui tutte le fasi" (solo le principali).
    # I risultati restano in memoria finché il testo non cambia: si possono eseguire le fasi una alla volta
    s = st.session_state
    t = s.testo
    stesso_testo = bool(pronto and t and t["grezzo"] == grezzo.strip() and t["max_token"] == int(max_token))
    in_memoria = len(s.risultati) + len(s.errori)
    da_eseguire = [f for f in fasi if f.codice in richiesta] if bloccato and pronto else None
    with st.container(key="passo_fasi"):
        titolo_passo(st.empty(), 3, "Scegli le fasi",
                     "esecuzione in corso" if da_eseguire else
                     f"{len(s.risultati)} analisi in memoria" if stesso_testo and in_memoria else "da eseguire",
                     "codice" if da_eseguire else "ok" if stesso_testo and s.risultati else "attesa")
        if da_eseguire:
            st.info(f"In esecuzione: {', '.join(f'Fase {f.codice}' for f in da_eseguire)} "
                    f"({len(lavori(da_eseguire))} analisi). I comandi restano bloccati fino alla fine; l'avanzamento è "
                    "nella timeline in alto.", icon="⏳")
        if in_memoria and not stesso_testo:
            st.caption(f"⚠ Il testo è cambiato: eseguendo una fase le {in_memoria} analisi in memoria vengono "
                       "cancellate (salvale prima dalla pagina dei risultati).")
        gruppi = [("Fasi", principali(fasi)), ("Fasi extra · non fanno parte di Esegui tutte le fasi", extra(fasi))]
        for titolo, elenco_fasi in gruppi:
            if not elenco_fasi:
                continue
            mostra_html(f"<div class='gruppo-fasi'>{html_sicuro(titolo)}</div>")
            for fase in elenco_fasi:
                scheda_fase(fase, stesso_testo, pronto and not bloccato)

        elenco = lavori(principali(fasi))
        st.caption(f"*Esegui tutte le fasi* lancia {len(elenco)} analisi: "
                   + ", ".join(descrivi_lavoro(f, p).replace("Fase ", "") for f, p in elenco)
                   + ". Le fasi extra si eseguono solo dalla loro scheda.")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        c1.button("← Cambia modello", width="stretch", on_click=vai, args=("modello",), disabled=bloccato)
        bottone_ricarica(c2, bloccato)
        c3.button("Vai ai risultati →", width="stretch", on_click=vai, args=("risultati",),
                  disabled=bloccato or not (stesso_testo and in_memoria))
        c4.button("Esegui tutte le fasi →", type="primary", disabled=bloccato or not pronto, width="stretch",
                  on_click=richiedi_fasi, args=([f.codice for f in principali(fasi)],))

    if da_eseguire:
        if not stesso_testo:
            svuota_risultati()
            imposta_testo(grezzo.strip(), nome_fonte, int(max_token), dettagli)
        esegui_fasi(da_eseguire, fasi)
        s["fase_scelta"] = da_eseguire[0]
        vai("risultati")
        st.rerun()


# Scheda di una fase nella pagina del testo: nome, spiegazione breve, analisi che lancia, esito se già eseguita sul
# testo attuale, spiegazione completa a richiesta e pulsante (attivo solo se attivo=True)
def scheda_fase(fase, stesso_testo, attivo):
    s = st.session_state
    elenco = lavori([fase])
    fatte = [c for c in (chiave(f, p) for f, p in elenco) if c in s.risultati] if stesso_testo else []
    errate = [c for c in (chiave(f, p) for f, p in elenco) if c in s.errori] if stesso_testo else []
    if errate:
        chip = f"<span class='chip errore'>{len(errate)} non riuscite</span>"
    elif fatte:
        chip = "<span class='chip ok'>✓ eseguita</span>"
    else:
        chip = ""
    with st.container(border=True, key=f"scheda_fase_{fase.codice}"):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            mostra_html(f"<div class='scheda-fase'><div class='testa'><span class='codice'>FASE "
                        f"{html_sicuro(fase.codice)}</span><span class='nome'>{html_sicuro(fase.nome)}</span>{chip}</div>"
                        f"<div class='breve'>{html_sicuro(fase.breve)}</div>"
                        f"<div class='analisi'>{len(elenco)} analisi: "
                        f"{html_sicuro(', '.join(' · '.join(map(str, p.values())) or 'unica' for _, p in elenco))}"
                        f"{'' if fase.usa_testo else ' · usa solo i pesi'}</div></div>")
            with st.expander("Cosa misura"):
                st.markdown(fase.spiegazione)
        c2.button("Riesegui" if fatte or errate else "Esegui", key=f"esegui_fase_{fase.codice}",
                  disabled=not attivo, width="stretch", on_click=richiedi_fasi, args=([fase.codice],))


# Passo ③: tutte le fasi già eseguite, sfogliabili da un menu; salvataggio su disco solo a richiesta
def pagina_risultati(fasi):
    info, t = st.session_state.info, st.session_state.testo
    risultati, errori = st.session_state.risultati, st.session_state.errori
    eseguite = [f for f in fasi if any(k[0] == f.codice for k in list(risultati) + list(errori))]
    hero("Risultati", info["nome"],
         f"{html_sicuro(DISPOSITIVI[st.session_state.dispositivo])} · {t['ids'].shape[1]} token · "
         f"{html_sicuro(t['fonte'])} · {len(risultati)} analisi completate"
         + (f", {len(errori)} non riuscite" if errori else ""))
    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    c1.button("← Testo e fasi", width="stretch", on_click=vai, args=("testo",),
              help="Torna alla scelta delle fasi: con lo stesso testo i risultati restano in memoria e si aggiungono.")
    c2.button("← Cambia modello", width="stretch", on_click=vai, args=("modello",))
    bottone_ricarica(c3)
    if not eseguite:
        st.warning("Nessuna fase eseguita: torna a *Testo e fasi* e scegli una fase o *Esegui tutte le fasi*.")
        return

    colonne = st.columns([3] + [1] * max(len(p.parametri) for p in eseguite))
    fase = colonne[0].selectbox("Fase", eseguite, key="fase_scelta",
                                format_func=lambda f: f"Fase {f.codice} · {f.nome}")
    parametri = {}
    for col, p in zip(colonne[1:], fase.parametri):
        parametri[p.chiave] = col.selectbox(p.etichetta, p.scelte, key=f"param_{p.chiave}", help=p.aiuto or None,
                                            index=p.scelte.index(p.predefinito) if p.predefinito in p.scelte else 0)

    mostra_html(f"<div class='testa-fase'><span class='codice'>FASE {html_sicuro(fase.codice)}</span>"
                f"<span class='nome'>{html_sicuro(fase.nome)}</span></div>"
                f"<div class='breve-fase'>{html_sicuro(fase.breve)}</div>")
    with st.expander("Cosa misura questa fase", icon="📘"):
        st.markdown(fase.spiegazione)

    k = chiave(fase, parametri)
    ris = risultati.get(k)
    if ris is not None:
        mostra_risultato(ris, "_".join([fase.codice] + [str(v) for v in parametri.values()]))
    elif k in errori:
        messaggio, dettaglio = errori[k]
        st.error(f"{descrivi_lavoro(fase, parametri)} non è riuscita: {messaggio}")
        if dettaglio:
            with st.expander("Dettagli"):
                st.code(dettaglio)

    st.divider()
    with st.container(key="passo_salvataggio"):
        da_salvare = [c for c, r in risultati.items() if not r.info.get("file")]
        titolo_passo(st.empty(), "↓", "Salvataggio in risultati/",
                     "tutto salvato" if risultati and not da_salvare else f"{len(da_salvare)} da salvare",
                     "ok" if risultati and not da_salvare else "attesa")
        st.caption(f"I risultati restano in memoria finché l'app è aperta. Per conservarli si salvano in Excel "
                   f"nella cartella `{CARTELLA_RISULTATI}`.")
        c1, c2 = st.columns(2)
        c1.button(f"Salva {descrivi_lavoro(fase, parametri)}", width="stretch", disabled=ris is None,
                  on_click=salva_una, args=(k,))
        c2.button(f"Salva tutto in un file Excel ({len(risultati)} analisi)", type="primary", width="stretch",
                  disabled=not risultati, on_click=salva_tutte, args=(fasi,),
                  help="Un solo foglio: informazioni comuni, indice e, per ogni analisi, sintesi e tabella per "
                       "strato. I dati dei grafici restano nell'Excel della singola fase.")
        esito = st.session_state.esito_salvataggio
        if esito:
            salvati, problemi = esito
            if salvati:
                st.success(f"Salvato in `{salvati[0]}`.")
            for p in problemi:
                st.warning(f"Salvataggio non riuscito: {p}")


fasi, errori_fasi = carica_fasi()
if st.session_state.model is None:
    st.session_state.passo = "modello"
elif st.session_state.passo == "risultati" and st.session_state.testo is None:
    st.session_state.passo = "testo"
timeline = st.container(key="timeline").empty()
disegna_timeline()

for nome, errore in errori_fasi:
    st.error(f"Errore nel file fasi/{nome}.py: la fase non compare. Correggi il file e premi *Ricarica le fasi*.",
             icon="⚠️")
    with st.expander("Dettagli"):
        st.code(errore)

if st.session_state.messaggio:
    tipo, testo_msg = st.session_state.messaggio
    getattr(st, tipo)(testo_msg)
    st.session_state.messaggio = None

{"modello": pagina_modello, "testo": pagina_testo, "risultati": pagina_risultati}[st.session_state.passo](fasi)
mostra_html("<div class='piede'>Elaborazione locale · nessuna connessione di rete</div>")
