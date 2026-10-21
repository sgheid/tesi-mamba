import io
import math
import os
import re
from datetime import datetime

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .formato import FORMATI

BLU_SCURO = "1F3864"
SFONDO_INTESTAZIONE = PatternFill("solid", fgColor=BLU_SCURO)
SFONDO_EVIDENZA = PatternFill("solid", fgColor="FFF2CC")
FONT_INTESTAZIONE = Font(bold=True, color="FFFFFF")
FONT_SEZIONE = Font(bold=True, size=12, color=BLU_SCURO)
BORDO = Border(bottom=Side(style="thin", color="BFBFBF"))
SINISTRA = Alignment(horizontal="left", vertical="top", wrap_text=True)


def nome_file(ris):
    data = ris.info.get("data", datetime.now())
    nome = ris.info.get("modello", "modello").split("/")[-1].split(" @ ")[0]
    # I parametri nel nome (es. _FP16): esecuzioni della stessa fase nello stesso secondo non si sovrascrivono
    parametri = "".join("_" + re.sub(r"\W+", "-", str(v)).strip("-") for v in ris.info.get("parametri", {}).values())
    return f"{data:%Y%m%d_%H%M%S}_{nome}_Fase{ris.codice}{parametri}.xlsx"


def _nome_foglio(testo, usati):
    base = re.sub(r"[\\/*?:\[\]]", "", testo)[:28] or "Dati"
    nome, k = base, 2
    while nome in usati:
        nome, k = f"{base[:25]} {k}", k + 1
    usati.add(nome)
    return nome


def _valore_cella(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v.item() if hasattr(v, "item") else v


def _intestazione(cella):
    cella.font = FONT_INTESTAZIONE
    cella.fill = SFONDO_INTESTAZIONE


# Excel di un Risultato: foglio della fase (info, sintesi, tabella, legenda) + un foglio per grafico
def crea_excel(ris):
    wb = openpyxl.Workbook()
    ws = wb.active
    usati = set()
    ws.title = _nome_foglio(f"Fase {ris.codice}", usati)

    col = ris.colonne
    n_col = len(col)
    ultima = max(n_col + 1, 7)
    info = ris.info
    riga = 1

    def sezione(titolo):
        nonlocal riga
        ws.cell(riga, 1, titolo).font = FONT_SEZIONE
        riga += 1

    def coppia(etichetta, valore):
        nonlocal riga
        ws.merge_cells(start_row=riga, start_column=1, end_row=riga, end_column=2)
        ws.merge_cells(start_row=riga, start_column=3, end_row=riga, end_column=ultima)
        e = ws.cell(riga, 1, etichetta)
        e.font = Font(bold=True)
        e.alignment = SINISTRA
        ws.cell(riga, 3, valore).alignment = SINISTRA
        riga += 1

    ws.cell(1, 1, ris.titolo).font = Font(bold=True, size=14)
    ws.cell(2, 1, ris.descrizione).font = Font(italic=True)
    riga = 4

    sezione("INFORMAZIONI")
    elenco = [
        ("Modello", info.get("modello", "")),
        ("Cartella", info.get("percorso", "")),
        ("Commit dei pesi", info.get("commit", "")),
        ("Struttura", info.get("struttura", "")),
        ("Dispositivo", info.get("dispositivo", "")),
        ("Formato numerico", ris.note_formato),
        ("Data e ora", f"{info.get('data', datetime.now()):%d/%m/%Y %H:%M:%S}"),
        ("Durata", f"{info.get('durata', 0):.1f} s"),
    ]
    elenco += [(f"Parametro: {k}", str(v)) for k, v in info.get("parametri", {}).items()]
    testo = info.get("testo")
    if testo is not None:
        elenco += [("Fonte del testo", info.get("fonte_testo", "")),
                   ("Token analizzati", str(info.get("n_token", "")))]
    for etichetta, valore in elenco:
        coppia(etichetta, valore)
    if testo is not None:
        caratteri_per_riga = 23 * (ultima - 2)
        n_righe = sum(len(p) // caratteri_per_riga + 1 for p in testo.split("\n"))
        ws.row_dimensions[riga].height = min(409, 15 * (n_righe + 1))
        coppia("Testo analizzato", testo[:32000])
    riga += 1

    sezione("SINTESI")
    for ind in ris.indicatori:
        coppia(ind.etichetta, ind.valore + (f"  -  {ind.nota}" if ind.nota else ""))
    riga += 1

    # Tabella per strato, con le righe evidenziate e la nota a destra
    sezione("RISULTATI PER STRATO")
    for j, c in enumerate(col, start=1):
        cella = ws.cell(riga, j, f"{c.titolo}\n{c.sottotitolo}".strip())
        _intestazione(cella)
        cella.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.row_dimensions[riga].height = 32
    riga_intestazione = riga
    riga += 1
    for r in ris.righe:
        evid = r.get("strato") in ris.evidenzia
        for j, c in enumerate(col, start=1):
            cella = ws.cell(riga, j, _valore_cella(r.get(c.chiave)))
            cella.number_format = FORMATI[c.formato][1]
            cella.border = BORDO
            if c.chiave == "strato":
                cella.alignment = Alignment(horizontal="center")
            if evid:
                cella.fill = SFONDO_EVIDENZA
        if evid:
            nota = ws.cell(riga, n_col + 1, ris.evidenzia[r["strato"]])
            nota.font = Font(italic=True)
            nota.alignment = Alignment(indent=1)
        riga += 1
    riga += 1

    sezione("COME LEGGERE LE COLONNE")
    for nome_col, spiegazione in ris.legenda:
        coppia(nome_col, spiegazione)

    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 30
    for j in range(3, ultima + 1):
        ws.column_dimensions[get_column_letter(j)].width = 20
    ws.column_dimensions[get_column_letter(n_col + 1)].width = 26
    ws.freeze_panes = ws.cell(riga_intestazione + 1, 1)
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0

    # Un foglio con i dati di ogni grafico
    for g in ris.grafici:
        if g.dati is None or len(g.dati) == 0:
            continue
        wg = wb.create_sheet(_nome_foglio(g.titolo, usati))
        wg.cell(1, 1, g.titolo).font = Font(bold=True, size=12)
        if g.spiegazione:
            wg.cell(2, 1, g.spiegazione).font = Font(italic=True)
        for j, nome in enumerate(g.dati.columns, start=1):
            _intestazione(wg.cell(4, j, str(nome)))
            wg.column_dimensions[get_column_letter(j)].width = max(14, len(str(nome)) + 2)
        for i, riga_dati in enumerate(g.dati.itertuples(index=False), start=5):
            for j, v in enumerate(riga_dati, start=1):
                wg.cell(i, j, _valore_cella(v))
        wg.freeze_panes = "A5"

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def salva_excel(ris, cartella="risultati"):
    os.makedirs(cartella, exist_ok=True)
    percorso = os.path.join(cartella, nome_file(ris))
    with open(percorso, "wb") as f:
        f.write(crea_excel(ris))
    return percorso
