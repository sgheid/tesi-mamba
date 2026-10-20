"""
tesi_mamba.py - Analisi interattiva dei modi di Mamba (Fasi 1, 2a, 2b).

Uso (dalla cmd, con l'ambiente attivo):
    cd tesi-mamba
    .venv\\Scripts\\activate
    pip install openpyxl          (solo la prima volta: serve per il file Excel)
    python tesi_mamba.py

Fasi (tutte in FP32, la baseline della tesi):
    Fase 1  - poli continui a = -exp(A_log), letti dai pesi (non usa il testo)
    Fase 2a - passo di campionamento Delta, prodotto dal modello sul testo
    Fase 2b - poli discreti a_bar = exp(Delta * a), in FP32 con riferimento FP64

Ogni fase stampa una tabella e salva UN file Excel nella cartella "risultati"
con: informazioni sulla misura, testo analizzato, sintesi, tabella e legenda.
"""
import gc
import math
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MambaForCausalLM

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    EXCEL_DISPONIBILE = True
except ImportError:
    EXCEL_DISPONIBILE = False

# ---------------------------------------------------------------------------
# Impostazioni
# ---------------------------------------------------------------------------
MODELLI = {
    "1": "state-spaces/mamba-130m-hf",
    "2": "state-spaces/mamba-1.4b-hf",
}
MAX_TOKEN = 1024            # testi piu' lunghi vengono troncati (limite per la VRAM da 8 GB)
CARTELLA_RISULTATI = "risultati"
PAROLA_FINE = "FINE"        # riga che chiude il testo incollato

TESTO_ESEMPIO = (
    "Control theory studies how dynamical systems evolve over time and how their "
    "behaviour can be shaped by feedback. A linear system is described by a state "
    "vector that summarises the past, a matrix that governs how this state evolves, "
    "and an input that drives it. The eigenvalues of the state matrix decide whether "
    "the system forgets its initial condition, oscillates, or diverges. In discrete "
    "time, a system is stable when every eigenvalue lies strictly inside the unit "
    "circle; an eigenvalue close to one corresponds to a slow mode with a long memory, "
    "while an eigenvalue close to zero corresponds to a mode that forgets almost "
    "immediately. Engineers have used these ideas for decades to design autopilots, "
    "chemical plants and power grids. Recently, the same mathematics has appeared in "
    "language models, where a recurrent state must carry information across thousands "
    "of words. Whether such a model remembers the beginning of a long document depends, "
    "once again, on how close its eigenvalues are to the boundary of stability, and on "
    "whether the hardware can represent numbers that are almost, but not exactly, one."
)

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
NOME_DISPOSITIVO = (f"GPU {torch.cuda.get_device_name(0)}" if DISPOSITIVO == "cuda" else "CPU")


# ---------------------------------------------------------------------------
# Formattazione dei numeri
# ---------------------------------------------------------------------------
def fmt_sci(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if math.isinf(x):
        return "infinito"
    return f"{x:.2e}"


def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")        # 495310 -> "495 310"


def fmt_perc(p):
    """p e' una frazione: 0.0229 -> "2.29 %", 0.0000016 -> "0.00016 %" (3 cifre significative)."""
    if p == 0:
        return "0 %"
    x = p * 100
    decimali = max(0, 2 - math.floor(math.log10(x)))
    return f"{x:.{decimali}f} %"


def fmt_dec(x):
    return f"{x:.1f}"


FORMATI = {   # formato a schermo, formato Excel
    "sci": (fmt_sci, "0.00E+00"),
    "int": (fmt_int, "#,##0"),
    "perc": (fmt_perc, "0.00000%"),
    "dec": (fmt_dec, "0.0"),
}


# ---------------------------------------------------------------------------
# Interazione
# ---------------------------------------------------------------------------
def chiedi(messaggio):
    try:
        return input(messaggio).strip()
    except EOFError:
        print("\nFine dell'input. Chiusura.")
        sys.exit(0)


def scegli(titolo, opzioni):
    print(f"\n{titolo}")
    for chiave, testo in opzioni:
        print(f"  {chiave}) {testo}")
    valide = {chiave for chiave, _ in opzioni}
    while True:
        risposta = chiedi("> ")
        if risposta in valide:
            return risposta
        print(f"Scelta non valida. Scrivi uno tra: {', '.join(sorted(valide))}")


def nome_breve(modello):
    return modello.split("/")[-1]


# ---------------------------------------------------------------------------
# Modello e testo
# ---------------------------------------------------------------------------
def scegli_modello():
    scelta = scegli("Quale modello vuoi esaminare?", [
        ("1", MODELLI["1"] + "   (24 strati, ~0.5 GB in FP32)"),
        ("2", MODELLI["2"] + "   (48 strati, ~5.6 GB in FP32)"),
        ("3", "altro: inserisco io il nome (es. state-spaces/mamba-370m-hf)"),
    ])
    if scelta == "3":
        while True:
            nome = chiedi("Nome del modello su Hugging Face: ")
            if nome:
                return nome
    return MODELLI[scelta]


def commit_modello(model, nome):
    commit = getattr(model.config, "_commit_hash", None)
    if commit:
        return str(commit)[:10]
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == nome:
                return ", ".join(sorted(r.commit_hash[:10] for r in repo.revisions))
    except Exception:
        pass
    return "n/d"


def carica_modello(nome):
    if DISPOSITIVO == "cpu" and "130m" not in nome:
        print("\nATTENZIONE: nessuna GPU disponibile. Un modello grande in FP32 richiede")
        print("molta RAM (il 1.4B circa 5.6 GB solo di pesi) e puo' bloccare il PC.")
        if chiedi("Continuare comunque? (s/n) ").lower() != "s":
            return None, None, None

    print(f"\nCaricamento di {nome} in FP32 su {NOME_DISPOSITIVO}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(nome)
    try:
        model = MambaForCausalLM.from_pretrained(nome, dtype=torch.float32, device_map=DISPOSITIVO)
    except Exception:
        model = MambaForCausalLM.from_pretrained(nome).to(DISPOSITIVO)
    model = model.float().eval()

    cfg = model.config
    info = dict(nome=nome, commit=commit_modello(model, nome), strati=cfg.num_hidden_layers,
                canali=cfg.intermediate_size, stati=cfg.state_size)
    print(f"Caricato in {time.time() - t0:.1f} s:  {info['strati']} strati, {info['canali']} canali, "
          f"{info['stati']} stati per canale, commit {info['commit']}")
    return model, tok, info


def libera_modello(model):
    del model
    gc.collect()
    if DISPOSITIVO == "cuda":
        torch.cuda.empty_cache()


def chiedi_testo():
    scelta = scegli("Da dove prendo il testo da far leggere al modello?", [
        ("1", f"lo incollo qui (poi scrivo {PAROLA_FINE} su una riga da sola e premo Invio)"),
        ("2", "lo leggo da un file .txt"),
        ("3", "uso il testo di esempio (quello delle fasi gia' fatte, 211 token)"),
        ("0", "annulla"),
    ])
    if scelta == "0":
        return None, None
    if scelta == "3":
        return TESTO_ESEMPIO, "testo di esempio"
    if scelta == "2":
        while True:
            percorso = chiedi("Percorso del file (es. testo.txt): ").strip('"')
            try:
                with open(percorso, encoding="utf-8", errors="replace") as f:
                    testo = f.read()
                if testo.strip():
                    return testo, f"file {os.path.basename(percorso)}"
                print("Il file e' vuoto.")
            except OSError as e:
                print(f"Impossibile leggere il file: {e}")
            if chiedi("Riprovare? (s/n) ").lower() != "s":
                return None, None

    print(f"\nIncolla il testo. Quando hai finito, scrivi {PAROLA_FINE} su una riga da sola e premi Invio.")
    righe = []
    while True:
        riga = chiedi("")
        if riga == PAROLA_FINE:
            break
        righe.append(riga)
    testo = "\n".join(righe).strip()
    if not testo:
        print("Testo vuoto.")
        return None, None
    return testo, "testo incollato"


def prepara_testo(tok, testo):
    """Tokenizza e restituisce (ids, testo effettivamente analizzato)."""
    ids = tok(testo, return_tensors="pt").input_ids
    if ids.shape[1] > MAX_TOKEN:
        print(f"Il testo ha {ids.shape[1]} token: analizzo solo i primi {MAX_TOKEN}.")
        ids = ids[:, :MAX_TOKEN]
        testo = tok.decode(ids[0])
    return ids.to(DISPOSITIVO), testo


# ---------------------------------------------------------------------------
# Rappresentazione comune dei risultati
#   colonne: (chiave, intestazione riga 1, intestazione riga 2, formato)
# ---------------------------------------------------------------------------
def risultato(fase, titolo, descrizione, colonne, righe, sintesi, legenda, evidenzia=None):
    return dict(fase=fase, titolo=titolo, descrizione=descrizione, colonne=colonne,
                righe=righe, sintesi=sintesi, legenda=legenda, evidenzia=evidenzia or {})


def stampa(ris, contesto):
    print("\n" + "=" * 78)
    print(f" {ris['titolo']}")
    print(f" {ris['descrizione']}")
    print(f" {contesto}")
    print("=" * 78)

    col = ris["colonne"]
    celle = [[FORMATI[c[3]][0](r[c[0]]) for c in col] for r in ris["righe"]]
    larghezze = [max(len(c[1]), len(c[2]), *(len(riga[j]) for riga in celle)) for j, c in enumerate(col)]

    def linea(valori):
        return " | ".join(v.rjust(w) for v, w in zip(valori, larghezze))

    print(linea([c[1] for c in col]))
    print(linea([c[2] for c in col]))
    print("-+-".join("-" * w for w in larghezze))
    for r, riga in zip(ris["righe"], celle):
        nota = ris["evidenzia"].get(r["strato"], "")
        print(linea(riga) + (f"   <-- {nota}" if nota else ""))

    print("\nSINTESI")
    for etichetta, valore in ris["sintesi"]:
        print(f"  - {etichetta}: {valore}")


def salva_excel(ris, info_modello, testo=None, fonte=None, n_token=None):
    if not EXCEL_DISPONIBILE:
        print("\n(File Excel non creato: manca openpyxl. Installalo con  pip install openpyxl)")
        return
    os.makedirs(CARTELLA_RISULTATI, exist_ok=True)
    adesso = datetime.now()
    nome_file = f"{adesso:%Y%m%d_%H%M%S}_{nome_breve(info_modello['nome'])}_{ris['fase']}.xlsx"
    percorso = os.path.join(CARTELLA_RISULTATI, nome_file)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = ris["fase"]
    col = ris["colonne"]
    n_col = len(col)
    grassetto = Font(bold=True)
    titolo_sez = Font(bold=True, size=12, color="1F3864")
    sfondo_int = PatternFill("solid", fgColor="1F3864")
    sfondo_evid = PatternFill("solid", fgColor="FFF2CC")
    bordo = Border(bottom=Side(style="thin", color="BFBFBF"))
    a_capo = Alignment(wrap_text=True, vertical="top")

    ultima = max(n_col + 1, 7)                       # ultima colonna usata (tabella + nota)
    sinistra = Alignment(horizontal="left", vertical="top", wrap_text=True)

    def coppia(r, etichetta, valore):
        """Etichetta su A:B, valore su C:ultima, entrambi allineati a sinistra."""
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=ultima)
        e = ws.cell(r, 1, etichetta)
        e.font = grassetto
        e.alignment = sinistra
        v = ws.cell(r, 3, valore)
        v.alignment = sinistra
        return v

    riga = 1
    ws.cell(riga, 1, ris["titolo"]).font = Font(bold=True, size=14)
    riga += 1
    ws.cell(riga, 1, ris["descrizione"]).font = Font(italic=True)
    riga += 2

    # --- Informazioni sulla misura
    ws.cell(riga, 1, "INFORMAZIONI").font = titolo_sez
    riga += 1
    info = [
        ("Modello", info_modello["nome"]),
        ("Commit dei pesi", info_modello["commit"]),
        ("Struttura", f"{info_modello['strati']} strati x {info_modello['canali']} canali x "
                      f"{info_modello['stati']} stati"),
        ("Dispositivo", NOME_DISPOSITIVO),
        ("Formato numerico", "FP32, con riferimento esatto in FP64 per il margine 1 - abar"
                             if ris["fase"] == "Fase2b" else "FP32"),
        ("Data e ora", f"{adesso:%d/%m/%Y %H:%M:%S}"),
    ]
    if testo is not None:
        info += [("Fonte del testo", fonte), ("Token analizzati", str(n_token))]
    for etichetta, valore in info:
        coppia(riga, etichetta, valore)
        riga += 1
    if testo is not None:
        coppia(riga, "Testo analizzato", testo[:32000])
        caratteri_per_riga = 23 * (ultima - 2)
        n_righe = sum(len(p) // caratteri_per_riga + 1 for p in testo.split("\n"))
        ws.row_dimensions[riga].height = min(409, 15 * (n_righe + 1))
        riga += 1
    riga += 1

    # --- Sintesi
    ws.cell(riga, 1, "SINTESI").font = titolo_sez
    riga += 1
    for etichetta, valore in ris["sintesi"]:
        coppia(riga, etichetta, valore)
        riga += 1
    riga += 1

    # --- Tabella
    ws.cell(riga, 1, "RISULTATI PER STRATO").font = titolo_sez
    riga += 1
    for j, c in enumerate(col, start=1):
        cella = ws.cell(riga, j, f"{c[1]}\n{c[2]}")
        cella.font = Font(bold=True, color="FFFFFF")
        cella.fill = sfondo_int
        cella.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.row_dimensions[riga].height = 32
    riga_intestazione = riga
    riga += 1
    for r in ris["righe"]:
        for j, c in enumerate(col, start=1):
            valore = r[c[0]]
            if isinstance(valore, float) and (math.isinf(valore) or math.isnan(valore)):
                valore = None
            cella = ws.cell(riga, j, valore)
            cella.number_format = FORMATI[c[3]][1]
            cella.border = bordo
            if c[0] == "strato":
                cella.alignment = Alignment(horizontal="center")
            if r["strato"] in ris["evidenzia"]:
                cella.fill = sfondo_evid
        if r["strato"] in ris["evidenzia"]:
            nota = ws.cell(riga, n_col + 1, ris["evidenzia"][r["strato"]])
            nota.font = Font(italic=True)
            nota.alignment = Alignment(indent=1)
        riga += 1
    riga += 1

    # --- Legenda
    ws.cell(riga, 1, "COME LEGGERE LE COLONNE").font = titolo_sez
    riga += 1
    for nome_col, spiegazione in ris["legenda"]:
        coppia(riga, nome_col, spiegazione)
        riga += 1

    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 30
    for j in range(3, ultima + 1):
        ws.column_dimensions[get_column_letter(j)].width = 20
    ws.column_dimensions[get_column_letter(n_col + 1)].width = 26
    ws.freeze_panes = ws.cell(riga_intestazione + 1, 1)
    ws.page_setup.orientation = "landscape"             # stampa: tutto in larghezza su una pagina
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    wb.save(percorso)
    print(f"\nRisultati salvati in un unico file: {percorso}")


# ---------------------------------------------------------------------------
# Le fasi
# ---------------------------------------------------------------------------
def fase1(model):
    strati = model.backbone.layers
    righe = []
    with torch.no_grad():
        for i, layer in enumerate(strati):
            a = -torch.exp(layer.mixer.A_log.float())
            a_max = a.max().item()
            righe.append(dict(strato=i, a_min=a.min().item(), a_mediana=a.median().item(),
                              a_max=a_max, tau=1.0 / abs(a_max) if a_max != 0 else float("inf"),
                              instabili=int((a >= 0).sum().item()), totale=a.numel()))

    lento = max(righe, key=lambda r: r["a_max"])
    veloce = min(righe, key=lambda r: r["a_min"])
    instabili = sum(r["instabili"] for r in righe)
    totale = sum(r["totale"] for r in righe)
    sintesi = [
        ("Poli continui esaminati", fmt_int(totale)),
        ("Poli instabili (a >= 0)", f"{fmt_int(instabili)}  ->  "
                                    + ("tutti i modi sono asintoticamente stabili" if instabili == 0
                                       else "ATTENZIONE: esistono modi non stabili")),
        ("Polo piu' lento del modello", f"strato {lento['strato']}, a = {fmt_sci(lento['a_max'])}, "
                                        f"costante di tempo {fmt_sci(lento['tau'])}"),
        ("Polo piu' veloce del modello", f"strato {veloce['strato']}, a = {fmt_sci(veloce['a_min'])}"),
        ("Ampiezza dello spettro", f"{math.log10(veloce['a_min'] / lento['a_max']):.1f} ordini di grandezza"),
    ]
    evidenzia = {lento["strato"]: "polo piu' lento", veloce["strato"]: "polo piu' veloce"}
    if lento["strato"] == veloce["strato"]:
        evidenzia = {lento["strato"]: "polo piu' lento e piu' veloce"}
    return risultato(
        "Fase1", "FASE 1 - Poli continui  a = -exp(A_log)",
        "Letti dai pesi del modello: non dipendono dal testo.",
        [("strato", "Strato", "", "int"),
         ("a_min", "Polo piu' veloce", "a_min", "sci"),
         ("a_mediana", "Polo tipico", "mediana", "sci"),
         ("a_max", "Polo piu' lento", "a_max", "sci"),
         ("tau", "Costante di tempo", "max = 1/|a_max|", "sci"),
         ("instabili", "Poli instabili", "a >= 0", "int")],
        righe, sintesi,
        [("Polo piu' veloce", "Il polo piu' negativo dello strato: il modo che si estingue piu' in fretta."),
         ("Polo tipico", "La mediana dei poli dello strato: meta' sono piu' veloci, meta' piu' lenti."),
         ("Polo piu' lento", "Il polo piu' vicino a zero: il modo con la memoria piu' lunga."),
         ("Costante di tempo", "1/|a_max| in unita' di tempo continuo; in token dipende da Delta (Fase 2)."),
         ("Poli instabili", "Poli con a >= 0. Per costruzione (a = -exp) devono essere zero.")],
        evidenzia)


def esegui_con_sonde(model, ids, crea_hook):
    strati = model.backbone.layers
    maniglie = [layer.mixer.x_proj.register_forward_hook(crea_hook(i, layer.mixer))
                for i, layer in enumerate(strati)]
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for m in maniglie:
            m.remove()


def delta_fp32(mixer, ts):
    return F.softplus(ts.float() @ mixer.dt_proj.weight.float().T + mixer.dt_proj.bias.float())


def fase2a(model, ids):
    dati = {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            d = delta_fp32(mixer, uscita[0, :, :mixer.time_step_rank])
            d_min, d_max = d.min().item(), d.max().item()
            dati[i] = dict(strato=i, d_min=d_min, d_mediana=d.median().item(), d_max=d_max,
                           decadi=math.log10(d_max / d_min) if d_min > 0 else float("inf"),
                           valori=d.numel())
        return hook

    esegui_con_sonde(model, ids, crea_hook)
    righe = [dati[i] for i in range(len(dati))]
    lento = min(righe, key=lambda r: r["d_mediana"])
    rapido = max(righe, key=lambda r: r["d_mediana"])
    sintesi = [
        ("Valori di Delta per strato", f"{fmt_int(righe[0]['valori'])} "
                                       f"({ids.shape[1]} token x {righe[0]['valori'] // ids.shape[1]} canali)"),
        ("Strato con il passo tipico piu' piccolo", f"strato {lento['strato']}, Delta mediano = "
                                                    f"{fmt_sci(lento['d_mediana'])} (memoria piu' lunga)"),
        ("Strato con il passo tipico piu' grande", f"strato {rapido['strato']}, Delta mediano = "
                                                   f"{fmt_sci(rapido['d_mediana'])} (si resetta di piu')"),
        ("Delta minimo e massimo del modello", f"{fmt_sci(min(r['d_min'] for r in righe))}  /  "
                                               f"{fmt_sci(max(r['d_max'] for r in righe))}"),
    ]
    return risultato(
        "Fase2a", "FASE 2a - Passo di campionamento Delta",
        "Calcolato dal modello token per token: dipende dal testo letto.",
        [("strato", "Strato", "", "int"),
         ("d_min", "Delta minimo", "(ignora il token)", "sci"),
         ("d_mediana", "Delta tipico", "mediana", "sci"),
         ("d_max", "Delta massimo", "(si resetta)", "sci"),
         ("decadi", "Selettivita'", "decadi max/min", "dec")],
        righe, sintesi,
        [("Delta", "Periodo di campionamento del canale a quel token: abar = exp(Delta * a)."),
         ("Delta minimo", "Il token che lo strato 'salta' di piu': lo stato resta quasi invariato."),
         ("Delta tipico", "La mediana su tutti i token e i canali. Inizializzazione: tra 0.001 e 0.1."),
         ("Delta massimo", "Il token su cui lo strato 'riparte da zero': lo stato si svuota."),
         ("Selettivita'", "log10(Delta max / Delta min): quanti ordini di grandezza varia il passo.")],
        {lento["strato"]: "passo tipico piu' piccolo", rapido["strato"]: "passo tipico piu' grande"})


def fase2b(model, ids):
    dati = {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            ts = uscita[0, :, :mixer.time_step_rank]
            W, b = mixer.dt_proj.weight, mixer.dt_proj.bias
            d32 = delta_fp32(mixer, ts)
            a32 = -torch.exp(mixer.A_log.float())
            d64 = F.softplus(ts.double() @ W.double().T + b.double())
            a64 = -torch.exp(mixer.A_log.double())
            L, D = d32.shape
            passo = max(1, 4_000_000 // (D * a32.shape[1]))       # blocchi di token: memoria limitata
            margine_min, su_1, in_0, fuori = float("inf"), 0, 0, 0
            for s in range(0, L, passo):
                abar32 = torch.exp(d32[s:s + passo, :, None] * a32)             # come il modello (FP32)
                margine64 = -torch.expm1(d64[s:s + passo, :, None] * a64)        # 1 - abar esatto (FP64)
                margine_min = min(margine_min, margine64.min().item())
                su_1 += int((abar32 == 1.0).sum().item())
                in_0 += int((abar32 == 0.0).sum().item())
                fuori += int((abar32 > 1.0).sum().item())
            totale = L * a32.numel()
            dati[i] = dict(strato=i, margine=margine_min,
                           memoria=1.0 / margine_min if margine_min > 0 else float("inf"),
                           su_1=su_1, perc_1=su_1 / totale, in_0=in_0, fuori=fuori, totale=totale)
        return hook

    esegui_con_sonde(model, ids, crea_hook)
    righe = [dati[i] for i in range(len(dati))]
    totale = sum(r["totale"] for r in righe)
    su_1 = sum(r["su_1"] for r in righe)
    in_0 = sum(r["in_0"] for r in righe)
    fuori = sum(r["fuori"] for r in righe)
    piu_vicino = min(righe, key=lambda r: r["margine"])
    peggiori = [r for r in sorted(righe, key=lambda r: -r["su_1"]) if r["su_1"] > 0][:3]
    sintesi = [
        ("Poli discreti esaminati", f"{fmt_int(totale)}  ({ids.shape[1]} token)"),
        ("Fuori dal cerchio unitario (abar > 1)",
         f"{fmt_int(fuori)}  ->  " + ("nessun polo instabile" if fuori == 0 else "ATTENZIONE: poli instabili")),
        ("Arrotondati a 1 in FP32 (marginalmente stabili)", f"{fmt_int(su_1)}  ({fmt_perc(su_1 / totale)})"),
        ("Strati con piu' poli su 1",
         ", ".join(f"{r['strato']} ({fmt_int(r['su_1'])})" for r in peggiori) if peggiori else "nessuno"),
        ("Polo discreto piu' vicino a 1", f"strato {piu_vicino['strato']}, 1 - abar = "
                                          f"{fmt_sci(piu_vicino['margine'])} "
                                          f"(memoria circa {fmt_sci(piu_vicino['memoria'])} token)"),
        ("Arrotondati a 0 in FP32 (senza memoria, innocui)", f"{fmt_int(in_0)}  ({fmt_perc(in_0 / totale)})"),
    ]
    evidenzia = {r["strato"]: f"n. {k} per poli su 1" for k, r in enumerate(peggiori, start=1)}
    return risultato(
        "Fase2b", "FASE 2b - Poli discreti  abar = exp(Delta * a)",
        "Calcolati in FP32 come nel modello; il margine esatto 1 - abar e' calcolato in FP64.",
        [("strato", "Strato", "", "int"),
         ("margine", "Polo piu' vicino a 1", "margine 1 - abar", "sci"),
         ("memoria", "Memoria massima", "token ~ 1/(1-abar)", "sci"),
         ("su_1", "Poli su 1 in FP32", "numero", "int"),
         ("perc_1", "Poli su 1 in FP32", "% dello strato", "perc"),
         ("in_0", "Poli in 0 in FP32", "numero", "int")],
        righe, sintesi,
        [("Polo piu' vicino a 1", "Il margine di stabilita' piu' piccolo dello strato, calcolato esattamente (FP64)."),
         ("Memoria massima", "Quanti token servono a quel modo per dimenticare: circa 1/(1 - abar)."),
         ("Poli su 1 in FP32", "Poli che FP32 arrotonda esattamente a 1: da asintoticamente a marginalmente stabili."),
         ("% dello strato", "Quota dei poli dello strato (token x canali x stati) arrotondati a 1."),
         ("Poli in 0 in FP32", "Modi che si azzerano a ogni token: nessuna memoria, nessun rischio per la stabilita'.")],
        evidenzia)


# ---------------------------------------------------------------------------
# Programma principale
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" Analisi dei modi di Mamba - Fasi 1, 2a, 2b (FP32)")
    print(f" Dispositivo: {NOME_DISPOSITIVO}")
    if not EXCEL_DISPONIBILE:
        print(" NOTA: openpyxl non installato, i risultati non verranno salvati.")
        print("       Per salvarli: chiudi e lancia  pip install openpyxl")
    print("=" * 60)

    model = tok = info_modello = None
    while model is None:
        model, tok, info_modello = carica_modello(scegli_modello())

    testo = fonte = ids = None
    while True:
        info_testo = f"{ids.shape[1]} token ({fonte})" if ids is not None else "nessuno"
        scelta = scegli(f"--- Menu ---   modello: {nome_breve(info_modello['nome'])}   testo: {info_testo}", [
            ("1", "Fase 1  - poli continui (dai pesi, non usa il testo)"),
            ("2", "Fase 2a - passo di campionamento Delta (sul testo)"),
            ("3", "Fase 2b - poli discreti abar (sul testo)"),
            ("4", "Cambia testo"),
            ("5", "Cambia modello"),
            ("0", "Esci"),
        ])
        if scelta == "0":
            print("Chiusura.")
            break

        if scelta == "4" or (scelta in ("2", "3") and ids is None):
            nuovo, nuova_fonte = chiedi_testo()
            if nuovo is not None:
                ids, testo = prepara_testo(tok, nuovo)
                fonte = nuova_fonte
                print(f"Testo pronto: {ids.shape[1]} token.")
            if scelta == "4" or ids is None:
                continue

        if scelta == "5":
            libera_modello(model)
            model = None
            while model is None:
                model, tok, info_modello = carica_modello(scegli_modello())
            if testo is not None:
                ids, testo = prepara_testo(tok, testo)   # stesso testo, ritokenizzato
            continue

        contesto = f"Modello: {nome_breve(info_modello['nome'])}   Dispositivo: {NOME_DISPOSITIVO}   Formato: FP32"
        t0 = time.time()
        try:
            if scelta == "1":
                ris = fase1(model)
                stampa(ris, contesto)
                salva_excel(ris, info_modello)
            else:
                ris = fase2a(model, ids) if scelta == "2" else fase2b(model, ids)
                stampa(ris, contesto + f"   Testo: {ids.shape[1]} token ({fonte})")
                salva_excel(ris, info_modello, testo, fonte, ids.shape[1])
            print(f"Tempo: {time.time() - t0:.1f} s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("\nMemoria della GPU esaurita: prova con un testo piu' corto.")
        except Exception as e:
            print(f"\nErrore durante la fase: {type(e).__name__}: {e}")

        chiedi("\nPremi Invio per tornare al menu...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
