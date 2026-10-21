import glob
import hashlib
import os
import sys
from contextlib import contextmanager

import torch

from nucleo.modelli import _carica_pesi, libera
from nucleo.risultato import Indicatore
from nucleo.sonde import delta_modello, esegui_con_sonde

# Formati dei pesi per la Fase 3: il controllo FP32 deve dare errori esattamente nulli
FORMATI = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32 (controllo)": torch.float32}

# In FP16 un valore positivo sotto 2^-25 (metà del più piccolo subnormale, 2^-24) diventa 0
SOGLIA_FP16_ZERO = 2.0 ** -25

# Memoria (GB) lasciata libera quando si sposta il modello FP32 su CPU
MARGINE_GB = 1.5


# Turing (RTX 20xx) non ha BF16 nativo: PyTorch lo emula calcolando in FP32 e riarrotondando
def bf16_emulato(dispositivo):
    return (dispositivo == "cuda" and torch.cuda.is_bf16_supported()
            and not torch.cuda.is_bf16_supported(including_emulation=False))


def descrivi_formato(nome, dispositivo):
    tipo = FORMATI[nome]
    if tipo == torch.float32:
        return "pesi in FP32 (controllo), ricorrenza in FP32"
    emulato = " emulato sulla GPU" if tipo == torch.bfloat16 and bf16_emulato(dispositivo) else ""
    return f"pesi in {nome}{emulato}, ricorrenza in FP32"


# Impronta dei pesi FP32 calcolata sul dispositivo: somma dei bit (int64) e somma in FP64 di ogni parametro
def impronta(model):
    with torch.no_grad():
        return [(nome, int(p.view(torch.int32).sum(dtype=torch.int64).item()), p.double().sum().item())
                for nome, p in model.named_parameters()]


def _gb(model, tipo=None):
    return sum(p.numel() * (torch.finfo(tipo).bits // 8 if tipo else p.element_size())
               for p in model.parameters()) / 2 ** 30


# Memoria virtuale (commit) libera in GB. Su Windows anche la memoria della GPU consuma commit (WDDM):
# col 1.4B, FP32 su CPU più la copia sulla GPU possono superare RAM + file di paging
def commit_libero_gb():
    if sys.platform != "win32":
        return None
    import ctypes

    class Stato(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] +                    [(k, ctypes.c_ulonglong) for k in ("totale", "fisica", "commit_totale", "commit_libero",
                                                      "virtuale", "virtuale_libera", "estesa")]
    stato = Stato(dwLength=ctypes.sizeof(Stato))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stato))
    return stato.commit_libero / 2 ** 30


# Rilegge dal disco i pesi FP32 dentro gli stessi Parameter del modello, un tensore alla volta (pread, senza mmap)
def _rileggi_pesi(model, percorso, dispositivo):
    from safetensors import safe_open
    parametri = dict(model.named_parameters())
    letti = set()
    for file in sorted(glob.glob(os.path.join(percorso, "*.safetensors"))):
        try:
            aperto = safe_open(file, framework="pt", device="cpu", backend="pread")
        except TypeError:
            aperto = safe_open(file, framework="pt", device="cpu")
        with aperto as f:
            for chiave in f.keys():
                if chiave in parametri:
                    p = parametri[chiave]
                    p.data = f.get_tensor(chiave).to(device=dispositivo, dtype=p.dtype)
                    letti.add(chiave)
    if letti != set(parametri):
        raise RuntimeError(f"Pesi FP32 non trovati nei file: {sorted(set(parametri) - letti)[:5]}")


# Seconda copia del modello nel formato richiesto, letta dal disco. Sulla GPU il modello FP32 condiviso
# libera la VRAM (col 1.4B le due copie non entrano in 8 GB): va temporaneamente su CPU se la memoria lo
# permette, altrimenti i suoi pesi vengono tolti e alla fine riletti dal disco. In entrambi i casi torna
# sulla GPU anche in caso di errore, e alla fine si controlla che i pesi FP32 siano identici a prima.
@contextmanager
def copia_in_formato(ctx, tipo):
    prima = impronta(ctx.model)
    modo = None
    if ctx.dispositivo == "cuda":
        libero = commit_libero_gb()
        serve = _gb(ctx.model) + _gb(ctx.model, tipo) + MARGINE_GB
        modo = "cpu" if libero is None or libero > serve else "disco"
    esito = dict(modo=modo, fp32_intatto=None)
    copia = None
    try:
        if modo == "cpu":
            ctx.model.to("cpu")
        elif modo == "disco":
            for p in ctx.model.parameters():
                p.data = torch.empty(0, dtype=p.dtype, device=ctx.dispositivo)
        libera()
        copia = _carica_pesi(ctx.info_modello["percorso"], ctx.dispositivo, tipo).eval()
        yield copia, esito
    finally:
        if copia is not None:
            # Libera la memoria anche se chi chiama tiene ancora un riferimento alla copia
            for t in list(copia.parameters()) + list(copia.buffers()):
                t.data = torch.empty(0, dtype=t.dtype)
            del copia
        libera()
        if modo == "cpu":
            ctx.model.to(ctx.dispositivo)
        elif modo == "disco":
            _rileggi_pesi(ctx.model, ctx.info_modello["percorso"], ctx.dispositivo)
        esito["fp32_intatto"] = impronta(ctx.model) == prima
        if not esito["fp32_intatto"]:
            raise RuntimeError("I pesi del modello FP32 sono cambiati durante la Fase 3: ricarica il modello.")


# Passaggio di riferimento sul modello FP32 condiviso: Delta di ogni strato, tenuti su CPU in FP32
def delta_baseline(ctx, avanzamento=lambda frazione, messaggio="": None):
    n = len(ctx.model.backbone.layers)
    base = {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            base[i] = delta_modello(mixer, uscita).float().cpu()
            avanzamento((i + 1) / n, f"baseline FP32, strato {i + 1} di {n}")
        return hook

    esegui_con_sonde(ctx.model, ctx.ids, crea_hook)
    return base


# Stesso testo nella copia a bassa precisione: crea_hook(i, mixer) riceve i mixer della copia
def esegui_in_formato(ctx, tipo, crea_hook):
    with copia_in_formato(ctx, tipo) as (copia, esito):
        esegui_con_sonde(copia, ctx.ids, crea_hook)
    return esito


# Passaggio nella copia a bassa precisione, condiviso dalle Fasi 2 e 3: una sola volta per modello, testo e formato.
# Per ogni strato tiene su CPU il Delta della copia nel suo formato, l'A_log della copia e i valori non finiti
# all'ingresso di x_proj; le fasi li elaborano dopo, a copia già liberata. Col 1.4B e 1024 token sono circa
# 400 MB per formato; la cache si svuota quando cambiano modello o testo.
_CACHE = {}


def svuota_cache():
    _CACHE.clear()
    libera()


def _chiave_cache(ctx, nome):
    impronta_testo = hashlib.sha1(ctx.ids.cpu().numpy().tobytes()).hexdigest()
    return ctx.info_modello["percorso"], ctx.info_modello["commit"], ctx.dispositivo, impronta_testo, nome


def passaggio_ridotto(ctx, nome, avanzamento=lambda frazione, messaggio="": None):
    chiave = _chiave_cache(ctx, nome)
    if chiave in _CACHE:
        avanzamento(1.0, f"riuso il passaggio in {nome} già fatto su questo testo")
        return _CACHE[chiave]
    if any(k[:4] != chiave[:4] for k in _CACHE):
        svuota_cache()
    n = len(ctx.model.backbone.layers)
    strati = {}

    def crea_hook(i, mixer):
        def hook(modulo, ingresso, uscita):
            strati[i] = dict(delta=delta_modello(mixer, uscita).cpu(), A_log=mixer.A_log.detach().cpu(),
                             non_finiti_ingresso=int((~torch.isfinite(ingresso[0])).sum().item()))
            avanzamento((i + 1) / n, f"{nome}, strato {i + 1} di {n}")
        return hook

    avanzamento(0.0, f"carico la copia in {nome}")
    esito = esegui_in_formato(ctx, FORMATI[nome], crea_hook)
    _CACHE[chiave] = dict(strati=[strati[i] for i in range(n)], esito=esito)
    return _CACHE[chiave]


def indicatore_formato(nome, dispositivo, esito):
    note = []
    if FORMATI[nome] == torch.bfloat16 and bf16_emulato(dispositivo):
        note.append("BF16 emulato da PyTorch sulla GPU (Turing non ha BF16 nativo)")
    elif FORMATI[nome] == torch.float32:
        note.append("controllo: seconda copia in FP32, gli errori devono essere nulli")
    if esito.get("modo") == "cpu":
        note.append("FP32 spostato su CPU durante il confronto")
    elif esito.get("modo") == "disco":
        note.append("pesi FP32 tolti dalla GPU durante il confronto e riletti dal disco (memoria insufficiente per tenerli su CPU)")
    note.append("modello FP32 intatto (impronta dei pesi verificata)" if esito.get("fp32_intatto")
                else "modello FP32 da ricaricare")
    return Indicatore("Formato dei pesi", nome, " · ".join(note), "ok" if esito.get("fp32_intatto") else "attenzione")
