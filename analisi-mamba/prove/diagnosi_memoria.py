# Diagnosi di un'uscita silenziosa (crash senza errore): esegue le fasi fuori da Streamlit e registra, a ogni passo
# dell'avanzamento, memoria della GPU (PyTorch), memoria del processo (RAM e commit) e commit libero di Windows.
# Ogni riga è scritta subito su disco, e faulthandler scrive lo stack Python se il processo muore in codice nativo
# (es. violazione di accesso): l'ultima riga del registro dice dove e con quanta memoria si è fermato.
# Uso:  python prove/diagnosi_memoria.py [--modello 1.4b] [--corpus bbc] [--articolo 371] [--token 2048]
#                                       [--fasi 2] [--confronto "FP32 vs FP16"] [--cpu]
# Registro in risultati/diagnosi_AAAAMMGG_HHMMSS.txt
import argparse
import faulthandler
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fasi import carica_fasi
from nucleo import corpora
from nucleo.modelli import carica_e_verifica, cartella_cache_predefinita, trova_modelli
from nucleo.precisione import commit_libero_gb
from nucleo.risultato import Contesto
from nucleo.testo import TESTO_ESEMPIO, prepara

CARTELLA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Memoria del processo su Windows: (working set, commit privato) in GB
def memoria_processo():
    if sys.platform != "win32":
        return None, None
    import ctypes
    from ctypes import wintypes

    class Contatori(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (k, ctypes.c_size_t) for k in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                           "PrivateUsage")]
    c = Contatori(cb=ctypes.sizeof(Contatori))
    processo = ctypes.windll.kernel32.GetCurrentProcess
    processo.restype = wintypes.HANDLE
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Contatori), wintypes.DWORD]
    info(processo(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2 ** 30, c.PrivateUsage / 2 ** 30


class Registro:
    def __init__(self, percorso):
        self.f = open(percorso, "w", encoding="utf-8", buffering=1)
        self.t0 = time.time()
        self.ultimo = 0.0

    def scrivi(self, messaggio, forza=True):
        adesso = time.time()
        if not forza and adesso - self.ultimo < 2:
            return
        self.ultimo = adesso
        ram, commit = memoria_processo()
        libero = commit_libero_gb()
        gpu = (f"GPU allocata {torch.cuda.memory_allocated() / 2 ** 30:5.2f} GB, riservata "
               f"{torch.cuda.memory_reserved() / 2 ** 30:5.2f} GB, picco {torch.cuda.max_memory_allocated() / 2 ** 30:5.2f} GB"
               if torch.cuda.is_available() else "GPU -")
        proc = f"RAM processo {ram:5.2f} GB, commit processo {commit:5.2f} GB" if ram is not None else "RAM -"
        sistema = f"commit libero Windows {libero:5.2f} GB" if libero is not None else ""
        riga = f"[{adesso - self.t0:7.1f} s] {messaggio} | {gpu} | {proc} | {sistema}"
        print(riga, flush=True)
        self.f.write(riga + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modello", default="1.4b")
    p.add_argument("--corpus", default="bbc", help="bbc, wikitext o esempio")
    p.add_argument("--articolo", type=int, default=371, help="numero come nell'Excel (n. 371 = il 371°)")
    p.add_argument("--token", type=int, default=2048)
    p.add_argument("--fasi", default="2")
    p.add_argument("--confronto", default="FP32 vs FP16")
    p.add_argument("--cpu", action="store_true")
    arg = p.parse_args()

    os.makedirs(os.path.join(CARTELLA, "risultati"), exist_ok=True)
    base = os.path.join(CARTELLA, "risultati", f"diagnosi_{datetime.now():%Y%m%d_%H%M%S}")
    crash = open(base + "_crash.txt", "w", encoding="utf-8")
    faulthandler.enable(file=crash, all_threads=True)
    reg = Registro(base + ".txt")
    reg.scrivi(f"avvio: {' '.join(sys.argv[1:]) or 'parametri predefiniti'}")

    dispositivo = "cpu" if arg.cpu or not torch.cuda.is_available() else "cuda"
    modello = next(m for m in trova_modelli(cartella_cache_predefinita()) if m.is_mamba and arg.modello in m.nome)
    model, tok, info, controlli = carica_e_verifica(
        modello, dispositivo, lambda superati, prossimo: reg.scrivi(f"caricamento: {prossimo}"))
    if model is None:
        reg.scrivi(f"caricamento fallito: {controlli[-1]}")
        return
    reg.scrivi(f"modello caricato: {info['nome']} su {dispositivo}")

    if arg.corpus == "esempio":
        testo = TESTO_ESEMPIO
    else:
        nome = next(c for c in os.listdir(corpora.CARTELLA_CORPORA) if c.startswith(arg.corpus))
        corpus = corpora.carica_corpus(os.path.join(corpora.CARTELLA_CORPORA, nome))
        testo = corpus["articoli"][arg.articolo - 1]["testo"]
    ids, token, testo, _ = prepara(tok, testo, dispositivo, arg.token)
    reg.scrivi(f"testo pronto: {ids.shape[1]} token")

    ctx = Contesto(model=model, tok=tok, info_modello=info, dispositivo=dispositivo, nome_dispositivo=dispositivo,
                   ids=ids, token=token, testo=testo, fonte_testo=arg.corpus,
                   avanzamento=lambda frazione, messaggio="": reg.scrivi(f"{frazione:6.1%} {messaggio}", forza=False))
    fasi, errori = carica_fasi()
    for fase in [f for f in fasi if f.codice in arg.fasi.split(",")]:
        ctx.parametri = {q.chiave: (arg.confronto if q.chiave == "confronto" else q.predefinito) for q in fase.parametri}
        reg.scrivi(f"inizio Fase {fase.codice} {ctx.parametri}")
        try:
            fase.esegui(ctx)
            reg.scrivi(f"fine Fase {fase.codice}")
        except Exception as e:
            reg.scrivi(f"ERRORE nella Fase {fase.codice}: {type(e).__name__}: {str(e).splitlines()[0]}")
            raise
    reg.scrivi("completato")


if __name__ == "__main__":
    main()
