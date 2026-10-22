import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ESTENSIONI_PESI = (".safetensors", ".bin")


@dataclass
class ModelloLocale:
    nome: str
    percorso: str
    commit: str
    tipo: str
    dimensione_gb: float

    @property
    def is_mamba(self):
        return self.tipo == "mamba"


@dataclass
class Controllo:
    descrizione: str
    ok: bool
    dettaglio: str = ""


class _Fallimento(Exception):
    pass


def cartella_cache_predefinita():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return str(Path.home() / ".cache" / "huggingface" / "hub")


def _leggi_tipo(cartella):
    try:
        with open(os.path.join(cartella, "config.json"), encoding="utf-8") as f:
            return json.load(f).get("model_type", "sconosciuto")
    except (OSError, ValueError):
        return "illeggibile"


def _leggi_ref(percorso):
    try:
        with open(percorso, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _dimensione_gb(cartella):
    totale = 0
    for f in os.listdir(cartella):
        if f.endswith(ESTENSIONI_PESI):
            try:
                totale += os.path.getsize(os.path.join(cartella, f))
            except OSError:
                pass
    return totale / 1024 ** 3


# Ricerca dei modelli: cartella-modello, cache di Hugging Face o cartelle salvate con save_pretrained
def trova_modelli(cartella):
    cartella = os.path.expanduser(cartella.strip().strip('"'))
    if not os.path.isdir(cartella):
        return []
    trovati = []

    def aggiungi(nome, percorso, commit):
        if os.path.isfile(os.path.join(percorso, "config.json")):
            trovati.append(ModelloLocale(nome, percorso, commit, _leggi_tipo(percorso), _dimensione_gb(percorso)))

    aggiungi(os.path.basename(os.path.normpath(cartella)), cartella, "-")
    if trovati:
        return trovati

    for voce in sorted(os.listdir(cartella)):
        sotto = os.path.join(cartella, voce)
        if not os.path.isdir(sotto):
            continue
        if voce.startswith("models--"):
            snapshots = os.path.join(sotto, "snapshots")
            if not os.path.isdir(snapshots):
                continue
            nome = voce[len("models--"):].replace("--", "/")
            principale = _leggi_ref(os.path.join(sotto, "refs", "main"))
            commit = sorted(os.listdir(snapshots), key=lambda c: (c != principale, c))
            for c in commit:
                aggiungi(nome if c == commit[0] else f"{nome} @ {c[:10]}", os.path.join(snapshots, c), c[:10])
        elif os.path.isfile(os.path.join(sotto, "config.json")):
            aggiungi(voce, sotto, "-")
        else:
            for voce2 in sorted(os.listdir(sotto)):
                sotto2 = os.path.join(sotto, voce2)
                if os.path.isdir(sotto2):
                    aggiungi(f"{voce}/{voce2}", sotto2, "-")
    return trovati


def dispositivi_disponibili():
    elenco = [("cuda", f"GPU {torch.cuda.get_device_name(0)}")] if torch.cuda.is_available() else []
    return elenco + [("cpu", "CPU")]


# Caricamento (FP32 se non indicato) con ripiego per transformers vecchi (torch_dtype) o senza accelerate (device_map)
def _carica_pesi(percorso, dispositivo, tipo=torch.float32):
    from transformers import MambaForCausalLM
    carica = lambda **kw: MambaForCausalLM.from_pretrained(percorso, local_files_only=True, **kw)
    try:
        return carica(dtype=tipo, device_map=dispositivo)
    except (TypeError, ValueError, ImportError):
        pass
    try:
        model = carica(dtype=tipo)
    except TypeError:
        model = carica(torch_dtype=tipo)
    return model.to(dispositivo)


# Carica il modello dal disco ed esegue i sei controlli: al primo fallito restituisce model = None.
# avanzamento(superati, prossimo) viene chiamata prima di ogni controllo
def carica_e_verifica(modello: ModelloLocale, dispositivo: str, avanzamento=lambda superati, prossimo: None):
    from transformers import AutoTokenizer

    controlli = []
    model = None

    def superato(descrizione, dettaglio=""):
        controlli.append(Controllo(descrizione, True, dettaglio))

    def errore(e):
        return f"{type(e).__name__}: {e}"

    try:
        avanzamento(0, "Architettura Mamba-1")
        if modello.tipo != "mamba":
            aiuto = (" Mamba-2 ha una A scalare per testa: le fasi sono scritte per Mamba-1."
                     if modello.tipo == "mamba2" else "")
            raise _Fallimento("Architettura Mamba-1 (model_type = mamba)",
                              f"config.json indica '{modello.tipo}'.{aiuto}")
        superato("Architettura Mamba-1 (model_type = mamba)")

        avanzamento(1, "Tokenizer")
        try:
            tok = AutoTokenizer.from_pretrained(modello.percorso, local_files_only=True)
        except Exception as e:
            raise _Fallimento("Tokenizer letto dalla cartella", errore(e))
        superato("Tokenizer letto dalla cartella", type(tok).__name__)

        avanzamento(2, "Pesi in FP32")
        t0 = time.time()
        try:
            model = _carica_pesi(modello.percorso, dispositivo).float().eval()
        except torch.cuda.OutOfMemoryError:
            raise _Fallimento("Pesi caricati in FP32", "Memoria della GPU esaurita: il modello non entra nella VRAM.")
        except Exception as e:
            raise _Fallimento("Pesi caricati in FP32", errore(e))
        superato("Pesi caricati in FP32", f"in {time.time() - t0:.1f} s")

        avanzamento(3, "Struttura interna")
        try:
            strati = model.backbone.layers
            mancanti = [a for a in ("A_log", "x_proj", "dt_proj", "time_step_rank") if not hasattr(strati[0].mixer, a)]
        except AttributeError as e:
            mancanti = [str(e)]
        if mancanti:
            raise _Fallimento("Struttura interna (A_log, x_proj, dt_proj)", "Mancano: " + ", ".join(mancanti))
        superato("Struttura interna (A_log, x_proj, dt_proj)", f"{len(strati)} strati")

        avanzamento(4, "Valori di A_log")
        with torch.no_grad():
            non_finiti = sum(int((~torch.isfinite(l.mixer.A_log)).sum().item()) for l in strati)
        if non_finiti:
            raise _Fallimento("A_log senza NaN o infiniti", f"{non_finiti} valori non finiti")
        superato("A_log senza NaN o infiniti")

        avanzamento(5, "Prova di lettura")
        try:
            ids = tok("Control theory", return_tensors="pt").input_ids.to(dispositivo)
            with torch.no_grad():
                logits = model(ids, use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise ValueError("uscita non finita")
        except Exception as e:
            raise _Fallimento("Prova di lettura su un testo breve", errore(e))
        superato("Prova di lettura su un testo breve")

    except _Fallimento as f:
        model = None
        libera()
        descrizione, dettaglio = f.args
        controlli.append(Controllo(descrizione, False, dettaglio))
        return None, None, None, controlli

    cfg = model.config
    info = dict(nome=modello.nome, percorso=modello.percorso, commit=modello.commit,
                strati=cfg.num_hidden_layers, canali=cfg.intermediate_size, stati=cfg.state_size,
                rank_delta=strati[0].mixer.time_step_rank,
                poli_continui=cfg.num_hidden_layers * cfg.intermediate_size * cfg.state_size,
                dimensione_gb=modello.dimensione_gb)
    return model, tok, info, controlli


def libera():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
