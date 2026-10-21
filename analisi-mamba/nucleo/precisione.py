from contextlib import contextmanager

import torch

from nucleo.modelli import _carica_pesi, libera

# Formati dei pesi per la Fase 3: il controllo FP32 deve dare errori esattamente nulli
FORMATI = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32 (controllo)": torch.float32}

# In FP16 un valore positivo sotto 2^-25 (metà del più piccolo subnormale, 2^-24) diventa 0
SOGLIA_FP16_ZERO = 2.0 ** -25


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


# Seconda copia del modello nel formato richiesto, letta dal disco. Sulla GPU il modello FP32 condiviso
# va temporaneamente su CPU (col 1.4B le due copie non entrano in 8 GB) e torna sulla GPU alla fine,
# anche in caso di errore. Alla fine si controlla che i pesi FP32 siano identici a prima.
@contextmanager
def copia_in_formato(ctx, tipo):
    prima = impronta(ctx.model)
    sposta = ctx.dispositivo == "cuda"
    esito = dict(spostato_su_cpu=sposta, fp32_intatto=None)
    copia = None
    try:
        if sposta:
            ctx.model.to("cpu")
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
        if sposta:
            ctx.model.to(ctx.dispositivo)
        esito["fp32_intatto"] = impronta(ctx.model) == prima
        if not esito["fp32_intatto"]:
            raise RuntimeError("I pesi del modello FP32 sono cambiati durante la Fase 3: ricarica il modello.")
