"""
Fase 2a - Lettura del passo di discretizzazione Delta di Mamba-130M su un testo fissato (FP32).

Uso:   python fase2a_delta.py
"""
import torch
import torch.nn.functional as F
from transformers import MambaForCausalLM, AutoTokenizer

MODELLO = "state-spaces/mamba-1.4b-hf"

# Testo fisso: lo useremo identico in tutte le fasi successive
TESTO = (
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

# 1. Carico tokenizer e modello in FP32, e trasformo il testo in token
tok = AutoTokenizer.from_pretrained(MODELLO)
model = MambaForCausalLM.from_pretrained(MODELLO).float().eval()
ids = tok(TESTO, return_tensors="pt").input_ids
strati = model.backbone.layers
print(f"Token nel testo: {ids.shape[1]}   formato: {strati[0].mixer.A_log.dtype}\n")

# 2. Preparo un "hook" per ogni strato: una funzione che PyTorch chiama
#    automaticamente quando lo strato calcola x_proj, passandoci il risultato
delta_per_strato = {}

def crea_hook(i, mixer):
    def hook(modulo, ingresso, uscita):
        R = mixer.time_step_rank                       # 48: i primi 48 numeri di x_proj servono per Delta
        ts = uscita[..., :R]                           # (1, token, 48)
        # Delta = softplus(W_dt * ts + b_dt), come nel modello
        delta = F.softplus(ts @ mixer.dt_proj.weight.T + mixer.dt_proj.bias)
        delta_per_strato[i] = delta[0].detach()        # (token, 1536)
    return hook

maniglie = [layer.mixer.x_proj.register_forward_hook(crea_hook(i, layer.mixer))
            for i, layer in enumerate(strati)]

# 3. Faccio girare il modello sul testo: gli hook raccolgono Delta
with torch.no_grad():
    model(ids, use_cache=False)
for m in maniglie:
    m.remove()

# 4. Riassumo Delta strato per strato
print(f"{'strato':>6} {'Delta_min':>11} {'Delta_mediana':>14} {'Delta_max':>11} {'valori':>9}")
print("-" * 55)
for i in range(len(strati)):
    d = delta_per_strato[i]
    print(f"{i:>6} {d.min().item():>11.3e} {d.median().item():>14.3e} "
          f"{d.max().item():>11.3e} {d.numel():>9}")
