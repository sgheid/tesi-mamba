"""
Fase 2b - Poli discreti a_bar = exp(Delta * a) di Mamba-130M sul testo fissato.
Calcolati come li calcola il modello (FP32) e con un riferimento esatto (FP64).

Uso:   python fase2b_poli_discreti.py
"""
import torch
import torch.nn.functional as F
from transformers import MambaForCausalLM, AutoTokenizer

MODELLO = "state-spaces/mamba-1.4b-hf"

# Stesso testo della Fase 2a
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

# 1. Carico modello e testo (come in Fase 2a)
tok = AutoTokenizer.from_pretrained(MODELLO)
model = MambaForCausalLM.from_pretrained(MODELLO).float().eval()
ids = tok(TESTO, return_tensors="pt").input_ids
strati = model.backbone.layers
print(f"Token nel testo: {ids.shape[1]}\n")

risultati = {}

# 2. La sonda: oltre a Delta, ora calcola i poli discreti
def crea_hook(i, mixer):
    def hook(modulo, ingresso, uscita):
        ts = uscita[0, :, :mixer.time_step_rank]          # (token, 48)
        W, b = mixer.dt_proj.weight, mixer.dt_proj.bias

        # 2a. Come fa il modello: tutto in FP32
        delta32 = F.softplus(ts.float() @ W.float().T + b.float())        # (token, 1536)
        a32 = -torch.exp(mixer.A_log.float())                             # (1536, 16)
        abar32 = torch.exp(delta32[:, :, None] * a32)                     # (token, 1536, 16)

        # 2b. Riferimento esatto: stessi dati in FP64.
        #     Calcolo direttamente il margine 1 - a_bar, senza passare da a_bar
        delta64 = F.softplus(ts.double() @ W.double().T + b.double())
        a64 = -torch.exp(mixer.A_log.double())
        margine64 = -torch.expm1(delta64[:, :, None] * a64)               # 1 - a_bar, sempre > 0

        risultati[i] = dict(
            margine_min=margine64.min().item(),                 # polo discreto piu' vicino a 1
            uguali_1=(abar32 == 1.0).sum().item(),              # poli che in FP32 sono finiti SU 1
            uguali_0=(abar32 == 0.0).sum().item(),              # poli che in FP32 sono finiti in 0
            maggiori_1=(abar32 > 1.0).sum().item(),             # poli fuori dal cerchio unitario (atteso: 0)
            totale=abar32.numel(),
        )
    return hook

maniglie = [layer.mixer.x_proj.register_forward_hook(crea_hook(i, layer.mixer))
            for i, layer in enumerate(strati)]

# 3. Faccio girare il modello sul testo
with torch.no_grad():
    model(ids, use_cache=False)
for m in maniglie:
    m.remove()

# 4. Tabella per strato
print(f"{'strato':>6} {'1-abar min':>11} {'memoria max':>12} {'abar=1 FP32':>12} {'%':>9} {'abar=0 FP32':>12}")
print("-" * 67)
tot_1, tot_0, tot_out, tot = 0, 0, 0, 0
for i in range(len(strati)):
    r = risultati[i]
    memoria = 1.0 / r["margine_min"]                   # in token
    perc = 100.0 * r["uguali_1"] / r["totale"]
    print(f"{i:>6} {r['margine_min']:>11.3e} {memoria:>12.2e} {r['uguali_1']:>12} "
          f"{perc:>8.5f}% {r['uguali_0']:>12}")
    tot_1 += r["uguali_1"]; tot_0 += r["uguali_0"]; tot_out += r["maggiori_1"]; tot += r["totale"]

print(f"\nPoli discreti totali: {tot}")
print(f"Uguali a 1 in FP32:   {tot_1}  ({100.0 * tot_1 / tot:.4f}%)")
print(f"Uguali a 0 in FP32:   {tot_0}  ({100.0 * tot_0 / tot:.4f}%)")
print(f"Fuori dal cerchio (> 1): {tot_out}")
