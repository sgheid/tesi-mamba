"""
Fase 1 - Autovalori a tempo continuo di Mamba-130M, letti dai pesi in FP32.

Uso:   python fase1_autovalori_continui.py
"""
import torch
from transformers import MambaForCausalLM

# 1. Carico il modello e mi assicuro che sia in FP32
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf").float()
strati = model.backbone.layers
print(f"Strati: {len(strati)}   formato di A_log: {strati[0].mixer.A_log.dtype}\n")

print(f"{'strato':>6} {'a_min':>10} {'a_mediana':>10} {'a_max':>12} {'non stabili':>12}")
print("-" * 54)

totale, non_stabili_totali = 0, 0

# 2. Per ogni strato leggo A_log e calcolo gli autovalori
for i, layer in enumerate(strati):
    A_log = layer.mixer.A_log.detach()      # 1536 canali x 16 stati
    a = -torch.exp(A_log)                   # autovalori continui

    # 3. Riassumo lo strato con tre numeri e conto i modi non stabili (a >= 0)
    a_min = a.min().item()                  # il piu' negativo: dimentica piu' in fretta
    a_med = a.median().item()               # valore tipico
    a_max = a.max().item()                  # il piu' vicino a zero: memoria piu' lunga
    non_stabili = (a >= 0).sum().item()

    print(f"{i:>6} {a_min:>10.3e} {a_med:>10.3e} {a_max:>12.3e} {non_stabili:>12}")
    totale += a.numel()
    non_stabili_totali += non_stabili

# 4. Verdetto finale
print(f"\nAutovalori totali: {totale}")
print(f"Non stabili (a >= 0): {non_stabili_totali}")
