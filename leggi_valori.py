"""Legge i valori control-teoretici di Mamba-130M su un testo reale:
Δ_t, ā = exp(Δ_t·A), margine di stabilità 1 − ā, quota di modi che FP16/BF16 arrotondano a 1."""
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, MambaForCausalLM

NOME = "state-spaces/mamba-130m-hf"
N_TOKEN = 512                                     # lunghezza del testo analizzato
SOGLIA = {"FP16": 1 - 2**-12, "BF16": 1 - 2**-9}  # sopra queste soglie ā viene arrotondato a 1.0

risultati = []


def crea_hook(i, mixer):
    A = -torch.exp(mixer.A_log.float())           # (1536, 16): autovalori continui, sempre < 0
    W, b = mixer.dt_proj.weight, mixer.dt_proj.bias
    R = mixer.time_step_rank

    def hook(modulo, ingresso, uscita):
        # uscita di x_proj: [time_step | B | C]; da time_step si ricava Δ_t come fa il modello
        delta = F.softplus(uscita[0, :, :R].float() @ W.float().T + b.float())   # (L, 1536)
        a_bar = torch.exp(delta[:, :, None] * A)                                   # (L, 1536, 16)
        r = {"strato": i,
             "|A| min": A.abs().min().item(),
             "Δ min": delta.min().item(),
             "Δ max": delta.max().item(),
             "margine min": 1 - a_bar.max().item()}
        for nome, s in SOGLIA.items():
            r[nome] = 100 * (a_bar > s).float().mean().item()
        risultati.append(r)
    return hook


def installa(model):
    for i, strato in enumerate(model.backbone.layers):
        strato.mixer.x_proj.register_forward_hook(crea_hook(i, strato.mixer))


def stampa():
    print(f"{'strato':>6} {'|A| min':>8} {'Δ min':>9} {'Δ max':>8} {'margine min':>12} {'%>FP16':>8} {'%>BF16':>8}")
    for r in risultati:
        print(f"{r['strato']:>6} {r['|A| min']:>8.4f} {r['Δ min']:>9.2e} {r['Δ max']:>8.3f} "
              f"{r['margine min']:>12.2e} {r['FP16']:>8.3f} {r['BF16']:>8.3f}")


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(NOME)
    model = MambaForCausalLM.from_pretrained(NOME).eval()
    installa(model)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("".join(ds["text"][:300]), return_tensors="pt")["input_ids"][:, :N_TOKEN]
    print(f"Testo analizzato: {ids.shape[1]} token di WikiText-2\n")

    with torch.no_grad():
        model(ids)
    stampa()
