import os

import torch
import torch.nn.functional as F

# Ricorrenza di transformers a blocchi di token, per stare nella VRAM col 1.4B.
# modeling_mamba.mamba_selective_scan (ramo ricorrente, transformers 5.17) calcola discrete_A, discrete_B e deltaB_u
# per l'intera sequenza: tre tensori FP32 (canali × token × stati), col 1.4B a 2048 token 512 MB ciascuno, più il
# prodotto A·Δ prima dell'exp. Con i pesi FP32 (5,5 GB) sulla GPU da 8 GB non ci stanno.
# Qui le stesse operazioni, nello stesso ordine, sono fatte un blocco di token alla volta: ogni elemento di
# discrete_A e deltaB_u è calcolato con la stessa operazione elemento per elemento, e il ciclo sui token è identico,
# quindi stato e uscita sono identici bit a bit (verifica: prove/prova_blocchi.py).
# I blocchi si usano solo sulla GPU e solo se un tensore intero supererebbe SOGLIA_BYTE: il 130M fino a 2048 token
# e ogni esecuzione su CPU passano dalla funzione originale.
SOGLIA_BYTE = 256 * 2 ** 20
BLOCCO_BYTE = 64 * 2 ** 20
# Per le prove: impone i token per blocco su ogni dispositivo (es. set MAMBA_TOKEN_PER_BLOCCO=50)
VARIABILE_PROVA = "MAMBA_TOKEN_PER_BLOCCO"

_ORIGINALE = None


# Token per blocco: seq_len (nessun blocco) se il tensore intero entra nella soglia
def token_per_blocco(hidden_states, n_stati):
    batch, canali, seq_len = hidden_states.shape
    forzato = os.environ.get(VARIABILE_PROVA, "").strip()
    if forzato:
        return max(1, int(forzato))
    if hidden_states.device.type != "cuda":
        return seq_len
    per_token = batch * canali * n_stati * 4
    if per_token * seq_len <= SOGLIA_BYTE:
        return seq_len
    return max(1, BLOCCO_BYTE // per_token)


# Blocco di token [s, e) discretizzato come in transformers: discrete_A e deltaB_u (FP32, forma batch × canali × token × stati)
def discretizza(hidden_states, dt, A, B, s, e):
    dt_b = dt[:, :, s:e]
    discrete_A = torch.exp(A[None, :, None, :] * dt_b[:, :, :, None])
    discrete_B = dt_b[:, :, :, None] * B[:, None, s:e, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, s:e, None].float()
    return discrete_A, deltaB_u


def _a_blocchi(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
               return_last_state=False, use_mambapy=False, use_associative_scan=False, **kwargs):
    batch_size, intermediate_size, seq_len = hidden_states.shape
    passo = token_per_blocco(hidden_states, A.shape[-1])
    if passo >= seq_len or use_mambapy or torch.compiler.is_compiling():
        return _ORIGINALE(hidden_states, dt, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus,
                          return_last_state=return_last_state, use_mambapy=use_mambapy,
                          use_associative_scan=use_associative_scan, **kwargs)
    input_dtype = hidden_states.dtype

    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = F.softplus(dt)
    B = B.transpose(1, 2)
    C = C.transpose(1, 2)

    ssm_state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype, device=hidden_states.device)
    scan_outputs = []
    for s in range(0, seq_len, passo):
        e = min(s + passo, seq_len)
        discrete_A, deltaB_u = discretizza(hidden_states, dt, A, B, s, e)
        for j in range(e - s):
            ssm_state = discrete_A[:, :, j] * ssm_state + deltaB_u[:, :, j]
            scan_output = torch.matmul(ssm_state.to(input_dtype), C[:, s + j, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        del discrete_A, deltaB_u
    scan_output = torch.stack(scan_outputs, dim=-1)

    if D is not None:
        scan_output = scan_output + hidden_states * D[None, :, None]
    if z is not None:
        scan_output = scan_output * F.silu(z)
    if return_last_state:
        return scan_output, ssm_state
    return scan_output


# Sostituisce una volta sola la funzione di modulo (MambaMixer.forward la cerca nel modulo a ogni chiamata).
# nucleo/ricorrenza.py la sostituisce a sua volta solo dentro il suo context manager e poi rimette questa
def installa():
    global _ORIGINALE
    from transformers.models.mamba import modeling_mamba
    if _ORIGINALE is None and hasattr(modeling_mamba, "mamba_selective_scan"):
        _ORIGINALE = modeling_mamba.mamba_selective_scan
        modeling_mamba.mamba_selective_scan = _a_blocchi
