# Diagnostica di B2 < B1 (Fase 6, fase extra): quale arrotondamento in più di B2 riduce l'errore del caso limite.
# Due ipotesi:
#   (a) assorbimento: con lo stato nel formato ridotto, un incremento più piccolo di metà spaziatura attorno a h va perso;
#   (b) azzeramento: deltaB_u nel formato ridotto va a 0 per underflow, e polo 1 con ingresso 0 è un modo congelato.
# Due strumenti, entrambi con i pesi in FP32:
#   1. varianti intermedie end-to-end (ablazione): B1 + stato ridotto, B1 + ingresso ridotto, confrontate con B1 e B2;
#   2. un passaggio FP32 con ricorrenze "ombra": in ogni strato, sugli stessi ingressi del modello FP32, girano in parallelo
#      le quattro ricorrenze ridotte; si misurano incrementi assorbiti/azzerati e la deriva dello stato rispetto a FP32.
#      L'uscita del modello resta quella FP32 (verificato a ogni esecuzione), quindi l'errore di uno strato non si
#      propaga ai successivi: è l'effetto locale della ricorrenza, strato per strato.
# nucleo/ricorrenza.py (B1 e B2 della Fase 5) non è toccato: qui B1 e B2 end-to-end si prendono da lì.
# La ricorrenza generica di questo file, con (stato, ingresso) = (FP32, FP32) e (ridotto, ridotto), dà risultati
# identici bit a bit a B1 e B2 della Fase 5 (prove/prova_diagnostica_b2.py).
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from nucleo import ricorrenza as rc
from nucleo.margini import REGIME_BASSO
from nucleo.scansione import discretizza, token_per_blocco

# nome -> (stato ridotto, ingresso deltaB_u ridotto); abar è sempre arrotondato al formato
VARIANTI = {"B1": (False, False), "B1 + stato": (True, False), "B1 + ingresso": (False, True), "B2": (True, True)}
INTERMEDIE = ("B1 + stato", "B1 + ingresso")
# Varianti con lo stato ridotto: le uniche in cui un incremento può essere assorbito dalla somma
CON_STATO = tuple(v for v, (s, _) in VARIANTI.items() if s)


def _aggiorna(stato, abar_r, inc, stato_ridotto, tipo):
    if stato_ridotto:
        return (abar_r.float() * stato.float() + inc.float()).to(tipo)
    return abar_r.float() * stato + inc


def _prepara(hidden_states, dt, B, C, delta_bias, delta_softplus):
    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = F.softplus(dt)
    return dt, B.transpose(1, 2), C.transpose(1, 2)


def _chiudi(scan_outputs, hidden_states, D, z):
    scan_output = torch.stack(scan_outputs, dim=-1)
    if D is not None:
        scan_output = scan_output + hidden_states * D[None, :, None]
    if z is not None:
        scan_output = scan_output * F.silu(z)
    return scan_output


# Ricorrenza ridotta generica (end-to-end): abar arrotondato a `tipo`, stato e/o deltaB_u ridotti secondo la variante.
# Stesse operazioni di nucleo/ricorrenza.py: (FP32, FP32) è B1, (ridotto, ridotto) è B2
@contextmanager
def ricorrenza_variante(tipo, variante):
    stato_ridotto, ingresso_ridotto = VARIANTI[variante]
    modulo = rc._modulo_mamba()
    originale = modulo.mamba_selective_scan
    strati = []

    def sostituta(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                  return_last_state=False, use_mambapy=False, use_associative_scan=False, **kwargs):
        if return_last_state or use_mambapy:
            raise RuntimeError("Ricorrenza simulata: previsto solo il passaggio senza cache e senza mambapy")
        batch_size, intermediate_size, seq_len = hidden_states.shape
        input_dtype = hidden_states.dtype
        dt, B, C = _prepara(hidden_states, dt, B, C, delta_bias, delta_softplus)
        passo = token_per_blocco(hidden_states, A.shape[-1])
        stato = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=tipo if stato_ridotto else input_dtype,
                            device=hidden_states.device)
        non_finiti = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
        scan_outputs = []
        for s in range(0, seq_len, passo):
            e = min(s + passo, seq_len)
            discrete_A, deltaB_u = discretizza(hidden_states, dt, A, B, s, e)
            abar_r = discrete_A.to(tipo)
            del discrete_A
            if ingresso_ridotto:
                deltaB_u = deltaB_u.to(tipo)
            for j in range(e - s):
                stato = _aggiorna(stato, abar_r[:, :, j], deltaB_u[:, :, j], stato_ridotto, tipo)
                non_finiti += (~torch.isfinite(stato)).sum()
                scan_output = torch.matmul(stato.to(input_dtype), C[:, s + j, :].unsqueeze(-1))
                scan_outputs.append(scan_output[:, :, 0])
            del abar_r, deltaB_u
        strati.append(dict(non_finiti_stato=int(non_finiti.item())))
        return _chiudi(scan_outputs, hidden_states, D, z)

    modulo.mamba_selective_scan = sostituta
    try:
        yield strati
    finally:
        modulo.mamba_selective_scan = originale


def stato_finale_variante(ctx, tipo, variante):
    n = len(ctx.model.backbone.layers)
    with ricorrenza_variante(tipo, variante) as strati, torch.no_grad():
        h = ctx.model.backbone(ctx.ids, use_cache=False).last_hidden_state[0].cpu()
    if len(strati) != n:
        raise RuntimeError(f"La ricorrenza sostituita è stata usata in {len(strati)} strati su {n}.")
    return h, strati


# Contatori di un gruppo di aggiornamenti (passo × canale × stato): quanti, quanti persi del tutto, massa |incremento|
# e massa persa |incremento - incremento applicato|
def _contatori(dispositivo):
    z = lambda tipo=torch.int64: torch.zeros((), dtype=tipo, device=dispositivo)
    return dict(n=z(), assorbiti=z(), azzerati=z(), massa=z(torch.float64), persa=z(torch.float64))


def _somma(c, maschera, inc32, persa, assorbito, azzerato):
    c["n"] += maschera.sum()
    c["massa"] += torch.where(maschera, inc32.abs(), 0).double().sum()
    if persa is not None:
        c["persa"] += torch.where(maschera, persa, 0).double().sum()
        c["assorbiti"] += (assorbito & maschera).sum()
    if azzerato is not None:
        c["azzerati"] += (azzerato & maschera).sum()


# Passaggio FP32 con le ricorrenze ombra. Per ogni strato restituisce:
#   modi: modi (canale, stato) "integratori spuri rilevanti" S = almeno un passo con abar32 < 1 e abar_r = 1, e
#         |a|θ ≥ REGIME_BASSO (memoria non trascurabile sul testo, come le Fasi 3 e 5);
#   gruppi I (aggiornamenti in cui il polo è un integratore spurio: abar32 < 1, abar_r = 1) e O (tutti gli altri):
#         per le varianti con stato ridotto incrementi assorbiti e massa persa; per l'ingresso ridotto incrementi azzerati;
#   errore di stato su S, per passo: num[variante, t] = Σ_S |h_v - h32|, den[t] = Σ_S |h32| (sommabili tra strati).
@contextmanager
def ricorrenze_ombra(tipo):
    modulo = rc._modulo_mamba()
    originale = modulo.mamba_selective_scan
    strati = []

    def sostituta(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                  return_last_state=False, use_mambapy=False, use_associative_scan=False, **kwargs):
        if return_last_state or use_mambapy:
            raise RuntimeError("Ricorrenze ombra: previsto solo il passaggio senza cache e senza mambapy")
        batch_size, intermediate_size, seq_len = hidden_states.shape
        if batch_size != 1:
            raise RuntimeError("Ricorrenze ombra: previsto un solo testo")
        input_dtype = hidden_states.dtype
        disp = hidden_states.device
        dt, B, C = _prepara(hidden_states, dt, B, C, delta_bias, delta_softplus)
        passo = token_per_blocco(hidden_states, A.shape[-1])

        # Pre-passaggio: modi S. theta = somma dei Delta del canale sul testo (in FP64 come nucleo/ricorrenza.py)
        rilevante = dt[0].double().sum(dim=-1)[:, None] * (-A.double()) >= REGIME_BASSO
        spurio = torch.zeros_like(rilevante)
        for s in range(0, seq_len, passo):
            e = min(s + passo, seq_len)
            dA = torch.exp(A[None, :, None, :] * dt[:, :, s:e, None])
            spurio |= ((dA.to(tipo) == 1) & (dA < 1))[0].any(dim=1)
            del dA
        S = (spurio & rilevante)[None]

        stato32 = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype, device=disp)
        ombre = {v: torch.zeros_like(stato32, dtype=tipo if VARIANTI[v][0] else input_dtype) for v in VARIANTI}
        num = torch.zeros(len(VARIANTI), seq_len, dtype=torch.float64, device=disp)
        den = torch.zeros(seq_len, dtype=torch.float64, device=disp)
        gruppi = {v: dict(I=_contatori(disp), O=_contatori(disp)) for v in VARIANTI}
        scan_outputs = []
        for s in range(0, seq_len, passo):
            e = min(s + passo, seq_len)
            discrete_A, deltaB_u = discretizza(hidden_states, dt, A, B, s, e)
            for j in range(e - s):
                t = s + j
                a32, inc32 = discrete_A[:, :, j], deltaB_u[:, :, j]
                # Ricorrenza FP32 del modello: stesse operazioni di transformers, l'uscita resta questa
                stato32 = a32 * stato32 + inc32
                scan_output = torch.matmul(stato32.to(input_dtype), C[:, t, :].unsqueeze(-1))
                scan_outputs.append(scan_output[:, :, 0])

                abar_r = a32.to(tipo)
                I = (abar_r == 1) & (a32 < 1)
                inc_r = inc32.to(tipo)
                azzerato = (inc_r == 0) & (inc32 != 0)
                for k, (v, (stato_ridotto, ingresso_ridotto)) in enumerate(VARIANTI.items()):
                    h = ombre[v]
                    inc = inc_r if ingresso_ridotto else inc32
                    persa = assorbito = None
                    if stato_ridotto:
                        prodotto = abar_r.float() * h.float()
                        nuovo = (prodotto + inc.float()).to(tipo)
                        applicato = nuovo.float() - prodotto.to(tipo).float()
                        persa = (inc32 - applicato).abs()
                        assorbito = (inc != 0) & (nuovo == prodotto.to(tipo))
                        ombre[v] = nuovo
                    else:
                        ombre[v] = abar_r.float() * h + inc
                    if stato_ridotto or ingresso_ridotto:
                        for nome_g, m in (("I", I), ("O", ~I)):
                            _somma(gruppi[v][nome_g], m, inc32, persa, assorbito, azzerato if ingresso_ridotto else None)
                    num[k, t] = torch.where(S, (ombre[v].float() - stato32).abs(), 0).double().sum()
                den[t] = torch.where(S, stato32.abs(), 0).double().sum()
            del discrete_A, deltaB_u

        strati.append(dict(
            modi_totali=int(rilevante.numel()), modi_spuri=int(spurio.sum().item()), modi_S=int(S.sum().item()),
            num=num.cpu().numpy(), den=den.cpu().numpy(),
            gruppi={v: {g: {k: x.item() for k, x in c.items()} for g, c in gr.items()} for v, gr in gruppi.items()}))
        return _chiudi(scan_outputs, hidden_states, D, z)

    modulo.mamba_selective_scan = sostituta
    try:
        yield strati
    finally:
        modulo.mamba_selective_scan = originale


def passaggio_ombra(ctx, tipo):
    n = len(ctx.model.backbone.layers)
    with ricorrenze_ombra(tipo) as strati, torch.no_grad():
        h = ctx.model.backbone(ctx.ids, use_cache=False).last_hidden_state[0].cpu()
    if len(strati) != n:
        raise RuntimeError(f"Le ricorrenze ombra sono state usate in {len(strati)} strati su {n}.")
    return h, strati
