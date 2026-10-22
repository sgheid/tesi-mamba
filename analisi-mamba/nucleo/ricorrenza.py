import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from nucleo import grafica as g
from nucleo.margini import ERR_MAX, ERR_MIN, N_ERR, mediana_errore
from nucleo.precisione import delta_baseline

# Formati della ricorrenza (Fase 5): il controllo FP32 non compare nel menu, lo usa prove/fotografa_fasi.py
FORMATI_RICORRENZA = {"FP16": torch.float16, "BF16": torch.bfloat16}
CONTROLLO = "FP32 (controllo)"
TIPI = dict(FORMATI_RICORRENZA, **{CONTROLLO: torch.float32})

# Soglie attese con l'arrotondamento al pari più vicino. Verso 1: metà del passo sotto 1 (FP16 2^-11, BF16 2^-8);
# il caso esattamente a metà va a 1 perché 1 ha la mantissa pari. Verso 0: metà del più piccolo subnormale
# (FP16 2^-24, BF16 2^-133); il caso a metà va a 0
SOGLIE_ATTESE = {torch.float16: (2.0 ** -12, 2.0 ** -25), torch.bfloat16: (2.0 ** -9, 2.0 ** -134),
                 torch.float32: (0.0, 0.0)}

# Istogramma della memoria persa: log10(tau) da 0 a 16 a centesimi di decade (per la mediana)
TAU_MIN, TAU_MAX, TAU_PASSO = 0.0, 16.0, 0.01
N_TAU = int(round((TAU_MAX - TAU_MIN) / TAU_PASSO))


def _da_bit(b, dispositivo):
    return torch.tensor([b], dtype=torch.int32, device=dispositivo).view(torch.float32)


# Più piccolo valore FP32 in [0.5, 1] che il formato porta a 1 e più grande valore FP32 >= 0 che porta a 0, cercati
# per bisezione sui bit (per i float positivi l'ordine dei bit è quello dei valori), sul dispositivo indicato.
# Restituisce le soglie sul margine e sul valore: abar_r = 1  <=>  1 - abar32 <= soglia_1;  abar_r = 0  <=>  abar32 <= soglia_0
def soglie_verificate(tipo, dispositivo):
    uno = lambda b: bool((_da_bit(b, dispositivo).to(tipo) == 1).item())
    zero = lambda b: bool((_da_bit(b, dispositivo).to(tipo) == 0).item())
    basso, alto = 0x3F000000, 0x3F800000          # 0.5 non va a 1, 1.0 sì: cerco il primo che va a 1
    while alto - basso > 1:
        medio = (basso + alto) // 2
        basso, alto = (basso, medio) if uno(medio) else (medio, alto)
    primo_a_1 = _da_bit(alto, "cpu").double().item()
    basso, alto = 0, 0x3F800000                   # 0 va a 0, 1.0 no: cerco l'ultimo che va a 0
    while alto - basso > 1:
        medio = (basso + alto) // 2
        basso, alto = (medio, alto) if zero(medio) else (basso, medio)
    return 1.0 - primo_a_1, _da_bit(basso, "cpu").double().item()


def verifica_soglie(dispositivo):
    esito = {}
    for nome, tipo in TIPI.items():
        s1, s0 = soglie_verificate(tipo, dispositivo)
        esito[nome] = dict(soglia_1=s1, soglia_0=s0, attese=SOGLIE_ATTESE[tipo], ok=(s1, s0) == SOGLIE_ATTESE[tipo])
    return esito


def potenza(x):
    return "0" if x == 0 else f"2^{round(math.log2(x))}" if math.log2(x).is_integer() else f"{x:.3e}"


# Parte A: abar32 = exp(Delta·a) in FP32 come il modello, poi arrotondato a ciascun formato; a blocchi di token.
# Per strato e formato: nuovi su 1, nuovi a 0, non finiti, errore relativo sul margine (istogramma), memoria persa
def poli_arrotondati(ctx, nomi, soglie):
    n = len(ctx.model.backbone.layers)
    base = delta_baseline(ctx, lambda f, m="": ctx.avanzamento(0.5 * f, m))
    A_log = {i: l.mixer.A_log.detach().float() for i, l in enumerate(ctx.model.backbone.layers)}
    righe = {nome: [] for nome in nomi}
    riferimento = []
    for i in range(n):
        d = base.pop(i).to(ctx.dispositivo)
        a32 = -torch.exp(A_log[i])
        a64 = -torch.exp(A_log[i].double())
        L, D = d.shape
        passo = max(1, 4_000_000 // (D * a32.shape[1]))
        r32 = dict(strato=i, su_1=0, su_1_esatto=0, in_0=0, totale=L * a32.numel())
        c = {nome: dict(nuovi_1=0, nuovi_1_soglia=0, nuovi_0=0, nuovi_0_soglia=0, su_1=0, non_finiti=0,
                        err_nulli=0, err=torch.zeros(N_ERR, dtype=torch.float64, device=d.device),
                        tau=torch.zeros(N_TAU, dtype=torch.float64, device=d.device), tau_max=0.0)
             for nome in nomi}
        for s in range(0, L, passo):
            ab32 = torch.exp(d[s:s + passo, :, None] * a32)
            m64 = -torch.expm1(d[s:s + passo].double()[:, :, None] * a64)
            uno32, zero32 = ab32 == 1.0, ab32 == 0.0
            m32 = 1.0 - ab32          # esatto per abar32 in [0.5, 1] (Sterbenz)
            r32["su_1"] += int(uno32.sum().item())
            r32["su_1_esatto"] += int((m64 < g.SOGLIA_FP32).sum().item())
            r32["in_0"] += int(zero32.sum().item())
            for nome in nomi:
                k, (s1, s0) = c[nome], soglie[nome]
                ar = ab32.to(TIPI[nome])
                finito = torch.isfinite(ar)
                uno_r = ar == 1.0
                nuovi = uno_r & ~uno32
                k["su_1"] += int(uno_r.sum().item())
                k["nuovi_1"] += int(nuovi.sum().item())
                k["nuovi_1_soglia"] += int(((m32 <= s1) & (ab32 >= 0.5) & ~uno32).sum().item())
                k["nuovi_0"] += int(((ar == 0.0) & ~zero32).sum().item())
                k["nuovi_0_soglia"] += int(((ab32 <= s0) & ~zero32).sum().item())
                k["non_finiti"] += int((~finito).sum().item())
                # Errore relativo sul margine (1 - abar_r) / (1 - abar32) - 1 in FP64; per i poli nuovi su 1 vale -1
                valido = finito & ~uno32
                err = ((1.0 - ar.double()) / (1.0 - ab32.double()) - 1.0)[valido].abs()
                k["err_nulli"] += int((err == 0).sum().item())
                lg = torch.log10(err[err > 0]).clamp(ERR_MIN, ERR_MAX - 1e-9)
                k["err"] += torch.histc(lg.float(), bins=N_ERR, min=ERR_MIN, max=ERR_MAX).double()
                # Memoria persa: tau = 1 / margine esatto FP32 dei poli nuovi su 1
                if nuovi.any():
                    tau = 1.0 / m64[nuovi]
                    k["tau_max"] = max(k["tau_max"], tau.max().item())
                    lt = torch.log10(tau).clamp(TAU_MIN, TAU_MAX - 1e-9)
                    k["tau"] += torch.histc(lt.float(), bins=N_TAU, min=TAU_MIN, max=TAU_MAX).double()
        riferimento.append(r32)
        for nome in nomi:
            k = c[nome]
            err, tau = k.pop("err").cpu().numpy(), k.pop("tau").cpu().numpy()
            if not k["nuovi_1"]:
                k["tau_max"] = float("nan")
            righe[nome].append(dict(strato=i, **k, totale=r32["totale"], perc_nuovi_1=k["nuovi_1"] / r32["totale"],
                                    err_mediana=mediana_errore(k["err_nulli"], err), isto_err=err, isto_tau=tau,
                                    tau_mediana=mediana_tau(tau)))
        ctx.avanzamento(0.5 + 0.5 * (i + 1) / n, f"poli arrotondati, strato {i + 1} di {n}")
    return riferimento, righe


def mediana_tau(isto):
    totale = isto.sum()
    if totale == 0:
        return float("nan")
    k = int(np.searchsorted(np.cumsum(isto), totale / 2))
    return 10 ** (TAU_MIN + (k + 0.5) * TAU_PASSO)


# Parte B: ricorrenza simulata nel formato ridotto. In transformers 5.x il ciclo della ricorrenza non è un metodo del
# mixer ma la funzione di modulo modeling_mamba.mamba_selective_scan, che MambaMixer.forward cerca nel modulo a ogni
# chiamata: il context manager la sostituisce con una copia del ramo ricorrente (stesse operazioni, stesso ordine)
# con gli arrotondamenti inseriti, e la ripristina sempre all'uscita, anche dopo un errore.
#   B1: solo abar arrotondato al formato e riportato in FP32 (come nella Parte A);
#   B2: abar, deltaB_u e stato nel formato: a ogni passo stato = abar_r·stato + dBu_r calcolato in FP32 e arrotondato.
# Con il formato FP32 entrambi devono dare logit identici bit a bit al modello originale (prove/prova_ricorrenza.py).
LIVELLI = ("B1", "B2")


def _modulo_mamba():
    from transformers.models.mamba import modeling_mamba
    return modeling_mamba


@contextmanager
def ricorrenza_in_formato(tipo, livello):
    if livello not in LIVELLI:
        raise ValueError(f"Livello sconosciuto: {livello}")
    modulo = _modulo_mamba()
    originale = modulo.mamba_selective_scan
    # Per ogni strato chiamato, in ordine: poli nuovi su 1, valori non finiti dello stato, massimo |stato| finito
    strati = []

    def sostituta(hidden_states, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                  return_last_state=False, use_mambapy=False, use_associative_scan=False, **kwargs):
        if return_last_state or use_mambapy:
            raise RuntimeError("Ricorrenza simulata: previsto solo il passaggio senza cache e senza mambapy")
        batch_size, intermediate_size, seq_len = hidden_states.shape
        input_dtype = hidden_states.dtype

        if delta_bias is not None:
            dt = dt + delta_bias.to(dt.dtype)[..., None]
        if delta_softplus:
            dt = F.softplus(dt)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)

        discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
        discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        del discrete_B

        abar_r = discrete_A.to(tipo)
        nuovi = int(((abar_r == 1) & (discrete_A < 1)).sum().item())
        if livello == "B1":
            discrete_A = abar_r.to(discrete_A.dtype)
            ssm_state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype,
                                    device=hidden_states.device)
        else:
            del discrete_A
            deltaB_u = deltaB_u.to(tipo)
            ssm_state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=tipo,
                                    device=hidden_states.device)
        non_finiti = torch.zeros((), dtype=torch.int64, device=hidden_states.device)
        massimo = torch.zeros((), dtype=torch.float32, device=hidden_states.device)

        scan_outputs = []
        for index in range(seq_len):
            if livello == "B1":
                ssm_state = discrete_A[:, :, index] * ssm_state + deltaB_u[:, :, index]
            else:
                ssm_state = (abar_r[:, :, index].float() * ssm_state.float() + deltaB_u[:, :, index].float()).to(tipo)
            finito = torch.isfinite(ssm_state)
            non_finiti += (~finito).sum()
            massimo = torch.maximum(massimo, torch.where(finito, ssm_state, 0).abs().max().float())
            scan_output = torch.matmul(ssm_state.to(input_dtype), C[:, index, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)

        if D is not None:
            scan_output = scan_output + hidden_states * D[None, :, None]
        if z is not None:
            scan_output = scan_output * F.silu(z)
        strati.append(dict(nuovi_1=nuovi, non_finiti_stato=int(non_finiti.item()), stato_max=massimo.item()))
        return scan_output

    modulo.mamba_selective_scan = sostituta
    try:
        yield strati
    finally:
        modulo.mamba_selective_scan = originale


# Stato finale (uscita di norm_f) del modello FP32 condiviso con la ricorrenza simulata; controlla che la ricorrenza
# sostituita sia stata usata in ogni strato
def stato_finale_simulato(ctx, tipo, livello):
    n = len(ctx.model.backbone.layers)
    with ricorrenza_in_formato(tipo, livello) as strati, torch.no_grad():
        h = ctx.model.backbone(ctx.ids, use_cache=False).last_hidden_state[0].cpu()
    if len(strati) != n:
        raise RuntimeError(f"La ricorrenza sostituita è stata usata in {len(strati)} strati su {n}: transformers "
                           "chiama la ricorrenza altrove (kernel installati?), il risultato non sarebbe simulato.")
    return h, strati
