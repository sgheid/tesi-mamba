import torch
import torch.nn.functional as F


# Sonda su x_proj di ogni strato: il modello legge il testo e ogni hook riceve l'uscita
def esegui_con_sonde(model, ids, crea_hook):
    maniglie = [layer.mixer.x_proj.register_forward_hook(crea_hook(i, layer.mixer))
                for i, layer in enumerate(model.backbone.layers)]
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for m in maniglie:
            m.remove()


def ingresso_delta(mixer, uscita_x_proj):
    return uscita_x_proj[0, :, :mixer.time_step_rank]


# Delta = softplus(W_dt · ts + b_dt), ricalcolato come nel modello: forma (token, canali)
def _delta(mixer, ts, tipo):
    return F.softplus(ts.to(tipo) @ mixer.dt_proj.weight.to(tipo).T + mixer.dt_proj.bias.to(tipo))


def delta_fp32(mixer, ts):
    return _delta(mixer, ts, torch.float32)


def delta_fp64(mixer, ts):
    return _delta(mixer, ts, torch.float64)


# Delta nel formato del modello, con le stesse operazioni di transformers (percorso PyTorch):
# matmul senza bias, bias convertito al formato e sommato, softplus nel formato. Forma (token, canali)
def delta_modello(mixer, uscita_x_proj):
    ts = uscita_x_proj[..., :mixer.time_step_rank]
    t = mixer.dt_proj.weight @ ts.transpose(1, 2)
    t = t + mixer.dt_proj.bias.float().to(t.dtype)[..., None]
    return F.softplus(t)[0].T
