# Verifica della correzione del caricamento: confronta il codice attuale del progetto (analisi-mamba) con la copia
# corretta in _verifica/nuovo/analisi-mamba. Scrive verifica_log.txt in questa cartella.
import json, os, subprocess, sys, time

QUI = os.path.dirname(os.path.abspath(__file__))
RADICE = os.path.dirname(QUI)
CODICE = {"vecchio": os.path.join(RADICE, "analisi-mamba"), "nuovo": os.path.join(QUI, "nuovo", "analisi-mamba")}
G = 2 ** 30
LOG = os.path.join(QUI, "verifica_log.txt")


def figlio(prova, versione, nome_tipo, uscita):
    sys.path.insert(0, CODICE[versione])
    import psutil, torch
    from nucleo import modelli, precisione
    tipo = dict(FP32=torch.float32, FP16=torch.float16, BF16=torch.bfloat16)[nome_tipo]
    m14 = [m for m in modelli.trova_modelli(modelli.cartella_cache_predefinita()) if "1.4b" in m.nome.lower()][0]
    torch.zeros(1, device="cuda")
    usato = {"diretto": False}
    if hasattr(modelli, "_carica_diretto"):
        originale = modelli._carica_diretto
        def spia(*a, **k):
            r = originale(*a, **k); usato["diretto"] = True; return r
        modelli._carica_diretto = spia
    esito = {}
    t0 = time.time()
    if prova == "carica":
        model = modelli._carica_pesi(m14.percorso, "cuda", tipo).eval()
    elif prova == "carica_e_verifica":
        model, tok, info, controlli = modelli.carica_e_verifica(m14, "cuda")
        esito["controlli"] = [f"{'OK' if c.ok else 'NO'} {c.descrizione} {c.dettaglio}" for c in controlli]
    elif prova == "rileggi":
        model = modelli._carica_pesi(m14.percorso, "cuda", tipo).eval()
        prima = precisione.impronta(model)
        for p in model.parameters():
            p.data = torch.empty(0, dtype=p.dtype, device="cuda")
        modelli.libera()
        precisione._rileggi_pesi(model, m14.percorso, "cuda")
        esito["rilettura_identica"] = precisione.impronta(model) == prima
    esito["secondi"] = time.time() - t0
    mi = psutil.Process().memory_info()
    esito.update(picco=mi.peak_pagefile / G, commit=mi.private / G, diretto=usato["diretto"])
    if model is not None and prova == "carica":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m14.percorso, local_files_only=True)
        ids = tok("Control theory studies how dynamical systems evolve over time.", return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(ids, use_cache=False).logits.float().cpu()
        bits = {torch.float32: torch.int32}.get(tipo, torch.int16)
        torch.save(dict(impronta=[(n, int(p.view(bits).sum(dtype=torch.int64).item()), p.double().sum().item())
                                  for n, p in model.named_parameters()], logits=logits), uscita + ".pt")
    json.dump(esito, open(uscita + ".json", "w"))


if len(sys.argv) > 1:
    figlio(*sys.argv[1:5]); sys.exit()

import torch
f = open(LOG, "w", encoding="utf-8")
def log(*a):
    s = " ".join(map(str, a)); print(s); f.write(s + "\n"); f.flush()

def esegui(prova, versione, nome_tipo="FP32"):
    out = os.path.join(QUI, f"_{prova}_{versione}_{nome_tipo}")
    r = subprocess.run([sys.executable, __file__, prova, versione, nome_tipo, out], capture_output=True, text=True)
    if r.returncode or not os.path.exists(out + ".json"):
        log(f"  {prova} {versione} {nome_tipo}: ERRORE {r.returncode}\n{r.stderr[-2500:]}")
        return None, None
    j = json.load(open(out + ".json")); os.remove(out + ".json")
    pt = None
    if os.path.exists(out + ".pt"):
        pt = torch.load(out + ".pt"); os.remove(out + ".pt")
    return j, pt

log("=== VERIFICA", time.strftime("%Y-%m-%d %H:%M:%S"), "===")
log("\n--- 1. Caricamento del 1.4B sulla GPU: codice attuale contro corretto ---")
for nome_tipo in ("FP32", "FP16", "BF16"):
    jv, pv = esegui("carica", "vecchio", nome_tipo)
    jn, pn = esegui("carica", "nuovo", nome_tipo)
    if jv and jn:
        log(f"  {nome_tipo}: picco commit attuale {jv['picco']:.2f} GB -> corretto {jn['picco']:.2f} GB "
            f"(caricamento diretto usato: {jn['diretto']}), tempo {jv['secondi']:.0f} s -> {jn['secondi']:.0f} s")
        log(f"        pesi identici bit a bit: {pv['impronta'] == pn['impronta']}, logit identici: {torch.equal(pv['logits'], pn['logits'])}")

log("\n--- 2. carica_e_verifica (come nell'app) col codice corretto ---")
j, _ = esegui("carica_e_verifica", "nuovo")
if j:
    log(f"  picco commit {j['picco']:.2f} GB, diretto {j['diretto']}, {j['secondi']:.0f} s")
    for c in j["controlli"]:
        log("   ", c)

log("\n--- 3. Rilettura dei pesi FP32 dal disco (modo 'disco' dei confronti) col codice corretto ---")
j, _ = esegui("rileggi", "nuovo")
if j:
    log(f"  pesi riletti identici: {j['rilettura_identica']}, picco commit {j['picco']:.2f} GB")

def fotografa(versione, modello, uscita, altro=()):
    script = os.path.join(CODICE[versione], "prove", "fotografa_fasi.py")
    r = subprocess.run([sys.executable, script, uscita, "--gpu", "--modello", modello, *altro],
                       capture_output=True, text=True, cwd=CODICE[versione])
    return r.returncode, (r.stdout + ("\n" + r.stderr[-2500:] if r.returncode else "")).strip()

for modello, fasi in (("130m", None), ("1.4b", "1")):
    etichetta = f"{modello}" + (f", Fase {fasi}" if fasi else ", tutte le fasi")
    log(f"\n--- 4. Fotografia delle fasi ({etichetta}, GPU): attuale contro corretto ---")
    ov, on = os.path.join(QUI, f"foto_{modello}_attuale"), os.path.join(QUI, f"foto_{modello}_corretto")
    extra = ("--fasi", fasi) if fasi else ()
    t0 = time.time()
    c, testo = fotografa("vecchio", modello, ov, extra)
    log(f"  attuale: codice {c}, {time.time()-t0:.0f} s")
    if c:
        log(testo); continue
    t0 = time.time()
    c, testo = fotografa("nuovo", modello, on, extra + ("--confronta", ov))
    log(f"  corretto: codice {c}, {time.time()-t0:.0f} s")
    log("\n".join("    " + r for r in testo.splitlines()))
log("\n=== FINE ===")
f.close()
