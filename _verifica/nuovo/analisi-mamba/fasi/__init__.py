import importlib
import pkgutil
import re
import traceback


def _chiave_ordinamento(fase):
    m = re.match(r"(\d+)(.*)", fase.codice)
    return (int(m.group(1)), m.group(2)) if m else (999, fase.codice)


# Ogni modulo di fasi/ con una variabile FASE diventa una voce del menu (ignorati quelli che iniziano con "_")
def carica_fasi(ricarica=False):
    fasi, errori = [], []
    for modulo in pkgutil.iter_modules(__path__):
        if modulo.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{modulo.name}")
            if ricarica:
                mod = importlib.reload(mod)
            if hasattr(mod, "FASE"):
                fasi.append(mod.FASE)
        except Exception:
            errori.append((modulo.name, traceback.format_exc(limit=3)))
    return sorted(fasi, key=_chiave_ordinamento), errori
