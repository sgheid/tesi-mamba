import math


def _non_finito(x):
    return isinstance(x, float) and not math.isfinite(x)


# Formattazione dei numeri: identica a schermo, nelle tabelle e nell'Excel
def fmt_sci(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if _non_finito(x):
        return "infinito"
    return f"{x:.2e}"


def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


def fmt_perc(p):
    if p == 0:
        return "0 %"
    x = p * 100
    decimali = max(0, 2 - math.floor(math.log10(abs(x))))
    return f"{x:.{decimali}f} %"


def fmt_dec(x):
    return "-" if x is None or _non_finito(x) else f"{x:.1f}"


def fmt_testo(x):
    return "" if x is None else str(x)


# formato -> (funzione per lo schermo, formato numerico di Excel)
FORMATI = {
    "sci": (fmt_sci, "0.00E+00"),
    "int": (fmt_int, "#,##0"),
    "perc": (fmt_perc, "0.00000%"),
    "dec": (fmt_dec, "0.0"),
    "testo": (fmt_testo, "@"),
}
