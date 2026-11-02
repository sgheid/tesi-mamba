import numpy as np

# Regimi di memoria sul testo (criterio L/tau, Capitolo 2): x = |a|·theta, con theta = somma dei Delta del canale sul
# testo. x > REGIME_ALTO: il modo dimentica entro il testo; x < REGIME_BASSO: conserva più del 90% dello stato
# (integratore di fatto sul testo); in mezzo il regime critico. Usati dalle Fasi 3 e 5
REGIME_BASSO, REGIME_ALTO = 0.1, 10.0

# Istogramma dell'errore relativo sul margine 1 - abar: log10 da -12 a 2 a centesimi di decade (per la mediana).
# Usato dalla Fase 3 (pesi ridotti) e dalla Fase 5 (abar arrotondato)
ERR_MIN, ERR_MAX, ERR_PASSO = -12.0, 2.0, 0.01
N_ERR = int(round((ERR_MAX - ERR_MIN) / ERR_PASSO))


# Mediana dall'istogramma: errori nulli contati a parte, poi la cumulata dei bin (centro del bin, in log)
def mediana_errore(zeri, isto):
    totale = zeri + isto.sum()
    if totale == 0:
        return float("nan")
    if zeri >= totale / 2:
        return 0.0
    k = int(np.searchsorted(np.cumsum(isto), totale / 2 - zeri))
    return 10 ** (ERR_MIN + (k + 0.5) * ERR_PASSO)
