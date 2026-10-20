import math

import plotly.graph_objects as go

# Palette comune a tutte le fasi: blu per i dati, grigi caldi per assi e griglia, rosso/verde per gli stati
BLU = "#2a78d6"
BLU_CHIARO = "#86b6ef"
BLU_TENUE = "rgba(42, 120, 214, 0.14)"
ARANCIO = "#eb6834"
ROSSO_STATO = "#d03b3b"
ROSSO_TENUE = "rgba(208, 59, 59, 0.08)"
VERDE_STATO = "#0ca30c"
GRIGIO_ZONA = "rgba(137, 135, 129, 0.13)"

INCHIOSTRO = "#0b0b0b"
INCHIOSTRO_2 = "#52514e"
TENUE = "#898781"
GRIGLIA = "#ebeae4"
ASSE = "#c3c2b7"
SUPERFICIE = "#ffffff"

SCALA_BLU = [[0.0, "#eef4fc"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
             [0.6, "#3987e5"], [0.75, "#256abf"], [0.9, "#184f95"], [1.0, "#0d366b"]]

CARATTERE = '"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif'
MONO = '"Cascadia Code", "JetBrains Mono", Consolas, monospace'

CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "grafico_mamba"},
}

# In FP32 un x < 1 diventa 1 se 1 - x < 2^-25 (metà della distanza tra 1 e il float precedente)
SOGLIA_FP32 = 2.0 ** -25


def _aggiorna_asse(fig, asse, **impostazioni):
    (fig.update_yaxes if asse == "y" else fig.update_xaxes)(**impostazioni)


def _e_log(fig, asse):
    return (fig.layout.yaxis if asse == "y" else fig.layout.xaxis).type == "log"


# Figura base con lo stile comune
def nuova_figura(asse_x, asse_y, log_y=False, altezza=420, hover_unificato=True):
    fig = go.Figure()
    fig.update_layout(
        height=altezza,
        margin=dict(l=70, r=24, t=20, b=56),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family=CARATTERE, size=13, color=INCHIOSTRO_2),
        hovermode="x unified" if hover_unificato else "closest",
        hoverlabel=dict(bgcolor="white", bordercolor=BLU_CHIARO, font=dict(family=CARATTERE, color=INCHIOSTRO)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(color=INCHIOSTRO_2), bgcolor="rgba(0,0,0,0)"),
        bargap=0.12,
    )
    stile_asse = dict(showgrid=True, gridcolor=GRIGLIA, gridwidth=1, zeroline=False,
                      linecolor=ASSE, ticks="outside", tickcolor=ASSE, ticklen=4,
                      tickfont=dict(family=MONO, size=11.5, color=TENUE), title_font=dict(color=INCHIOSTRO_2))
    fig.update_xaxes(title_text=asse_x, **stile_asse)
    fig.update_yaxes(title_text=asse_y, **stile_asse)
    fig.update_xaxes(showgrid=False)
    if log_y:
        fig.update_yaxes(type="log", exponentformat="power", dtick=1)
    return fig


def asse_strati(fig, n_strati):
    passo = 1 if n_strati <= 24 else (2 if n_strati <= 48 else 4)
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=passo, range=[-0.6, n_strati - 0.4])


def asse_decadi(fig, asse, minimo, massimo):
    k_min, k_max = math.floor(minimo), math.ceil(massimo)
    valori = list(range(k_min, k_max + 1, 1 if k_max - k_min <= 16 else 2))
    _aggiorna_asse(fig, asse, tickmode="array", tickvals=valori, ticktext=[f"10<sup>{k}</sup>" for k in valori])


# Etichetta di zona o soglia: sugli assi log Plotly vuole la posizione già in log10
def _etichetta(fig, asse, valore, testo, colore, ancora_x, ancora_y, x=None, y=None):
    pos = math.log10(valore) if _e_log(fig, asse) else valore
    posizione = (dict(x=x, xref="paper", y=pos, yref="y") if asse == "y"
                 else dict(x=pos, xref="x", y=y, yref="paper"))
    fig.add_annotation(**posizione, text=testo, showarrow=False, xanchor=ancora_x, yanchor=ancora_y,
                       font=dict(size=11, color=colore))


def zona(fig, asse, da, a, testo, colore=GRIGIO_ZONA):
    stile = dict(fillcolor=colore, line_width=0, layer="below")
    if asse == "y":
        fig.add_hrect(y0=da, y1=a, **stile)
        _etichetta(fig, "y", a, " " + testo, INCHIOSTRO_2, "left", "top", x=0)
    else:
        fig.add_vrect(x0=da, x1=a, **stile)
        _etichetta(fig, "x", da, " " + testo, INCHIOSTRO_2, "left", "top", y=1)


def linea_soglia(fig, asse, valore, testo, colore=ROSSO_STATO):
    stile = dict(line=dict(color=colore, width=1.5, dash="dash"))
    if asse == "y":
        fig.add_hline(y=valore, **stile)
        _etichetta(fig, "y", valore, testo + " ", colore, "right", "bottom", x=1)
    else:
        fig.add_vline(x=valore, **stile)
        _etichetta(fig, "x", valore, " " + testo, colore, "left", "top", y=0.93)


def scala_log(fig, asse, minimo, massimo, margine=0.3):
    lo = math.floor(math.log10(minimo)) - margine
    hi = math.ceil(math.log10(massimo)) + margine
    decadi = hi - lo
    _aggiorna_asse(fig, asse, range=[lo, hi], dtick=1 if decadi <= 10 else (2 if decadi <= 20 else 3))


def marcatore(dimensione=7):
    return dict(size=dimensione, color=BLU, line=dict(color=SUPERFICIE, width=1.5))


# Fascia min–max con la mediana in evidenza; nomi = (bordo superiore, mediana, bordo inferiore)
def fascia_min_mediana_max(fig, x, y_min, y_med, y_max, nomi, unita="", formato=".2e"):
    sup, med, inf = nomi
    hover = lambda nome: f"%{{y:{formato}}}{unita}<extra>{nome}</extra>"
    fig.add_trace(go.Scatter(x=x, y=y_max, name=sup, mode="lines",
                             line=dict(color=BLU_CHIARO, width=1.5), hovertemplate=hover(sup)))
    fig.add_trace(go.Scatter(x=x, y=y_min, name=inf, mode="lines",
                             line=dict(color=BLU_CHIARO, width=1.5, dash="dot"),
                             fill="tonexty", fillcolor=BLU_TENUE, hovertemplate=hover(inf)))
    fig.add_trace(go.Scatter(x=x, y=y_med, name=med, mode="lines+markers",
                             line=dict(color=BLU, width=2.5),
                             marker=marcatore(), hovertemplate=hover(med)))
