from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Strutture comuni: una fase riceve un Contesto e restituisce un Risultato
@dataclass
class Parametro:
    chiave: str
    etichetta: str
    scelte: list
    predefinito: Any = None
    aiuto: str = ""


@dataclass
class Fase:
    codice: str
    nome: str
    breve: str
    spiegazione: str
    usa_testo: bool
    esegui: Callable
    parametri: list = field(default_factory=list)


@dataclass
class Contesto:
    model: Any
    tok: Any
    info_modello: dict
    dispositivo: str
    nome_dispositivo: str
    ids: Any = None
    token: list = field(default_factory=list)
    testo: Optional[str] = None
    fonte_testo: Optional[str] = None
    parametri: dict = field(default_factory=dict)
    avanzamento: Callable = lambda frazione, messaggio="": None

    @property
    def n_token(self):
        return 0 if self.ids is None else int(self.ids.shape[1])


@dataclass
class Colonna:
    chiave: str
    titolo: str
    sottotitolo: str = ""
    formato: str = "sci"
    secondaria: bool = False


@dataclass
class Indicatore:
    etichetta: str
    valore: str
    nota: str = ""
    stato: Optional[str] = None
    secondario: bool = False


@dataclass
class Grafico:
    titolo: str
    figura: Any
    spiegazione: str = ""
    come_leggere: str = ""
    dati: Any = None
    secondario: bool = False


@dataclass
class Risultato:
    codice: str
    titolo: str
    descrizione: str
    colonne: list
    righe: list
    indicatori: list
    legenda: list
    grafici: list = field(default_factory=list)
    evidenzia: dict = field(default_factory=dict)
    note_formato: str = "FP32"
    info: dict = field(default_factory=dict)


def valori(righe, chiave, f=None):
    return [f(r[chiave]) if f else r[chiave] for r in righe]


# Indicatori che restano in primo piano anche se dichiarati secondari, quando segnalano un problema
ALLARMI = ("non finit", "fuori dal cerchio", "instabili", "controllo", "soglie", "coerenza", "conteggio",
           "formato dei pesi")


# Gerarchia dei risultati: i risultati fondamentali in primo piano, le misure di contorno in un riquadro compresso.
# indicatori: etichette; colonne: chiavi (valgono anche per la colonna affiancata con suffisso _r); grafici: inizio
# del titolo. Gli indicatori secondari vengono spostati dopo i principali, nello stesso ordine.
def gerarchia(ris, indicatori=(), colonne=(), grafici=()):
    for ind in ris.indicatori:
        allarme = ind.stato == "attenzione" and any(k in ind.etichetta.lower() for k in ALLARMI)
        ind.secondario = ind.etichetta in indicatori and not allarme
    ris.indicatori = [i for i in ris.indicatori if not i.secondario] + [i for i in ris.indicatori if i.secondario]
    for c in ris.colonne:
        base = c.chiave[:-2] if c.chiave.endswith("_r") else c.chiave
        c.secondaria = c.chiave in colonne or base in colonne
    for graf in ris.grafici:
        graf.secondario = any(graf.titolo.startswith(t) for t in grafici)
    ris.grafici = [x for x in ris.grafici if not x.secondario] + [x for x in ris.grafici if x.secondario]
    return ris
