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


@dataclass
class Indicatore:
    etichetta: str
    valore: str
    nota: str = ""
    stato: Optional[str] = None


@dataclass
class Grafico:
    titolo: str
    figura: Any
    spiegazione: str = ""
    come_leggere: str = ""
    dati: Any = None


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
