MAX_TOKEN_PREDEFINITO = 4096

# Testo di esempio fisso, per risultati ripetibili (211 token con il tokenizer di Mamba)
TESTO_ESEMPIO = (
    "Control theory studies how dynamical systems evolve over time and how their "
    "behaviour can be shaped by feedback. A linear system is described by a state "
    "vector that summarises the past, a matrix that governs how this state evolves, "
    "and an input that drives it. The eigenvalues of the state matrix decide whether "
    "the system forgets its initial condition, oscillates, or diverges. In discrete "
    "time, a system is stable when every eigenvalue lies strictly inside the unit "
    "circle; an eigenvalue close to one corresponds to a slow mode with a long memory, "
    "while an eigenvalue close to zero corresponds to a mode that forgets almost "
    "immediately. Engineers have used these ideas for decades to design autopilots, "
    "chemical plants and power grids. Recently, the same mathematics has appeared in "
    "language models, where a recurrent state must carry information across thousands "
    "of words. Whether such a model remembers the beginning of a long document depends, "
    "once again, on how close its eigenvalues are to the boundary of stability, and on "
    "whether the hardware can represent numbers that are almost, but not exactly, one."
)


# Tokenizza e tronca: restituisce (ids, token singoli, testo analizzato, n token originali)
def prepara(tok, testo, dispositivo, max_token=MAX_TOKEN_PREDEFINITO):
    ids = tok(testo, return_tensors="pt").input_ids
    n_originali = int(ids.shape[1])
    if n_originali > max_token:
        ids = ids[:, :max_token]
        testo = tok.decode(ids[0])
    token = [tok.decode([t]) for t in ids[0].tolist()]
    return ids.to(dispositivo), token, testo, n_originali
