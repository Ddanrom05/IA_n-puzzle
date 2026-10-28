"""
Utilidades para el rompecabezas deslizante n x n.

Representación: `bytes` de longitud n*n (inmutable, hasheable, 1 byte por ficha,
válido hasta n = 15 porque 15*15-1 = 224 < 256). Ventajas:
  - comparación y hash de estados en C (muy rápido)
  - 1 byte por ficha -> poca memoria por nodo
  - estado.index(0) localiza el espacio en blanco a velocidad C
"""
from functools import lru_cache
from pathlib import Path

MOVS = (("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1))


def leer_instancia(ruta):
    """Devuelve (n, inicial, meta) con los tableros como bytes."""
    lineas = [l.strip() for l in Path(ruta).read_text(encoding="utf-8").splitlines()
              if l.strip()]
    n = int(lineas[0])
    if len(lineas) != 1 + 2 * n:
        raise ValueError(f"Se esperaban {1 + 2 * n} líneas y hay {len(lineas)}.")

    def tablero(bloque):
        vals = [int(x) for fila in bloque for x in fila.split(",")]
        if sorted(vals) != list(range(n * n)):
            raise ValueError("El tablero debe contener exactamente 0..n*n-1.")
        return bytes(vals)

    return n, tablero(lineas[1:1 + n]), tablero(lineas[1 + n:])


@lru_cache(maxsize=None)
def tabla_vecinos(n):
    """VECINOS[pos_cero] -> tupla de (nueva_pos, letra) con la letra del movimiento del blanco."""
    tabla = []
    for pos in range(n * n):
        f, c = divmod(pos, n)
        lst = []
        for letra, df, dc in MOVS:
            nf, nc = f + df, c + dc
            if 0 <= nf < n and 0 <= nc < n:
                lst.append((nf * n + nc, letra))
        tabla.append(tuple(lst))
    return tuple(tabla)


def mover(estado, pos_cero, nueva_pos):
    b = bytearray(estado)
    b[pos_cero] = b[nueva_pos]
    b[nueva_pos] = 0
    return bytes(b)


def es_resoluble(n, inicial, meta):
    """
    Cada movimiento es una transposición (cambia la paridad de la permutación)
    y cambia en 1 la distancia Manhattan del blanco. Por tanto la meta es
    alcanzable sii paridad(permutación) == paridad(distancia del blanco).
    """
    N = n * n
    pos_meta = [0] * N
    for p, v in enumerate(meta):
        pos_meta[v] = p
    perm = [pos_meta[v] for v in inicial]
    visto = [False] * N
    ciclos = 0
    for i in range(N):
        if not visto[i]:
            ciclos += 1
            j = i
            while not visto[j]:
                visto[j] = True
                j = perm[j]
    f1, c1 = divmod(inicial.index(0), n)
    f2, c2 = divmod(pos_meta[0], n)
    dist = abs(f1 - f2) + abs(c1 - c2)
    return (N - ciclos) % 2 == dist % 2


def _paso(b, cero, n, m):
    f, c = divmod(cero, n)
    if m == "U":
        f -= 1
    elif m == "D":
        f += 1
    elif m == "L":
        c -= 1
    elif m == "R":
        c += 1
    else:
        raise ValueError(f"Movimiento inválido: {m}")
    if not (0 <= f < n and 0 <= c < n):
        raise ValueError("Movimiento fuera del tablero.")
    nueva = f * n + c
    b[cero] = b[nueva]
    b[nueva] = 0
    return nueva


def aplicar_camino(n, estado, camino):
    """Replica los movimientos del blanco y devuelve el estado final."""
    b = bytearray(estado)
    cero = b.index(0)
    for m in camino:
        cero = _paso(b, cero, n, m)
    return bytes(b)


def eliminar_ciclos(n, inicial, camino):
    """Recorta cualquier tramo del camino que regrese a un estado ya visitado."""
    b = bytearray(inicial)
    cero = b.index(0)
    estados = [bytes(b)]
    vistos = {estados[0]: 0}
    res = []
    for m in camino:
        cero = _paso(b, cero, n, m)
        s = bytes(b)
        if s in vistos:
            k = vistos[s]
            for e in estados[k + 1:]:
                del vistos[e]
            del estados[k + 1:]
            del res[k:]
        else:
            vistos[s] = len(estados)
            estados.append(s)
            res.append(m)
    return res
