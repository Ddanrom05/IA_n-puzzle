"""
BFS-IDA* con memoria, tabú y control de profundidad (Algoritmos 1 y 2 de tu documento).

Estructuras de datos
  info      : dict estado -> (g, padre, movimiento, h_parcial)   (Open ∪ Closed; sirve de tabú
              y de memoria de caminos; el h_parcial se cachea para no recalcularlo)
  abierta   : heap (f, desempate, estado, g)       nodos con f <= umbral
  reserva   : heap (f, desempate, estado, g)       "NodosExcedidos": f > umbral, se conservan
              para la siguiente iteración (persistencia entre iteraciones IDA*)
  cerrados  : set de estados ya expandidos
Las entradas de los heaps se invalidan perezosamente (si el estado ya está cerrado o
se encontró un camino mejor).

Control de memoria
  - prof_max : profundidad máxima por tanda (relativa a la raíz actual). Al agotarse,
               se reinicia desde la mejor hoja (menor f) y se descartan info/heaps.
  - max_nodos: tope duro de nodos en memoria; si se rebasa, reinicia desde el nodo
               de menor f en ese momento.

Uso:
  python ida_estrella.py instances/5x5/facil/001_5_10.txt
  python ida_estrella.py archivo.txt --tiempo 120 --prof-max 100 --escalar-gmax --detalle
"""
import argparse
import heapq
import itertools
import sys
import time

from puzzle_utils import (leer_instancia, tabla_vecinos, mover, es_resoluble,
                          aplicar_camino, eliminar_ciclos)
from heuristica import FuncionEvaluacion

INF = float("inf")


def reconstruir(info, estado):
    movs = []
    while True:
        _, padre, m, _ = info[estado]
        if padre is None:
            break
        movs.append(m)
        estado = padre
    movs.reverse()
    return movs


def resolver(n, inicial, meta, prof_max=100, limite_seg=None, max_nodos=2_000_000,
             gmax=100.0, escalar_gmax=False, pesos=None):
    """
    Devuelve (camino | None, estatus, stats).
    estatus: RESUELTO | TIMEOUT | SIN_SOLUCION (paridad) | AGOTADO
    """
    t0 = time.perf_counter()
    limite = t0 + limite_seg if limite_seg else INF
    stats = {"expandidos": 0, "generados": 0, "reinicios": 0, "iteraciones": 0,
             "max_nodos_mem": 0, "tiempo": 0.0}

    def fin(camino, estatus):
        stats["tiempo"] = time.perf_counter() - t0
        return camino, estatus, stats

    if inicial == meta:
        return fin([], "RESUELTO")
    if not es_resoluble(n, inicial, meta):
        return fin(None, "SIN_SOLUCION")

    if escalar_gmax:                       # mantiene el balance g/h al crecer n
        gmax = gmax * (n / 5.0) ** 3
    fe = FuncionEvaluacion(n, meta, pesos=pesos, gmax=gmax)
    vecinos = tabla_vecinos(n)

    camino_global = []
    raices = {inicial}
    actual = inicial

    while True:                                           # ---- Algoritmo 1: reinicios
        stats["reinicios"] += 1
        hp0 = fe.h_parcial(actual)
        info = {actual: (0, None, None, hp0)}
        umbral = fe.f(hp0, 0)
        abierta = [(umbral, 0, actual, 0)]
        reserva = []
        cerrados = set()
        desempate = itertools.count(1)
        mejor_f, mejor_hoja = INF, None                   # (hoja, padre, letra)
        reiniciar_en = None

        while True:                                       # ---- bucle de umbrales
            stats["iteraciones"] += 1

            while abierta:                                # ---- Algoritmo 2: It_IDA
                _, _, x, gx = heapq.heappop(abierta)
                if x in cerrados or info[x][0] != gx:
                    continue                              # entrada obsoleta

                if x == meta:
                    camino = camino_global + reconstruir(info, x)
                    camino = eliminar_ciclos(n, inicial, camino)
                    assert aplicar_camino(n, inicial, camino) == meta
                    return fin(camino, "RESUELTO")

                stats["expandidos"] += 1
                if (stats["expandidos"] & 511) == 0:
                    if time.perf_counter() > limite:
                        return fin(None, "TIMEOUT")
                    if len(info) > stats["max_nodos_mem"]:
                        stats["max_nodos_mem"] = len(info)
                if len(info) > max_nodos:                 # tope de memoria
                    reiniciar_en = x
                    break

                cero = x.index(0)
                gh = gx + 1
                for nueva, letra in vecinos[cero]:
                    h = mover(x, cero, nueva)
                    ex = info.get(h)
                    if ex is not None and ex[0] <= gh:    # tabú / repetido sin mejora
                        continue
                    hp = ex[3] if ex is not None else fe.h_parcial(h)
                    fh = hp + fe.g_ponderado(gh)
                    stats["generados"] += 1

                    if gh > prof_max:                     # control de profundidad
                        if fh < mejor_f and h not in raices:
                            mejor_f, mejor_hoja = fh, (h, x, letra)
                        continue

                    info[h] = (gh, x, letra, hp)          # nuevo o camino más corto
                    cerrados.discard(h)
                    if fh > umbral:
                        heapq.heappush(reserva, (fh, next(desempate), h, gh))
                    else:
                        heapq.heappush(abierta, (fh, next(desempate), h, gh))
                cerrados.add(x)

            if reiniciar_en is not None:
                break

            # fin de la iteración: persistir memoria y subir el umbral
            while reserva and (reserva[0][2] in cerrados
                               or info[reserva[0][2]][0] != reserva[0][3]):
                heapq.heappop(reserva)
            if reserva:
                umbral = reserva[0][0]                    # mínimo f excedido
                while reserva and reserva[0][0] <= umbral:
                    e = heapq.heappop(reserva)
                    if e[2] not in cerrados and info[e[2]][0] == e[3]:
                        heapq.heappush(abierta, e)
                continue
            break                                          # profundidad máxima agotada

        # ---- reinicio desde la mejor hoja (o por tope de memoria)
        if reiniciar_en is not None:
            camino_global += reconstruir(info, reiniciar_en)
            actual = reiniciar_en
        elif mejor_hoja is not None:
            hoja, padre, letra = mejor_hoja
            camino_global += reconstruir(info, padre) + [letra]
            actual = hoja
        else:
            return fin(None, "AGOTADO")
        raices.add(actual)
        camino_global = eliminar_ciclos(n, inicial, camino_global)
        actual = aplicar_camino(n, inicial, camino_global)


def main():
    ap = argparse.ArgumentParser(description="BFS-IDA* para el rompecabezas n x n")
    ap.add_argument("archivo")
    ap.add_argument("--prof-max", type=int, default=100)
    ap.add_argument("--tiempo", type=float, default=180.0, help="límite en segundos")
    ap.add_argument("--max-nodos", type=int, default=2_000_000)
    ap.add_argument("--gmax", type=float, default=100.0)
    ap.add_argument("--escalar-gmax", action="store_true",
                    help="Gmax = gmax * (n/5)^3 para balancear g contra las h")
    ap.add_argument("--detalle", action="store_true", help="imprime estadísticas a stderr")
    a = ap.parse_args()

    n, ini, meta = leer_instancia(a.archivo)
    camino, estatus, st = resolver(n, ini, meta, a.prof_max, a.tiempo, a.max_nodos,
                                   a.gmax, a.escalar_gmax)
    if camino is not None:
        print(",".join(camino))                           # salida a consola
    if a.detalle or camino is None:
        print(f"[{estatus}] pasos={len(camino) if camino is not None else '-'} "
              f"tiempo={st['tiempo']:.3f}s expandidos={st['expandidos']} "
              f"reinicios={st['reinicios']} nodos_mem_max={st['max_nodos_mem']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
