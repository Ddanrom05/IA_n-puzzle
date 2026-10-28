"""
Longitud ÓPTIMA real de una instancia, con IDA* clásico + distancia Manhattan
(heurística admisible y consistente => garantiza optimalidad). Sirve como referencia
para la "comparativa contra la solución óptima" en los tableros donde es viable
(típicamente 3x3 a 5x5 con pocos movimientos).

Solo calcula el optimo. La comparacion contra lo que
encuentra el solver la hace experimentos.py (--solo-graficas), cruzando por nombre de archivo.

Uso:
  python3 optimo.py --tamanos 5 --movs 10 20 --instancias 30 --tiempo 60
Salida: results/optimo_<n>x<n>_<dif>.csv  (n, movs_inyectados, archivo, optimo, t_optimo)
"""
import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

from puzzle_utils import leer_instancia, tabla_vecinos, es_resoluble

NOMBRES = {10: "facil", 20: "medio", 50: "dificil"}
INF = float("inf")


class Tiempo(Exception):
    pass


def longitud_optima(n, inicial, meta, limite_seg=60.0):
    """Devuelve la longitud óptima, o None si se agota el tiempo."""
    N = n * n
    if not es_resoluble(n, inicial, meta):
        return None
    gp = [0] * N
    for p, v in enumerate(meta):
        gp[v] = p
    # D[v][q] = distancia Manhattan de la ficha v puesta en la casilla q (v=0 no cuenta)
    D = [[0] * N for _ in range(N)]
    for v in range(1, N):
        for q in range(N):
            D[v][q] = abs(q // n - gp[v] // n) + abs(q % n - gp[v] % n)
    vec = [tuple(q for q, _ in lst) for lst in tabla_vecinos(n)]
    est = bytearray(inicial)
    h0 = sum(D[v][q] for q, v in enumerate(est) if v)
    limite = time.perf_counter() + limite_seg
    cuenta = [0]
    sys.setrecursionlimit(10000)

    def dfs(g, h, cero, prev, cota):
        f = g + h
        if f > cota:
            return f
        if h == 0:
            return -1                       # todas las fichas en su sitio => meta
        cuenta[0] += 1
        if (cuenta[0] & 4095) == 0 and time.perf_counter() > limite:
            raise Tiempo
        minimo = INF
        for q in vec[cero]:
            if q == prev:
                continue
            v = est[q]
            est[cero] = v
            est[q] = 0
            r = dfs(g + 1, h - D[v][q] + D[v][cero], q, cero, cota)
            est[q] = v
            est[cero] = 0
            if r == -1:
                return -1
            if r < minimo:
                minimo = r
        return minimo

    cota = h0
    try:
        while True:
            r = dfs(0, h0, est.index(0), -1, cota)
            if r == -1:
                return cota
            cota = r
    except Tiempo:
        return None


def main():
    ap = argparse.ArgumentParser(description="Longitud optima real por instancia (IDA* + Manhattan)")
    ap.add_argument("--tamanos", type=int, nargs="+", default=[5])
    ap.add_argument("--movs", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--instancias", type=int, default=100)
    ap.add_argument("--tiempo", type=float, default=60.0, help="limite por instancia (s)")
    ap.add_argument("--instancias-dir", default="instances")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)

    for n in a.tamanos:
        for m in a.movs:
            dif = NOMBRES.get(m, f"{m}_movimientos")
            carpeta = Path(a.instancias_dir) / f"{n}x{n}" / dif
            filas = []
            for ruta in sorted(carpeta.glob("*.txt"))[:a.instancias]:
                _, ini, meta = leer_instancia(ruta)
                t = time.perf_counter()
                opt = longitud_optima(n, ini, meta, a.tiempo)
                dt = time.perf_counter() - t
                if opt is not None and opt > m:
                    print(f"[AVISO] {ruta.name}: optimo={opt} > inyectados={m} (no deberia pasar)")
                filas.append({"n": n, "movs_inyectados": m, "archivo": ruta.name,
                              "optimo": opt if opt is not None else "",
                              "t_optimo": round(dt, 4)})
                print(f"{n}x{n}/{dif} {ruta.name}: optimo={opt} ({dt:.2f}s)", flush=True)
            if not filas:
                print(f"[aviso] sin instancias en {carpeta}")
                continue
            with (out / f"optimo_{n}x{n}_{dif}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
                w.writeheader()
                w.writerows(filas)
            ok = [x["optimo"] for x in filas if x["optimo"] != ""]
            print(f"== {n}x{n} {dif}: optimo calculado en {len(ok)}/{len(filas)}; "
                  f"optimo medio = {st.mean(ok) if ok else float('nan'):.2f}\n")


if __name__ == "__main__":
    main()