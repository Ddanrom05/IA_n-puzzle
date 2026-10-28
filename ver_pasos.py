"""
Muestra la secuencia de tableros de una solución, paso a paso.

Uso:
  python3 ver_pasos.py instances/5x5/facil/001_5_10.txt
  python3 ver_pasos.py instances/6x6/dificil/009_6_50.txt --tiempo 60
  python3 ver_pasos.py instances/5x5/facil/001_5_10.txt --camino "U,L,L,D,R"
  python3 ver_pasos.py instances/5x5/facil/001_5_10.txt --animar 0.3
  python3 ver_pasos.py instances/5x5/facil/001_5_10.txt --solo-final
"""
import argparse
import sys
import time

from puzzle_utils import leer_instancia, aplicar_camino
from ida_estrella import resolver

NOMBRE = {"U": "arriba", "D": "abajo", "L": "izquierda", "R": "derecha"}


def formatear(n, estado, resaltar=None):
    ancho = len(str(n * n - 1))
    filas = []
    for f in range(n):
        celdas = []
        for c in range(n):
            v = estado[f * n + c]
            txt = "·" if v == 0 else str(v)
            celdas.append(txt.rjust(ancho))
        filas.append(" ".join(celdas))
    return "\n".join(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo")
    ap.add_argument("--camino", help='secuencia ya conocida, p. ej. "U,L,D" (si no, se resuelve)')
    ap.add_argument("--tiempo", type=float, default=120.0)
    ap.add_argument("--prof-max", type=int, default=100)
    ap.add_argument("--max-nodos", type=int, default=2_000_000)
    ap.add_argument("--animar", type=float, default=0.0,
                    help="segundos de pausa entre pasos (0 = imprime todo de corrido)")
    ap.add_argument("--solo-final", action="store_true",
                    help="solo inicial, final y la secuencia")
    a = ap.parse_args()

    n, ini, meta = leer_instancia(a.archivo)

    if a.camino:
        camino = [m.strip() for m in a.camino.split(",") if m.strip()]
    else:
        camino, estatus, st = resolver(n, ini, meta, a.prof_max, a.tiempo, a.max_nodos)
        if camino is None:
            print(f"No se encontró solución ({estatus}, {st['tiempo']:.1f} s).")
            sys.exit(1)
        print(f"Resuelto en {st['tiempo']:.3f} s, {len(camino)} pasos, "
              f"{st['expandidos']} nodos expandidos.")

    print("\nSecuencia: " + ",".join(camino))
    print(f"\nTablero inicial:\n{formatear(n, ini)}")

    estado = ini
    if not a.solo_final:
        for i, m in enumerate(camino, 1):
            estado = aplicar_camino(n, estado, [m])
            print(f"\nPaso {i}/{len(camino)}: el blanco se mueve {NOMBRE[m]} ({m})")
            print(formatear(n, estado))
            if a.animar:
                time.sleep(a.animar)
    else:
        estado = aplicar_camino(n, ini, camino)

    print(f"\nTablero final:\n{formatear(n, estado)}")
    print("\n✔ Coincide con la meta." if estado == meta else "\n✘ NO coincide con la meta.")


if __name__ == "__main__":
    main()
