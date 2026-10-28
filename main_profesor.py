"""
Programa para probar UNA instancia aislada.

Uso:
    python3 main_profesor.py ruta/al/archivo.txt
    python3 main_profesor.py                      (pregunta la ruta por teclado)
    python3 main_profesor.py archivo.txt 300      (limite de tiempo opcional, en segundos)

Formato del archivo: 1a linea = n; siguientes n lineas = tablero inicial; siguientes n lineas
= tablero meta (numeros separados por comas, 0 = espacio en blanco).

Salida por CONSOLA: la secuencia de movimientos del espacio en blanco (U, D, L, R)
separados por comas. Ejemplo:  U,L
"""
import sys

from puzzle_utils import leer_instancia
from ida_estrella import resolver

# ---- parametros (modificables) -------------------------------------------------------
LIMITE_TIEMPO_SEG = 120      # limite de tiempo para resolver la instancia
PROFUNDIDAD_MAX = 100        # control interno de profundidad por tanda
MAX_NODOS = 2_000_000        # tope de nodos en memoria
# --------------------------------------------------------------------------------------


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else input("Ruta del archivo de instancia: ").strip().strip('"\'')
    limite = float(sys.argv[2]) if len(sys.argv) > 2 else LIMITE_TIEMPO_SEG

    try:
        n, inicial, meta = leer_instancia(ruta)
    except (OSError, ValueError) as error:
        print(f"Error al leer el archivo: {error}", file=sys.stderr)
        sys.exit(2)

    camino, estatus, _ = resolver(n, inicial, meta, prof_max=PROFUNDIDAD_MAX,
                                  limite_seg=limite, max_nodos=MAX_NODOS)
    if camino is None:
        mensajes = {"TIMEOUT": f"No se encontro solucion en {limite:.0f} s.",
                    "SIN_SOLUCION": "La instancia no tiene solucion (paridad incompatible).",
                    "AGOTADO": "Espacio de busqueda agotado sin encontrar la meta."}
        print(mensajes.get(estatus, estatus), file=sys.stderr)
        sys.exit(1)

    print(",".join(camino))


if __name__ == "__main__":
    main()
