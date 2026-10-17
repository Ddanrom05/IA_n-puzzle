
import argparse
import random
from pathlib import Path


# Directorio donde se guardan las instancias.
DIRECTORIO_INSTANCIAS = Path("instances")

# Número de movimientos asociado a cada nivel experimental.
DIFICULTADES = {
    10: "facil",
    20: "medio",
    50: "dificil",
}


def crear_meta(n):
    """
    Crea el tablero meta:
    1, 2, 3, ..., n*n-1, 0.

    La representación interna es una tupla unidimensional.
    """
    return tuple(range(1, n * n)) + (0,)


def obtener_vecinos(estado, n):
    """
    Genera los estados vecinos moviendo el espacio vacío.

    Devuelve pares (nuevo_estado, movimiento).
    Los movimientos indican el desplazamiento del espacio:
    U = arriba, D = abajo, L = izquierda, R = derecha.
    """
    posicion_cero = estado.index(0)
    fila, columna = divmod(posicion_cero, n)

    movimientos = [
        (-1, 0, "U"),
        (1, 0, "D"),
        (0, -1, "L"),
        (0, 1, "R"),
    ]

    for df, dc, movimiento in movimientos:
        nueva_fila = fila + df
        nueva_columna = columna + dc

        if not (
            0 <= nueva_fila < n
            and 0 <= nueva_columna < n
        ):
            continue

        nueva_posicion = nueva_fila * n + nueva_columna

        nuevo_estado = list(estado)
        nuevo_estado[posicion_cero], nuevo_estado[nueva_posicion] = (
            nuevo_estado[nueva_posicion],
            nuevo_estado[posicion_cero],
        )

        yield tuple(nuevo_estado), movimiento


def generar_tablero(n, cantidad_movimientos, rng):
    """
    Parte del estado meta y realiza movimientos legales.

    Evita repetir estados completos y regresar a la posición
    del espacio vacío ocupada cuatro pasos antes.
    """
    estado = crear_meta(n)
    visitados = {estado}
    movimientos_realizados = []
    posiciones_cero = [estado.index(0)]

    for _ in range(cantidad_movimientos):
        opciones = []

        for vecino, movimiento in obtener_vecinos(estado, n):
            # Evita repetir un estado completo.
            if vecino in visitados:
                continue

            # Evita cerrar un ciclo de cuatro movimientos
            # en la trayectoria del espacio vacío.
            nueva_posicion_cero = vecino.index(0)

            if (
                len(posiciones_cero) >= 4
                and nueva_posicion_cero == posiciones_cero[-4]
            ):
                continue

            opciones.append((vecino, movimiento))

        if not opciones:
            raise RuntimeError(
                "Recorrido bloqueado: genera una nueva trayectoria."
            )

        estado, movimiento = rng.choice(opciones)

        visitados.add(estado)
        movimientos_realizados.append(movimiento)
        posiciones_cero.append(estado.index(0))

    return estado, movimientos_realizados


def guardar_instancia(ruta, n, inicial, meta):
    """
    Guarda el tablero inicial y el tablero meta en el formato
    de entrada definido por el profesor.
    """
    with ruta.open("w", encoding="utf-8") as archivo:
        archivo.write(f"{n}\n")

        for tablero in (inicial, meta):
            for fila in range(n):
                inicio = fila * n
                fin = inicio + n
                valores = tablero[inicio:fin]

                archivo.write(
                    ",".join(map(str, valores)) + "\n"
                )


def generar_instancias(n, cantidad_movimientos, cantidad_instancias,
                       semilla=None):
    """
    Genera instancias únicas para una combinación de tamaño
    y cantidad de movimientos.
    """
    if not 5 <= n <= 15:
        raise ValueError(
            "El tamaño del tablero debe estar entre 5 y 15."
        )

    if cantidad_movimientos <= 0:
        raise ValueError(
            "La cantidad de movimientos debe ser positiva."
        )

    if cantidad_instancias <= 0:
        raise ValueError(
            "La cantidad de instancias debe ser positiva."
        )

    rng = random.Random(semilla)
    meta = crear_meta(n)

    dificultad = DIFICULTADES.get(
        cantidad_movimientos,
        f"{cantidad_movimientos}_movimientos",
    )

    directorio = (
        DIRECTORIO_INSTANCIAS
        / f"{n}x{n}"
        / dificultad
    )
    directorio.mkdir(parents=True, exist_ok=True)

    estados_generados = set()
    numero_instancia = 1
    intentos = 0
    max_intentos = max(1000, cantidad_instancias * 100)

    while numero_instancia <= cantidad_instancias:
        intentos += 1

        if intentos > max_intentos:
            raise RuntimeError(
                "No se pudo completar la cantidad solicitada "
                "de tableros únicos. Intenta con otra semilla."
            )

        try:
            inicial, recorrido = generar_tablero(
                n, cantidad_movimientos, rng
            )
        except RuntimeError:
            # Un recorrido puede quedarse sin vecinos disponibles.
            continue

        # Evita guardar dos veces el mismo tablero inicial.
        if inicial in estados_generados:
            continue

        estados_generados.add(inicial)

        nombre = (
            f"{numero_instancia:03d}_"
            f"{n}_"
            f"{cantidad_movimientos}.txt"
        )
        ruta = directorio / nombre

        guardar_instancia(ruta, n, inicial, meta)

        numero_instancia += 1

    print("Generación completada.")
    print(f"Tamaño: {n}x{n}")
    print(f"Movimientos inyectados: {cantidad_movimientos}")
    print(f"Instancias generadas: {cantidad_instancias}")
    print(f"Directorio: {directorio.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generador de instancias del rompecabezas."
    )

    parser.add_argument(
        "n",
        type=int,
        help="Dimensión del tablero, de 5 a 15.",
    )
    parser.add_argument(
        "movimientos",
        type=int,
        help="Cantidad de movimientos a inyectar.",
    )
    parser.add_argument(
        "instancias",
        type=int,
        help="Cantidad de instancias que se generarán.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para reproducir la generación.",
    )

    args = parser.parse_args()

    try:
        generar_instancias(
            args.n,
            args.movimientos,
            args.instancias,
            args.seed,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()