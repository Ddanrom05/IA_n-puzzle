"""
Función de evaluación  f(e) = Wg*g1 + W1*h1' + W2*h2 + W3*h3' + W4*h4 + W5*h5
(todas normalizadas a [0,1]; h1' y h3' son -h porque tienden a maximizarse).

h_parcial(estado) calcula SOLO la parte heurística (todo menos g) en una pasada
O(n^2) sobre el tablero + una descomposición en ciclos para h5.
Como h_parcial no depende de g, se calcula una vez por estado y se reutiliza.
"""

PESOS = {"g": 0.40, "h1": 0.05, "h2": 0.25, "h3": 0.05, "h4": 0.10, "h5": 0.15}


class FuncionEvaluacion:
    def __init__(self, n, meta, pesos=None, gmax=100.0):
        p = dict(PESOS)
        if pesos:
            p.update(pesos)
        self.n, self.N, self.meta = n, n * n, meta
        N = self.N

        self.pr = [i // n for i in range(N)]          # fila de cada posición
        self.pc = [i % n for i in range(N)]           # columna de cada posición
        self.gp = [0] * N                             # posición meta de cada ficha
        for pos, v in enumerate(meta):
            self.gp[v] = pos
        self.gr = [self.gp[v] // n for v in range(N)]  # fila meta de la ficha v
        self.gc = [self.gp[v] % n for v in range(N)]   # columna meta de la ficha v

        # Constantes máximas (las de tu documento)
        dmax = n ** 3 if n % 2 == 0 else n ** 3 - n
        emax = 2 * N
        mmax = 1.5 * (N - 1)

        # Coeficientes precalculados: peso / máximo
        self.k1 = -p["h1"] / N          # h1' = -h1/N
        self.k2 = p["h2"] / dmax
        self.k3 = -p["h3"] / (2 * n)    # h3' = -h3/(2n)
        self.k4 = p["h4"] / emax
        self.k5 = p["h5"] / mmax
        self.wg = p["g"]
        self.gmax = float(gmax)

    # ---- heurísticas crudas -------------------------------------------------
    def crudas(self, estado):
        n, N = self.n, self.N
        meta, pr, pc, gr, gc, gp = self.meta, self.pr, self.pc, self.gr, self.gc, self.gp
        h1 = h2 = h4 = 0
        fila_mal = [False] * n
        col_mal = [False] * n
        for i, v in enumerate(estado):
            if v == meta[i]:
                h1 += 1
            else:
                fila_mal[pr[i]] = True
                col_mal[pc[i]] = True
            a = pr[i] - gr[v]
            b = pc[i] - gc[v]
            if a < 0:
                a = -a
            if b < 0:
                b = -b
            h2 += a + b
            if v:
                h4 += (a != 0) + (b != 0)
        h3 = (n - sum(fila_mal)) + (n - sum(col_mal))

        # h5: intercambios directos con el blanco (descomposición en ciclos)
        visto = bytearray(N)
        h5 = 0
        for i in range(N):
            if visto[i]:
                continue
            j, largo, con_cero = i, 0, False
            while not visto[j]:
                visto[j] = 1
                largo += 1
                v = estado[j]
                if v == 0:
                    con_cero = True
                j = gp[v]
            if largo > 1:
                h5 += largo - 1 if con_cero else largo + 1
        return h1, h2, h3, h4, h5

    # ---- parte heurística ponderada y f -------------------------------------
    def h_parcial(self, estado):
        h1, h2, h3, h4, h5 = self.crudas(estado)
        return (self.k1 * h1 + self.k2 * h2 + self.k3 * h3
                + self.k4 * h4 + self.k5 * h5)

    def g_ponderado(self, g):
        x = g / self.gmax
        return self.wg * (x if x < 1.0 else 1.0)

    def f(self, h_parcial, g):
        return h_parcial + self.g_ponderado(g)

    def desglose(self, estado, g):
        """Valores crudos, normalizados y ponderados (útil para el reporte)."""
        h1, h2, h3, h4, h5 = self.crudas(estado)
        return {
            "h1": (h1, self.k1 * h1), "h2": (h2, self.k2 * h2),
            "h3": (h3, self.k3 * h3), "h4": (h4, self.k4 * h4),
            "h5": (h5, self.k5 * h5), "g": (g, self.g_ponderado(g)),
            "f": self.f(self.h_parcial(estado), g),
        }


if __name__ == "__main__":
    # Ejemplo de tu documento: debe dar f = 0.0358
    meta = bytes([1, 2, 3, 4, 5, 6, 7, 8, 0])
    ex = bytes([1, 2, 3, 4, 0, 6, 7, 5, 8])
    fe = FuncionEvaluacion(3, meta, gmax=100)
    d = fe.desglose(ex, g=2)
    for k, v in d.items():
        print(k, v)
