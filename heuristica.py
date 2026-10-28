"""
Función de evaluación  f(e) = Wg*g1 + W1*h1' + W2*h2 + W3*h3' + W4*h4 + W5*h5

h_parcial(estado) calcula SOLO la parte heurística (todo menos g) en una pasada
O(n^2) sobre el tablero + una descomposición en ciclos para h5.
Como h_parcial no depende de g, se calcula una vez por estado y se reutiliza.
"""

PESOS = {"g": 0.40, "h1": 0.05, "h2": 0.25, "h3": 0.05, "h4": 0.10, "h5": 0.15}


class FuncionEvaluacion:
    def __init__(self, n, meta, pesos=None, gmax=100.0):
        p = dict(PESOS)
        self.n, self.N, self.meta = n, n * n, meta
        N = self.N

        self.pr = [i // n for i in range(N)]          # fila de cada posición
        self.pc = [i % n for i in range(N)]           # columna de cada posición
        self.gp = [0] * N                             # posición meta de cada ficha
        for pos, v in enumerate(meta):
            self.gp[v] = pos
        self.gr = [self.gp[v] // n for v in range(N)]  # fila meta de la ficha v
        self.gc = [self.gp[v] % n for v in range(N)]   # columna meta de la ficha v

        # Constantes máximas 
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

        # Tablas para la evaluación incremental (un movimiento solo cambia 2 casillas)
        self.D2 = [[abs(self.pr[q] - self.gr[v]) + abs(self.pc[q] - self.gc[v])
                    for q in range(N)] for v in range(N)]            # Manhattan de v en q
        self.D4 = [[(self.pr[q] != self.gr[v]) + (self.pc[q] != self.gc[v])
                    for q in range(N)] for v in range(N)]            # fila/col incorrecta
        self.meta_fila = [meta[r * n:(r + 1) * n] for r in range(n)]
        self.meta_col = [meta[c::n] for c in range(n)]

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

    # ---- evaluación incremental --------------------------------------------
    # Un nodo guarda UN entero con sus componentes empaquetadas:
    #   bits  0-7   h1 | 8-19 h2 | 20-25 h3 | 26-34 h4 | 35-44 h5 | 45-59 máscara de filas incorrectas
    def completo(self, estado):
        """Componentes empaquetadas calculadas desde cero (se usa en las raíces)."""
        h1, h2, h3, h4, h5 = self.crudas(estado)
        n, mf = self.n, self.meta_fila
        mask = 0
        for r in range(n):
            if estado[r * n:(r + 1) * n] != mf[r]:
                mask |= 1 << r
        return h1 | (h2 << 8) | (h3 << 20) | (h4 << 26) | (h5 << 35) | (mask << 45)

    def hp_de(self, comp):
        return (self.k1 * (comp & 255) + self.k2 * ((comp >> 8) & 4095)
                + self.k3 * ((comp >> 20) & 63) + self.k4 * ((comp >> 26) & 511)
                + self.k5 * ((comp >> 35) & 1023))

    def hijo(self, padre, comp, cero, nueva, h):
        """
        Componentes del hijo `h` (ficha que estaba en `nueva` pasa a `cero`) a partir de
        las del padre. h1, h2, h4: delta O(1) con tablas. h3: solo se recomprueban las
        2-3 líneas afectadas (comparación de rebanadas en C). h5: ciclos solo sobre las
        filas incorrectas (una posición en fila correcta es punto fijo).
        Devuelve (h_parcial, comp_empaquetado).
        """
        n, meta, pr, pc = self.n, self.meta, self.pr, self.pc
        v = padre[nueva]
        mc, mn = meta[cero], meta[nueva]
        D2v, D20, D4v = self.D2[v], self.D2[0], self.D4[v]

        h1 = (comp & 255) + (v == mc) + (mn == 0) - (mc == 0) - (v == mn)
        h2 = ((comp >> 8) & 4095) + D2v[cero] - D2v[nueva] + D20[nueva] - D20[cero]
        h4 = ((comp >> 26) & 511) + D4v[cero] - D4v[nueva]
        h3 = (comp >> 20) & 63
        mask = comp >> 45

        mfila, mcol = self.meta_fila, self.meta_col
        rp, rq = pr[cero], pr[nueva]
        if rp == rq:
            filas, cols = (rp,), (pc[cero], pc[nueva])
        else:
            filas, cols = (rp, rq), (pc[cero],)
        for r in filas:
            a = r * n
            antes = padre[a:a + n] == mfila[r]
            despues = h[a:a + n] == mfila[r]
            h3 += despues - antes
            if despues:
                mask &= ~(1 << r)
            else:
                mask |= 1 << r
        for c in cols:
            h3 += (h[c::n] == mcol[c]) - (padre[c::n] == mcol[c])

        gp = self.gp
        visto = set()
        h5 = 0
        m, r = mask, 0
        while m:
            if m & 1:
                for i in range(r * n, r * n + n):
                    if h[i] == meta[i] or i in visto:
                        continue
                    j, largo, con_cero = i, 0, False
                    while j not in visto:
                        visto.add(j)
                        largo += 1
                        t = h[j]
                        if t == 0:
                            con_cero = True
                        j = gp[t]
                    h5 += largo - 1 if con_cero else largo + 1
            m >>= 1
            r += 1

        hp = (self.k1 * h1 + self.k2 * h2 + self.k3 * h3 + self.k4 * h4 + self.k5 * h5)
        return hp, h1 | (h2 << 8) | (h3 << 20) | (h4 << 26) | (h5 << 35) | (mask << 45)

    def g_ponderado(self, g):
        x = g / self.gmax
        return self.wg * (x if x < 1.0 else 1.0)

    def f(self, h_parcial, g):
        return h_parcial + self.g_ponderado(g)

    def desglose(self, estado, g):
        """Valores crudos, normalizados y ponderados."""
        h1, h2, h3, h4, h5 = self.crudas(estado)
        return {
            "h1": (h1, self.k1 * h1), "h2": (h2, self.k2 * h2),
            "h3": (h3, self.k3 * h3), "h4": (h4, self.k4 * h4),
            "h5": (h5, self.k5 * h5), "g": (g, self.g_ponderado(g)),
            "f": self.f(self.h_parcial(estado), g),
        }


if __name__ == "__main__":
    # Ejemplo del reporte (3x3, g = 2, Gmax = 100)
    meta = bytes([1, 2, 3, 4, 5, 6, 7, 8, 0])
    ex = bytes([1, 2, 3, 4, 0, 6, 7, 5, 8])
    d = FuncionEvaluacion(3,meta).desglose(ex, g=2)
    for k in ("g", "h1", "h2", "h3", "h4", "h5"):
        print(f"   {k}: crudo={d[k][0]}  aporte={d[k][1]:+.4f}")
    print(f"   f = {d['f']:.4f}\n")