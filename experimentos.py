"""
Corre el solver sobre batches de instancias (un batch = un tamaño + una dificultad).

  - Cada batch se guarda en su propio CSV dentro de results/ (se escribe fila por fila,
    así que si interrumpes la corrida no pierdes lo ya resuelto):
        results/5x5_facil.csv, results/8x8_dificil.csv, ...
  - Al terminar, lee TODOS los CSV de results/ y genera:
        results/resumen_<etiqueta>.csv
        results/graficas/*.png

Ejemplos:
  python3 experimentos.py --tamanos 5 6 --movs 10 20 50 --instancias 100 --workers 3
  python3 experimentos.py --solo-graficas                      # regenera tablas y gráficas
"""
import argparse
import csv
import statistics as st
from multiprocessing import Pool
from pathlib import Path

from puzzle_utils import leer_instancia
from ida_estrella import resolver

try:
    import generador
    NOMBRES = generador.DIFICULTADES
    DIR_INST = generador.DIRECTORIO_INSTANCIAS
except ImportError:
    generador = None
    NOMBRES = {10: "facil", 20: "medio", 50: "dificil"}
    DIR_INST = Path("instances")

CAMPOS = ["etiqueta", "n", "movs_inyectados", "archivo", "estatus", "resuelto", "tiempo",
          "longitud", "ratio_vs_inyectados", "expandidos", "reinicios", "nodos_mem_max"]


def nombre_dif(movs):
    return NOMBRES.get(movs, f"{movs}_movimientos")


def carpeta(n, movs):
    return DIR_INST / f"{n}x{n}" / nombre_dif(movs)


def ruta_csv(out, n, movs, etiqueta):
    sufijo = "" if etiqueta == "base" else f"_{etiqueta}"
    return out / f"{n}x{n}_{nombre_dif(movs)}{sufijo}.csv"


def _tarea(arg):
    ruta, n_movs, cfg, etiqueta = arg
    n, ini, meta = leer_instancia(ruta)
    camino, estatus, s = resolver(n, ini, meta, **cfg)
    return {
        "etiqueta": etiqueta, "n": n, "movs_inyectados": n_movs, "archivo": Path(ruta).name,
        "estatus": estatus, "resuelto": int(camino is not None),
        "tiempo": round(s["tiempo"], 4),
        "longitud": len(camino) if camino is not None else "",
        "ratio_vs_inyectados": round(len(camino) / n_movs, 3) if camino else "",
        "expandidos": s["expandidos"], "reinicios": s["reinicios"],
        "nodos_mem_max": s["max_nodos_mem"],
    }


def correr_lote(n, movs, a, cfg, out, pool):
    d = carpeta(n, movs)
    archivos = sorted(d.glob("*.txt")) if d.exists() else []
    if len(archivos) < a.instancias:
        if a.generar and generador is not None:
            generador.generar_instancias(n, movs, a.instancias, a.semilla)
            archivos = sorted(d.glob("*.txt"))
        else:
            print(f"[aviso] {d}: solo {len(archivos)} instancias (usa --generar).")
    tareas = [(str(r), movs, cfg, a.etiqueta) for r in archivos[:a.instancias]]
    if not tareas:
        return
    ruta = ruta_csv(out, n, movs, a.etiqueta)
    print(f"\n== Batch {n}x{n} / {nombre_dif(movs)} ({len(tareas)} instancias) -> {ruta}")
    it = pool.imap(_tarea, tareas) if pool else map(_tarea, tareas)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        for i, r in enumerate(it, 1):
            w.writerow(r)
            fh.flush()
            print(f"[{i}/{len(tareas)}] {r['archivo']} {r['estatus']} {r['tiempo']}s", flush=True)


def cargar(out, etiqueta):
    filas = []
    for p in sorted(out.glob("*.csv")):
        if p.name.startswith(("resumen", "optimo")):
            continue
        with p.open(encoding="utf-8") as fh:
            lector = csv.DictReader(fh)
            if not set(CAMPOS) <= set(lector.fieldnames or []):
                print(f"[aviso] se ignora {p.name}: no tiene el formato de resultados.")
                continue
            for r in lector:
                if r.get("etiqueta", "base") != etiqueta:
                    continue
                r["n"], r["movs_inyectados"] = int(r["n"]), int(r["movs_inyectados"])
                r["resuelto"], r["tiempo"] = int(r["resuelto"]), float(r["tiempo"])
                for k in ("longitud", "ratio_vs_inyectados"):
                    r[k] = float(r[k]) if r[k] != "" else ""
                filas.append(r)
    return filas


def cargar_optimo(out):
    """{(n, movs, archivo): (optimo, t_optimo)} leido de results/optimo_*.csv"""
    opt = {}
    for p in sorted(out.glob("optimo_*.csv")):
        with p.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("optimo", "") == "":
                    continue
                opt[(int(r["n"]), int(r["movs_inyectados"]), r["archivo"])] = (
                    int(r["optimo"]), float(r["t_optimo"]))
    return opt


def _pares(g, n, m, opt):
    """Instancias resueltas por el solver y con optimo conocido."""
    return [x for x in g if x["resuelto"] and opt.get((n, m, x["archivo"]), (0,))[0] > 0]


def resumen(filas, out, etiqueta, opt=None):
    opt = opt or {}
    grupos = {}
    for f in filas:
        grupos.setdefault((f["n"], f["movs_inyectados"]), []).append(f)
    print(f"\nResumen [{etiqueta}]")
    print(f"{'n':>3} {'movs':>5} {'resueltos':>10} {'t_media(s)':>11} {'t_std':>9} "
          f"{'t_max':>9} {'pasos/inyect':>13} {'pasos/optimo':>13}")
    tabla = []
    for (n, m), g in sorted(grupos.items()):
        ok = [x for x in g if x["resuelto"]]
        ts = [x["tiempo"] for x in ok]
        rs = [x["ratio_vs_inyectados"] for x in ok]
        tm = st.mean(ts) if ts else float("nan")
        td = st.pstdev(ts) if len(ts) > 1 else 0.0
        tx = max(ts) if ts else float("nan")
        rm = st.mean(rs) if rs else float("nan")
        pares = _pares(g, n, m, opt)
        ro = (st.mean([x["longitud"] / opt[(n, m, x["archivo"])][0] for x in pares])
              if pares else float("nan"))
        print(f"{n:>3} {m:>5} {len(ok):>4}/{len(g):<5} {tm:>11.4f} {td:>9.4f} {tx:>9.3f} "
              f"{rm:>13.3f} {ro:>13.3f}")
        tabla.append({"n": n, "movs": m, "total": len(g), "resueltos": len(ok),
                      "tasa_resolucion_pct": round(100 * len(ok) / len(g), 1),
                      "t_media": round(tm, 4), "t_std": round(td, 4), "t_max": round(tx, 4),
                      "pasos_sobre_inyectados": round(rm, 3),
                      "pasos_sobre_optimo": round(ro, 3), "con_optimo": len(pares)})
    if tabla:
        with (out / f"resumen_{etiqueta}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tabla[0].keys()))
            w.writeheader()
            w.writerows(tabla)
    return grupos


def graficas(grupos, out, etiqueta, opt=None):
    opt = opt or {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado: se omiten las graficas (pip3 install matplotlib).")
        return
    dest = out / "graficas"
    dest.mkdir(parents=True, exist_ok=True)
    tamanos = sorted({n for n, _ in grupos})
    movs_lista = sorted({m for _, m in grupos})

    def media_std(vals):
        return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0) if vals else (0, 0)

    metricas = (
        ("tiempos", "Tiempo medio de casos resueltos (s)", True,
         lambda g, ok: media_std([x["tiempo"] for x in ok])),
        ("tasa_resolucion", "Instancias resueltas (%)", False,
         lambda g, ok: (100 * len(ok) / len(g), 0)),
        ("longitud_vs_inyectados", "Pasos encontrados / movimientos inyectados", False,
         lambda g, ok: media_std([x["ratio_vs_inyectados"] for x in ok])),
    )
    for nombre, ylabel, log, fn in metricas:
        plt.figure(figsize=(7, 4.5))
        for m in movs_lista:
            xs, ys, es = [], [], []
            for n in tamanos:
                g = grupos.get((n, m))
                if g:
                    y, e = fn(g, [x for x in g if x["resuelto"]])
                    xs.append(n); ys.append(y); es.append(e)
            plt.errorbar(xs, ys, yerr=None if nombre == "tasa_resolucion" else es,
                         marker="o", capsize=3, label=f"{m} movs")
        if log:
            plt.yscale("log")
        plt.xlabel("Dimension del tablero (n x n)")
        plt.xticks(tamanos)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} [{etiqueta}]", fontsize=10)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(dest / f"{nombre}_{etiqueta}.png", dpi=150)
        plt.close()

    # ---- comparativa contra la solucion optima (requiere results/optimo_*.csv) ----
    if opt:
        ratio, tiempos = {}, {}
        for m in movs_lista:
            for n in tamanos:
                p = _pares(grupos.get((n, m), []), n, m, opt)
                if p:
                    ratio.setdefault(m, []).append(
                        (n, st.mean([x["longitud"] / opt[(n, m, x["archivo"])][0] for x in p])))
                    tiempos.setdefault(m, []).append(
                        (n, st.mean([x["tiempo"] for x in p]),
                         st.mean([opt[(n, m, x["archivo"])][1] for x in p])))
        if ratio:
            plt.figure(figsize=(7, 4.5))
            for m, pts in ratio.items():
                plt.plot([a for a, _ in pts], [b for _, b in pts], marker="o", label=f"{m} movs")
            plt.axhline(1.0, color="gray", ls=":", label="optimo (=1)")
            plt.xlabel("Dimension del tablero (n x n)")
            plt.xticks(tamanos)
            plt.ylabel("Pasos encontrados / pasos optimos")
            plt.title(f"Calidad de la solucion frente al optimo [{etiqueta}]", fontsize=10)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dest / f"comparativa_longitud_optimo_{etiqueta}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(7, 4.5))
            for m, pts in tiempos.items():
                l, = plt.plot([a for a, _, _ in pts], [b for _, b, _ in pts],
                              marker="o", label=f"solver, {m} movs")
                plt.plot([a for a, _, _ in pts], [c for _, _, c in pts], marker="s", ls="--",
                         color=l.get_color(), label=f"optimo (Manhattan), {m} movs")
            plt.yscale("log")
            plt.xlabel("Dimension del tablero (n x n)")
            plt.xticks(tamanos)
            plt.ylabel("Tiempo medio (s)")
            plt.title(f"Tiempo: solver vs. solver optimo [{etiqueta}]", fontsize=10)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(dest / f"comparativa_tiempo_optimo_{etiqueta}.png", dpi=150)
            plt.close()
    else:
        print("(sin results/optimo_*.csv: se omiten las graficas de comparativa contra el optimo)")
    print(f"Graficas en {dest.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanos", type=int, nargs="+", default=[5])
    ap.add_argument("--movs", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--instancias", type=int, default=100)
    ap.add_argument("--tiempo", type=float, default=120.0)
    ap.add_argument("--prof-max", type=int, default=100)
    ap.add_argument("--max-nodos", type=int, default=2_000_000)
    ap.add_argument("--gmax", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--generar", action="store_true", help="genera instancias si faltan")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", default="results", help="carpeta de CSV (default: results)")
    ap.add_argument("--etiqueta", default="base",
                    help="nombre de la corrida (default: base); así no se sobrescriben CSV")
    ap.add_argument("--solo-graficas", action="store_true",
                    help="no resuelve nada; regenera resumen y gráficas desde results/")
    a = ap.parse_args()

    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)

    if not a.solo_graficas:
        cfg = dict(prof_max=a.prof_max, limite_seg=a.tiempo, max_nodos=a.max_nodos,
                   gmax=a.gmax)
        pool = Pool(a.workers) if a.workers > 1 else None
        try:
            for n in a.tamanos:
                for movs in a.movs:
                    correr_lote(n, movs, a, cfg, out, pool)
        finally:
            if pool:
                pool.close()
                pool.join()

    filas = cargar(out, a.etiqueta)
    if not filas:
        print(f"No hay CSV con etiqueta '{a.etiqueta}' en {out}/")
        return
    opt = cargar_optimo(out)
    graficas(resumen(filas, out, a.etiqueta, opt), out, a.etiqueta, opt)


if __name__ == "__main__":
    main()