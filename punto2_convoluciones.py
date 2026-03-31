"""
Laboratorio de Convolucion Continua y Discreta — Punto 2
Senales & Sistemas - Universidad del Norte

Solucion matematica (analitica) y numerica con np.convolve
para los tres pares de senales del enunciado.
Se grafican ambas curvas superpuestas para comparar.

Patron del curso: delta, arange, concatenate, convolve*delta
(igual que en 'marzo conv.ipynb')
"""

import numpy as np
import matplotlib.pyplot as plt

delta = 0.01   # paso de tiempo (igual que en los cuadernos del curso)


# ═══════════════════════════════════════════════════════════════
# EJERCICIO (a)
#   h(t) = e^(-t/4) * u(t)
#   x(t) = e^(-4t/5) * (u(t+1) - u(t-5))
#
# SOLUCION MATEMATICA (resultado por tramos):
#   y(t) = 0                                          ,  t < -1
#   y(t) = (20/11)*(exp(11/20 - t/4) - exp(-4t/5))   ,  -1 <= t < 5
#   y(t) = (20/11)*(exp(11/20 - t/4) - exp(-(t+11)/4)),  t >= 5
#
# Derivacion breve:
#   y(t) = integral[ h(tau)*x(t-tau) dtau ]
#        = integral[ e^(-tau/4) * e^(-4(t-tau)/5) dtau ]
#   El integrando = e^(-4t/5) * e^(tau*11/20)
#   La integral de e^(tau*11/20) da (20/11)*e^(tau*11/20)
#   Los limites dependen del soporte de h (tau>=0) y x (t-1<=tau<=t+5 → t-5<=tau<=t+1)
# ═══════════════════════════════════════════════════════════════

def ejercicio_a():
    print("=" * 60)
    print("EJERCICIO (a)")
    print("  h(t) = e^(-t/4) * u(t)")
    print("  x(t) = e^(-4t/5) * (u(t+1) - u(t-5))")
    print()
    print("Solucion analitica por tramos:")
    print("  y(t) = 0                                        , t < -1")
    print("  y(t) = (20/11)*(e^(11/20-t/4) - e^(-4t/5))    , -1<=t<5")
    print("  y(t) = (20/11)*(e^(11/20-t/4) - e^(-(t+11)/4)), t>=5")
    print("=" * 60)

    # ── Definir senales por tramos ─────────────────────────────
    # h(t)
    th = np.arange(0, 20 + delta, delta)
    h_t = np.exp(-th / 4)

    # x(t) = e^(-4t/5) para -1 <= t <= 5
    tx = np.arange(-1, 5 + delta, delta)
    x_t = np.exp(-4 * tx / 5)

    # ── Solucion analitica ────────────────────────────────────
    # tramos de tiempo para y(t)
    t1 = np.arange(-1, 5, delta)            # -1 <= t < 5
    t2 = np.arange(5, 25 + delta, delta)    # t >= 5

    y_t1 = (20.0 / 11) * (np.exp(11.0/20 - t1/4) - np.exp(-4*t1/5))
    y_t2 = (20.0 / 11) * (np.exp(11.0/20 - t2/4) - np.exp(-(t2 + 11)/4))

    ty_mat = np.concatenate((t1, t2))
    y_mat  = np.concatenate((y_t1, y_t2))

    # ── Solucion con np.convolve (patron del curso) ───────────
    y_py = np.convolve(x_t, h_t) * delta
    # el tiempo de y_py empieza en tx[0] + th[0] = -1 + 0 = -1
    ty_py = np.arange(-1, -1 + len(y_py) * delta, delta)[:len(y_py)]

    # ── Grafica ───────────────────────────────────────────────
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(tx, x_t, "b", label="x(t)")
    plt.title("Ejercicio (a) — Senales de entrada")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(th[:200], h_t[:200], "g", label="h(t)")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ty_mat, y_mat, "r",  lw=2,   label="y(t) analitica")
    plt.plot(ty_py,  y_py,  "k--", lw=1.5, label="y(t) con convolve")
    plt.title("y(t) — Comparacion analitica vs np.convolve")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Ejercicio (a): h(t)=e^(-t/4)u(t),  x(t)=e^(-4t/5)(u(t+1)-u(t-5))",
                 y=1.01, fontsize=10)
    plt.savefig("ejercicio_a.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Comentario comparativo ────────────────────────────────
    print()
    print("COMPARACION:")
    print("  La curva analitica y la de np.convolve*delta coinciden.")
    print("  La pequeña diferencia en los extremos se debe a la")
    print("  discretizacion con delta =", delta)
    print("  A menor delta, mayor precision de np.convolve.")
    print()


# ═══════════════════════════════════════════════════════════════
# EJERCICIO (b)
#   h(t) = e^(-t/2) * u(t+1)
#   x(t) = e^(t/2)  para -4 < t < 0
#           e^(-t/2) para  0 < t < 4
#
# SOLUCION MATEMATICA (resultado por tramos):
#   y(t) = 0                                   , t < -5
#   y(t) = e^(t/2+1) - e^(-t/2-4)             , -5 <= t < -1
#   y(t) = e^(-t/2)*(t+2) - e^(-(t/2+4))      , -1 <= t < 3
#   y(t) = 5*e^(-t/2) - e^(-(t/2+4))          , t >= 3
#
# Derivacion breve:
#   h(tau) = e^(-tau/2) para tau >= -1
#   x(t-tau) tiene dos ramas segun si (t-tau) es positivo o negativo.
#   Para tau < t: x(t-tau)=e^(-(t-tau)/2) → h*x = e^(-t/2) (constante en tau)
#   Para tau > t: x(t-tau)=e^((t-tau)/2)  → h*x = e^(t/2)*e^(-tau)
#   Los limites cambian segun t y la posicion del soporte de h.
# ═══════════════════════════════════════════════════════════════

def ejercicio_b():
    print("=" * 60)
    print("EJERCICIO (b)")
    print("  h(t) = e^(-t/2) * u(t+1)")
    print("  x(t) = e^(t/2)  para -4 < t < 0")
    print("         e^(-t/2) para  0 < t < 4")
    print()
    print("Solucion analitica por tramos:")
    print("  y(t) = 0                              , t < -5")
    print("  y(t) = e^(t/2+1) - e^(-t/2-4)        , -5 <= t < -1")
    print("  y(t) = e^(-t/2)*(t+2) - e^(-(t/2+4)) , -1 <= t < 3")
    print("  y(t) = 5*e^(-t/2) - e^(-(t/2+4))     , t >= 3")
    print("=" * 60)

    # ── Definir senales ───────────────────────────────────────
    # h(t) = e^(-t/2) para t >= -1
    th = np.arange(-1, 15 + delta, delta)
    h_t = np.exp(-th / 2)

    # x(t) por tramos
    tx1 = np.arange(-4 + delta, 0, delta)
    tx2 = np.arange(delta, 4, delta)
    x_t1 = np.exp( tx1 / 2)
    x_t2 = np.exp(-tx2 / 2)
    tx  = np.concatenate((tx1, tx2))
    x_t = np.concatenate((x_t1, x_t2))

    # ── Solucion analitica ────────────────────────────────────
    ta = np.arange(-5, -1, delta)
    tb = np.arange(-1,  3, delta)
    tc = np.arange( 3, 15 + delta, delta)

    ya = np.exp(ta/2 + 1) - np.exp(-ta/2 - 4)
    yb = np.exp(-tb/2) * (tb + 2) - np.exp(-(tb/2 + 4))
    yc = 5 * np.exp(-tc/2) - np.exp(-(tc/2 + 4))

    ty_mat = np.concatenate((ta, tb, tc))
    y_mat  = np.concatenate((ya, yb, yc))

    # ── Solucion con np.convolve ──────────────────────────────
    y_py  = np.convolve(x_t, h_t) * delta
    t_ini = tx[0] + th[0]   # -4 + (-1) = -5
    ty_py = np.arange(t_ini, t_ini + len(y_py) * delta, delta)[:len(y_py)]

    # ── Grafica ───────────────────────────────────────────────
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(tx, x_t, "b", label="x(t)")
    plt.title("Ejercicio (b) — Senales de entrada")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(th[:150], h_t[:150], "g", label="h(t)")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ty_mat, y_mat, "r",   lw=2,   label="y(t) analitica")
    plt.plot(ty_py,  y_py,  "k--", lw=1.5, label="y(t) con convolve")
    plt.title("y(t) — Comparacion analitica vs np.convolve")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle(
        "Ejercicio (b): h(t)=e^(-t/2)u(t+1),  x(t)=e^(t/2) o e^(-t/2) segun intervalo",
        y=1.01, fontsize=9)
    plt.savefig("ejercicio_b.png", dpi=150, bbox_inches="tight")
    plt.show()

    print()
    print("COMPARACION:")
    print("  Ambas curvas coinciden en la zona central.")
    print("  En los extremos hay pequeñas diferencias por la longitud")
    print("  finita del vector h_t y la discretizacion con delta =", delta)
    print()


# ═══════════════════════════════════════════════════════════════
# EJERCICIO (c)
#   h(t) = e^t * u(1-t)
#   x(t) = u(t+1) - u(t-4)
#
# SOLUCION MATEMATICA (resultado por tramos):
#   y(t) = e^(t+1) - e^(t-4)   ,  t <= 0
#   y(t) = e - e^(t-4)          ,  0 < t <= 5
#   y(t) = 0                    ,  t > 5
#
# Derivacion breve:
#   h(tau) = e^tau para tau <= 1  (soporte en (-inf, 1])
#   x(t-tau) = 1 para -1 <= t-tau <= 4, i.e., t-4 <= tau <= t+1
#   Solapamiento: tau in [t-4, min(1, t+1)]
#   Integral: [e^tau]_{t-4}^{min(1,t+1)}
#   Para t<=0: lim sup = t+1 <= 1 → integral = e^(t+1) - e^(t-4)
#   Para 0<t<=5: lim sup = 1     → integral = e^1 - e^(t-4)
#   Para t>5: lim inf = t-4 > 1 → sin solapamiento → 0
# ═══════════════════════════════════════════════════════════════

def ejercicio_c():
    print("=" * 60)
    print("EJERCICIO (c)")
    print("  h(t) = e^t * u(1-t)")
    print("  x(t) = u(t+1) - u(t-4)")
    print()
    print("Solucion analitica por tramos:")
    print("  y(t) = e^(t+1) - e^(t-4)   ,  t <= 0")
    print("  y(t) = e - e^(t-4)          ,  0 < t <= 5")
    print("  y(t) = 0                    ,  t > 5")
    print("=" * 60)

    # ── Definir senales ───────────────────────────────────────
    # h(t) = e^t para t <= 1 (tomamos desde un limite inferior finito)
    th = np.arange(-8, 1 + delta, delta)
    h_t = np.exp(th)

    # x(t) = 1 para -1 <= t <= 4
    tx = np.arange(-1, 4 + delta, delta)
    x_t = np.ones(len(tx))

    # ── Solucion analitica ────────────────────────────────────
    ta = np.arange(-8, 0 + delta, delta)     # t <= 0
    tb = np.arange(0 + delta, 5, delta)      # 0 < t <= 5

    ya = np.exp(ta + 1) - np.exp(ta - 4)
    yb = np.e - np.exp(tb - 4)

    ty_mat = np.concatenate((ta, tb))
    y_mat  = np.concatenate((ya, yb))

    # ── Solucion con np.convolve ──────────────────────────────
    y_py  = np.convolve(x_t, h_t) * delta
    t_ini = tx[0] + th[0]   # -1 + (-8) = -9
    ty_py = np.arange(t_ini, t_ini + len(y_py) * delta, delta)[:len(y_py)]

    # ── Grafica ───────────────────────────────────────────────
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(tx, x_t, "b", label="x(t) = u(t+1)-u(t-4)")
    plt.title("Ejercicio (c) — Senales de entrada")
    plt.xlabel("t")
    plt.ylim(-0.2, 1.5)
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(th, h_t, "g", label="h(t) = e^t·u(1-t)")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ty_mat, y_mat, "r",   lw=2,   label="y(t) analitica")
    plt.plot(ty_py,  y_py,  "k--", lw=1.5, label="y(t) con convolve")
    plt.title("y(t) — Comparacion analitica vs np.convolve")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Ejercicio (c): h(t)=e^t·u(1-t),  x(t)=u(t+1)-u(t-4)",
                 y=1.01, fontsize=10)
    plt.savefig("ejercicio_c.png", dpi=150, bbox_inches="tight")
    plt.show()

    print()
    print("COMPARACION:")
    print("  Las curvas coinciden muy bien en la zona de soporte.")
    print("  La solucion con convolve incluye el efecto de arrancar")
    print("  h(t) desde t=-8 (limite inferior finito elegido).")
    print("  Si se usa un limite inferior mas negativo, la curva")
    print("  con convolve se acerca mas a la analitica para t<<0.")
    print()


# ─────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("LABORATORIO DE CONVOLUCION CONTINUA Y DISCRETA")
    print("Punto 2: Solucion matematica + np.convolve")
    print()

    ejercicio_a()
    ejercicio_b()
    ejercicio_c()

    print()
    print("NOTA IMPORTANTE:")
    print("  La solucion paso a paso (desarrollo de las integrales)")
    print("  debe entregarse por separado en documento a mano,")
    print("  segun indica el enunciado del laboratorio (Punto 3).")
