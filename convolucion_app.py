"""
Laboratorio de Convolucion Continua y Discreta
Senales & Sistemas - Universidad del Norte

Aplicacion Python con GUI para visualizar el proceso de convolucion
en tiempo continuo y discreto, paso a paso.

Librerias: numpy, matplotlib, tkinter (estandar del curso)
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ─────────────────────────────────────────────────────────────
#  DEFINICION DE SENALES CONTINUAS (Figura 1 del laboratorio)
# ─────────────────────────────────────────────────────────────

delta = 0.05   # paso de tiempo (igual que en clase)

def senal_a():
    """Fig 1a: bipolar — x(t)=2 en [0,3], x(t)=-2 en [3,4]"""
    t = np.arange(-1, 5 + delta, delta)
    x = np.zeros(len(t))
    x[(t >= 0) & (t <= 3)] =  2.0
    x[(t >  3) & (t <= 4)] = -2.0
    return t, x

def senal_b():
    """Fig 1b: rampa x(t) = -t para |t| <= 1"""
    t = np.arange(-2, 3 + delta, delta)
    x = np.zeros(len(t))
    mask = np.abs(t) <= 1
    x[mask] = -t[mask]
    return t, x

def senal_c():
    """Fig 1c: x(t)=2 en [-1,0], baja de 2 a -2 en [0,2], x(t)=-2 en [2,4]"""
    t = np.arange(-2, 5 + delta, delta)
    x = np.zeros(len(t))
    x[(t >= -1) & (t <= 0)] = 2.0
    mask_baja = (t > 0) & (t <= 2)
    x[mask_baja] = 2 - 2 * t[mask_baja]   # de 2 a -2 con pendiente -2
    x[(t >  2) & (t <= 4)] = -2.0
    return t, x

def senal_d():
    """Fig 1d: laplaciana x(t) = exp(-|t|) para |t| <= 3"""
    t = np.arange(-4, 4 + delta, delta)
    x = np.zeros(len(t))
    mask = np.abs(t) <= 3
    x[mask] = np.exp(-np.abs(t[mask]))
    return t, x

# Diccionario de senales continuas (nombre -> funcion)
SENALES_CONT = {
    "Fig 1a: Bipolar  x=2 en[0,3], x=-2 en[3,4]": senal_a,
    "Fig 1b: Rampa    x(t)=-t en [-1,1]":           senal_b,
    "Fig 1c: Trapecio x=2, baja a -2, plano -2":    senal_c,
    "Fig 1d: Laplaciana x(t)=e^(-|t|)":             senal_d,
}


# ─────────────────────────────────────────────────────────────
#  DEFINICION DE SENALES DISCRETAS (Punto 1 del laboratorio)
# ─────────────────────────────────────────────────────────────

def par_discreto_a():
    """
    x[n] = 6 - |n|  para |n| < 6;  0 en otro caso
    h[n] = u[n+5] - u[n-5]  (ventana de 11 muestras)
    """
    n_x = np.arange(-6, 7)
    x = np.where(np.abs(n_x) < 6, 6 - np.abs(n_x), 0).astype(float)

    n_h = np.arange(-6, 7)
    h = np.where((n_h >= -5) & (n_h <= 4), 1.0, 0.0)

    return n_x, x, n_h, h

def par_discreto_b():
    """
    x[n] = u[n+3] - u[n-7]  (ventana de 10 muestras)
    h[n] = (6/7)^n * (u[n] - u[n-9])  (exponencial truncada)
    """
    n_x = np.arange(-4, 9)
    x = np.where((n_x >= -3) & (n_x <= 6), 1.0, 0.0)

    n_h = np.arange(-1, 11)
    h = np.zeros(len(n_h))
    h[(n_h >= 0) & (n_h <= 8)] = (6.0 / 7.0) ** n_h[(n_h >= 0) & (n_h <= 8)]

    return n_x, x, n_h, h

SENALES_DISC = {
    "Par (a): x=6-|n|, h=u[n+5]-u[n-5]": par_discreto_a,
    "Par (b): x=ventana, h=(6/7)^n":      par_discreto_b,
}


# ─────────────────────────────────────────────────────────────
#  LOGICA DE CONVOLUCION PASO A PASO
# ─────────────────────────────────────────────────────────────

def preparar_convolucion_discreta(x, n_x, h, n_h, girar_h=True):
    """
    Prepara los arrays para mostrar el proceso de convolucion discreta.
    Retorna: eje_k, x_k, h_girada, n_y, total_pasos
    Sigue el patron de Untitled20.ipynb del curso.
    """
    lx = len(x)
    lh = len(h)

    if girar_h:
        senal_fija = x
        senal_girar = h
    else:
        senal_fija = h
        senal_girar = x

    # padding para visualizar el deslizamiento
    pad = lh
    eje_k = np.arange(-(pad), lx + pad)
    x_k = np.concatenate((np.zeros(pad), senal_fija, np.zeros(pad)))

    total_pasos = lx + lh - 1
    n_y_inicio = n_x[0] + n_h[0]
    n_y = np.arange(n_y_inicio, n_y_inicio + total_pasos)

    return eje_k, x_k, senal_girar, n_y, total_pasos

def calcular_paso_discreto(paso, eje_k, x_k, senal_girar, total_pasos):
    """
    Para el paso n, calcula: h[n-k] desplazado y el producto v[k].
    Devuelve h_n_k, v_k, y_n_acum (vector acumulado hasta este paso).
    """
    lh = len(senal_girar)
    pad = len(senal_girar)

    # h girada y desplazada (igual que en el cuaderno del curso)
    ceros_izq = paso + 1
    ceros_der = len(x_k) - lh - ceros_izq
    if ceros_der < 0:
        h_n_k = np.concatenate((np.zeros(max(0, ceros_izq)),
                                 senal_girar[::-1],
                                 np.zeros(0)))[:len(x_k)]
    else:
        h_n_k = np.concatenate((np.zeros(ceros_izq),
                                  senal_girar[::-1],
                                  np.zeros(ceros_der)))[:len(x_k)]

    # ajustar longitud
    if len(h_n_k) < len(x_k):
        h_n_k = np.append(h_n_k, np.zeros(len(x_k) - len(h_n_k)))

    v_k = x_k * h_n_k

    return h_n_k, v_k

def preparar_convolucion_continua(t_x, x, t_h, h, girar_h=True):
    """
    Prepara la convolucion continua discretizada.
    Usa la misma grilla de tiempo con paso delta (igual que 'marzo conv.ipynb').
    """
    if girar_h:
        t_fija, s_fija = t_x, x
        t_giro, s_giro = t_h, h
    else:
        t_fija, s_fija = t_h, h
        t_giro, s_giro = t_x, x

    # grilla comun
    t_min = min(t_fija[0], t_giro[0]) - 1
    t_max = max(t_fija[-1], t_giro[-1]) + 1
    t_comun = np.arange(t_min, t_max + delta, delta)

    def interpolar(t_orig, s_orig, t_nuevo):
        s_nuevo = np.zeros(len(t_nuevo))
        for i, ti in enumerate(t_nuevo):
            idx = np.where((t_orig >= ti - delta/2) & (t_orig < ti + delta/2))[0]
            if len(idx) > 0:
                s_nuevo[i] = np.mean(s_orig[idx])
        return s_nuevo

    x_k = interpolar(t_fija, s_fija, t_comun)
    h_orig = interpolar(t_giro, s_giro, t_comun)

    # convolve completo para el resultado final (patron del curso)
    y_total = np.convolve(x_k, h_orig) * delta
    N = len(t_comun)
    t_y = np.arange(2 * t_min, 2 * t_min + len(y_total) * delta, delta)[:len(y_total)]

    total_pasos = len(t_comun)
    return t_comun, x_k, h_orig, t_y, y_total, total_pasos


# ─────────────────────────────────────────────────────────────
#  CLASE PRINCIPAL DE LA APLICACION GUI
# ─────────────────────────────────────────────────────────────

class AppConvolucion:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Convolucion Continua y Discreta — Senales & Sistemas")
        self.ventana.geometry("1100x720")

        self.paso_actual = 0
        self.total_pasos = 0
        self.modo = "continuo"

        # datos actuales de convolucion
        self.datos = {}

        self._crear_controles()
        self._crear_figura()
        self._actualizar_interfaz()

    # ── Controles superiores ──────────────────────────────────

    def _crear_controles(self):
        frame_ctrl = tk.Frame(self.ventana, bg="#f0f0f0", pady=6)
        frame_ctrl.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Dominio
        tk.Label(frame_ctrl, text="Dominio:", bg="#f0f0f0").grid(row=0, column=0, padx=4)
        self.cb_dominio = ttk.Combobox(frame_ctrl, values=["Continuo", "Discreto"],
                                        state="readonly", width=10)
        self.cb_dominio.set("Continuo")
        self.cb_dominio.grid(row=0, column=1, padx=4)
        self.cb_dominio.bind("<<ComboboxSelected>>", self._cambiar_dominio)

        # Senal x
        tk.Label(frame_ctrl, text="Senal x:", bg="#f0f0f0").grid(row=0, column=2, padx=4)
        self.cb_senal_x = ttk.Combobox(frame_ctrl, state="readonly", width=28)
        self.cb_senal_x.grid(row=0, column=3, padx=4)

        # Senal h
        tk.Label(frame_ctrl, text="Senal h:", bg="#f0f0f0").grid(row=0, column=4, padx=4)
        self.cb_senal_h = ttk.Combobox(frame_ctrl, state="readonly", width=28)
        self.cb_senal_h.grid(row=0, column=5, padx=4)

        # Funcion a girar
        tk.Label(frame_ctrl, text="Girar:", bg="#f0f0f0").grid(row=0, column=6, padx=4)
        self.cb_girar = ttk.Combobox(frame_ctrl, values=["h (impulso)", "x (entrada)"],
                                      state="readonly", width=12)
        self.cb_girar.set("h (impulso)")
        self.cb_girar.grid(row=0, column=7, padx=4)

        # Boton iniciar
        tk.Button(frame_ctrl, text="Iniciar", command=self._iniciar,
                  bg="#4CAF50", fg="white", width=8).grid(row=0, column=8, padx=6)

        # Botones de paso
        tk.Button(frame_ctrl, text="◄ Anterior", command=self._paso_anterior,
                  width=9).grid(row=0, column=9, padx=2)
        tk.Button(frame_ctrl, text="Siguiente ►", command=self._paso_siguiente,
                  width=9).grid(row=0, column=10, padx=2)

        # Etiqueta de paso
        self.lbl_paso = tk.Label(frame_ctrl, text="Paso: 0 / 0", bg="#f0f0f0", width=12)
        self.lbl_paso.grid(row=0, column=11, padx=6)

        self._llenar_combos_continuo()

    def _llenar_combos_continuo(self):
        nombres = list(SENALES_CONT.keys())
        self.cb_senal_x["values"] = nombres
        self.cb_senal_h["values"] = nombres
        self.cb_senal_x.set(nombres[0])
        self.cb_senal_h.set(nombres[1])

    def _llenar_combos_discreto(self):
        nombres = list(SENALES_DISC.keys())
        self.cb_senal_x["values"] = nombres
        self.cb_senal_h["values"] = nombres
        self.cb_senal_x.set(nombres[0])
        self.cb_senal_h.set(nombres[0])

    def _cambiar_dominio(self, event=None):
        if self.cb_dominio.get() == "Continuo":
            self.modo = "continuo"
            self._llenar_combos_continuo()
        else:
            self.modo = "discreto"
            self._llenar_combos_discreto()
        self.paso_actual = 0
        self.datos = {}
        self._actualizar_interfaz()

    # ── Figura matplotlib ─────────────────────────────────────

    def _crear_figura(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 6.5))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.ventana)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    # ── Logica de inicio y pasos ──────────────────────────────

    def _iniciar(self):
        self.paso_actual = 0
        girar_h = (self.cb_girar.get() == "h (impulso)")

        if self.modo == "continuo":
            nombre_x = self.cb_senal_x.get()
            nombre_h = self.cb_senal_h.get()
            t_x, x = SENALES_CONT[nombre_x]()
            t_h, h = SENALES_CONT[nombre_h]()
            t_com, x_k, h_orig, t_y, y_total, total_pasos = \
                preparar_convolucion_continua(t_x, x, t_h, h, girar_h)
            self.datos = {
                "t_com": t_com, "x_k": x_k, "h_orig": h_orig,
                "t_y": t_y, "y_total": y_total,
                "t_x": t_x, "x": x, "t_h": t_h, "h": h,
                "girar_h": girar_h,
                "nombre_x": nombre_x, "nombre_h": nombre_h,
            }
            self.total_pasos = total_pasos

        else:  # discreto
            nombre_par = self.cb_senal_x.get()
            if nombre_par in SENALES_DISC:
                n_x, x, n_h, h = SENALES_DISC[nombre_par]()
            else:
                n_x, x, n_h, h = par_discreto_a()

            eje_k, x_k, senal_girar, n_y, total_pasos = \
                preparar_convolucion_discreta(x, n_x, h, n_h, girar_h)
            y_n_acum = np.convolve(x, h)
            self.datos = {
                "n_x": n_x, "x": x, "n_h": n_h, "h": h,
                "eje_k": eje_k, "x_k": x_k,
                "senal_girar": senal_girar,
                "n_y": n_y, "y_n_total": y_n_acum,
                "girar_h": girar_h,
            }
            self.total_pasos = total_pasos

        self.lbl_paso.config(text=f"Paso: 1 / {self.total_pasos}")
        self._actualizar_interfaz()

    def _paso_siguiente(self):
        if self.paso_actual < self.total_pasos - 1:
            self.paso_actual += 1
            self.lbl_paso.config(text=f"Paso: {self.paso_actual + 1} / {self.total_pasos}")
            self._actualizar_interfaz()

    def _paso_anterior(self):
        if self.paso_actual > 0:
            self.paso_actual -= 1
            self.lbl_paso.config(text=f"Paso: {self.paso_actual + 1} / {self.total_pasos}")
            self._actualizar_interfaz()

    # ── Dibujo de la figura ───────────────────────────────────

    def _actualizar_interfaz(self):
        for ax in self.axes.flat:
            ax.cla()

        if not self.datos:
            self._dibujar_bienvenida()
        elif self.modo == "continuo":
            self._dibujar_continuo()
        else:
            self._dibujar_discreto()

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def _dibujar_bienvenida(self):
        ax = self.axes[0, 0]
        ax.set_title("Seleccione senales y presione Iniciar")
        ax.text(0.5, 0.5, "Convolucion\nContinua y Discreta",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes)
        for ax in self.axes.flat[1:]:
            ax.axis("off")

    def _dibujar_continuo(self):
        d = self.datos
        t_com = d["t_com"]
        x_k   = d["x_k"]
        h_orig = d["h_orig"]
        t_y   = d["t_y"]
        y_total = d["y_total"]
        girar_h = d["girar_h"]
        paso = self.paso_actual

        # Subplot 1: senal fija x(t) o h(t)
        ax1 = self.axes[0, 0]
        if girar_h:
            ax1.plot(d["t_x"], d["x"], "b")
            ax1.set_title("x(t) — entrada (fija)")
        else:
            ax1.plot(d["t_h"], d["h"], "g")
            ax1.set_title("h(t) — impulso (fija)")
        ax1.axhline(0, color="k", lw=0.7)
        ax1.grid(True)

        # Subplot 2: senal que se gira h(-tau+t) desplazandose
        ax2 = self.axes[0, 1]
        h_girada = h_orig[::-1]
        # desplazamiento: el paso mapea a posicion en t_com
        step_size = max(1, len(t_com) // self.total_pasos)
        idx_desp = paso * step_size
        t_desp = t_com[idx_desp] if idx_desp < len(t_com) else t_com[-1]

        t_giro = t_com - (t_com[-1] - t_com[0]) + t_desp
        # la senal girada en el subplot 2
        if girar_h:
            ax2.plot(t_giro, h_girada, "g", label=f"h(τ) girada, t={t_desp:.2f}")
            ax2.set_title("h(−τ+t) desplazandose")
        else:
            ax2.plot(t_giro, x_k[::-1], "b", label=f"x(τ) girada, t={t_desp:.2f}")
            ax2.set_title("x(−τ+t) desplazandose")
        ax2.axhline(0, color="k", lw=0.7)
        ax2.axvline(t_desp, color="r", lw=0.8, linestyle="--", label="t actual")
        ax2.legend(fontsize=8)
        ax2.grid(True)

        # Subplot 3: producto (area de solapamiento)
        ax3 = self.axes[1, 0]
        # reconstruir la senal girada sobre t_com para el producto
        senal_girar = h_orig if girar_h else x_k
        h_g = senal_girar[::-1]
        # desplazar h_girada a posicion idx_desp
        h_despl = np.zeros(len(t_com))
        inicio = idx_desp
        fin = inicio + len(h_g)
        if fin <= len(t_com):
            h_despl[inicio:fin] = h_g
        elif inicio < len(t_com):
            h_despl[inicio:] = h_g[:len(t_com) - inicio]

        producto = x_k * h_despl
        ax3.fill_between(t_com, 0, producto, alpha=0.5, color="orange")
        ax3.plot(t_com, producto, "orange")
        ax3.set_title(f"Producto x(τ)·h(t−τ), t={t_desp:.2f}")
        ax3.axhline(0, color="k", lw=0.7)
        ax3.grid(True)

        # Subplot 4: y(t) acumulado hasta el paso actual
        ax4 = self.axes[1, 1]
        n_mostrar = min(idx_desp + step_size, len(t_y))
        ax4.plot(t_y[:n_mostrar], y_total[:n_mostrar], "r", lw=1.5)
        ax4.set_title("y(t) = x(t) * h(t) acumulado")
        ax4.axhline(0, color="k", lw=0.7)
        ax4.grid(True)

        # titulos de ejes
        for ax in self.axes.flat:
            ax.set_xlabel("t")

    def _dibujar_discreto(self):
        d = self.datos
        n_x    = d["n_x"]
        x      = d["x"]
        n_h    = d["n_h"]
        h      = d["h"]
        eje_k  = d["eje_k"]
        x_k    = d["x_k"]
        sg     = d["senal_girar"]
        n_y    = d["n_y"]
        y_tot  = d["y_n_total"]
        girar_h = d["girar_h"]
        paso   = self.paso_actual

        # Subplot 1: x[k] fija
        ax1 = self.axes[0, 0]
        ax1.stem(eje_k, x_k, linefmt="b", markerfmt="bo", basefmt="k")
        ax1.set_title("x[k] — fija")
        ax1.set_xlabel("k")
        ax1.grid(True)

        # Subplot 2: h[n-k] o x[n-k] desplazandose
        ax2 = self.axes[0, 1]
        h_n_k, v_k = calcular_paso_discreto(paso, eje_k, x_k, sg, self.total_pasos)
        if girar_h:
            ax2.stem(eje_k, h_n_k, linefmt="g", markerfmt="gs", basefmt="k")
            ax2.set_title(f"h[n−k], n={n_y[paso] if paso < len(n_y) else '?'}")
        else:
            ax2.stem(eje_k, h_n_k, linefmt="g", markerfmt="gs", basefmt="k")
            ax2.set_title(f"x[n−k], n={n_y[paso] if paso < len(n_y) else '?'}")
        ax2.set_xlabel("k")
        ax2.grid(True)

        # Subplot 3: producto v[k] = x[k]*h[n-k]
        ax3 = self.axes[1, 0]
        ax3.stem(eje_k, v_k, linefmt="orange", markerfmt="D", basefmt="k")
        ax3.set_title(f"v[k] = x[k]·h[n−k], suma = {np.sum(v_k):.3f}")
        ax3.set_xlabel("k")
        ax3.grid(True)

        # Subplot 4: y[n] acumulado
        ax4 = self.axes[1, 1]
        y_acum = np.zeros(len(n_y))
        pasos_mostrar = min(paso + 1, len(y_tot))
        y_acum[:pasos_mostrar] = y_tot[:pasos_mostrar]
        ax4.stem(n_y, y_acum, linefmt="r", markerfmt="r^", basefmt="k")
        ax4.set_title("y[n] = x[n] * h[n] acumulado")
        ax4.set_xlabel("n")
        ax4.grid(True)


# ─────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppConvolucion(ventana)
    ventana.mainloop()
