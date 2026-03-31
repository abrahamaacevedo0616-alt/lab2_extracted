"""
Laboratorio de Convolucion Continua y Discreta
Senales & Sistemas — Universidad del Norte

Interfaz web con Streamlit.
  Ejecutar localmente : python -m streamlit run app_web.py
  Publicar gratis     : https://share.streamlit.io
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ─────────────────────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Convolucion — Senales & Sistemas",
    page_icon="📡",
    layout="wide",
)
st.title("📡 Convolución Continua y Discreta")
st.caption("Señales & Sistemas · Universidad del Norte")

delta = 0.05   # paso de tiempo para Punto 1 (igual que en clase)


# ═══════════════════════════════════════════════════════════════
#  SEÑALES CONTINUAS — Figura 1 del laboratorio
# ═══════════════════════════════════════════════════════════════

def senal_a():
    """Fig 1a: x(t)=2 en [0,3];  x(t)=-2 en (3,4];  0 en otro caso"""
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
    """Fig 1c: x(t)=2 en [-1,0]; baja de 2 a -2 en [0,2]; x(t)=-2 en [2,4]"""
    t = np.arange(-2, 5 + delta, delta)
    x = np.zeros(len(t))
    x[(t >= -1) & (t <= 0)] = 2.0
    mask = (t > 0) & (t <= 2)
    x[mask] = 2.0 - 2.0 * t[mask]
    x[(t >  2) & (t <= 4)] = -2.0
    return t, x

def senal_d():
    """Fig 1d: laplaciana x(t) = e^(-|t|) para |t| <= 3"""
    t = np.arange(-4, 4 + delta, delta)
    x = np.zeros(len(t))
    mask = np.abs(t) <= 3
    x[mask] = np.exp(-np.abs(t[mask]))
    return t, x

SENALES_CONT = {
    "Fig 1a — Bipolar: 2 en [0,3], -2 en (3,4]":  senal_a,
    "Fig 1b — Rampa: x(t) = -t en [-1,1]":         senal_b,
    "Fig 1c — Trapecio: 2 → -2 → plano -2":        senal_c,
    "Fig 1d — Laplaciana: e^(-|t|) en [-3,3]":     senal_d,
}


# ═══════════════════════════════════════════════════════════════
#  SEÑALES DISCRETAS — Punto 1 del laboratorio
# ═══════════════════════════════════════════════════════════════

def par_discreto_a():
    """
    Par (a):
      x[n] = 6-|n| para |n|<6;  0 en otro caso
      h[n] = u[n+5] - u[n-5]  → 1 para -5 <= n <= 4
    """
    n_x = np.arange(-6, 7)
    x   = np.where(np.abs(n_x) < 6, 6 - np.abs(n_x), 0).astype(float)

    n_h = np.arange(-6, 7)
    # u[n+5]=1 si n>=-5; u[n-5]=1 si n>=5 → h=1 para -5<=n<=4
    h   = np.where((n_h >= -5) & (n_h <= 4), 1.0, 0.0)

    return n_x, x, n_h, h

def par_discreto_b():
    """
    Par (b):
      x[n] = u[n+3] - u[n-7]  → 1 para -3 <= n <= 6
      h[n] = (6/7)^n (u[n]-u[n-9]) → (6/7)^n para 0 <= n <= 8
    """
    n_x = np.arange(-4, 9)
    x   = np.where((n_x >= -3) & (n_x <= 6), 1.0, 0.0)

    n_h = np.arange(-1, 11)
    h   = np.zeros(len(n_h))
    mask = (n_h >= 0) & (n_h <= 8)
    h[mask] = (6.0 / 7.0) ** n_h[mask]

    return n_x, x, n_h, h

SENALES_DISC = {
    "Par (a): x[n]=6-|n|,   h[n]=u[n+5]-u[n-5]":    par_discreto_a,
    "Par (b): x[n]=ventana, h[n]=(6/7)^n truncada":  par_discreto_b,
}


# ═══════════════════════════════════════════════════════════════
#  LOGICA DE CONVOLUCION — CONTINUA
# ═══════════════════════════════════════════════════════════════

def interpolar_vectorizado(t_orig, s_orig, t_nuevo):
    """Proyecta s_orig sobre t_nuevo usando la grilla con paso delta."""
    indices = np.round((t_nuevo - t_orig[0]) / delta).astype(int)
    s_nuevo = np.zeros(len(t_nuevo))
    mask = (indices >= 0) & (indices < len(s_orig))
    s_nuevo[mask] = s_orig[indices[mask]]
    return s_nuevo

def preparar_continuo(t_x, x, t_h, h, girar_h):
    """Prepara datos para la animacion de convolucion continua."""
    if girar_h:
        t_fija, s_fija = t_x, x
        t_giro, s_giro = t_h, h
    else:
        t_fija, s_fija = t_h, h
        t_giro, s_giro = t_x, x

    t_min = min(t_fija[0], t_giro[0]) - 0.5
    t_max = max(t_fija[-1], t_giro[-1]) + 0.5
    t_com = np.arange(t_min, t_max + delta, delta)

    x_k   = interpolar_vectorizado(t_fija, s_fija, t_com)
    h_arr = interpolar_vectorizado(t_giro, s_giro, t_com)

    # resultado final con convolve (patron del curso)
    y_total = np.convolve(x_k, h_arr) * delta
    t_y = np.arange(2*t_min, 2*t_min + len(y_total)*delta, delta)[:len(y_total)]

    return t_com, x_k, h_arr, t_y, y_total


# ═══════════════════════════════════════════════════════════════
#  LOGICA DE CONVOLUCION — DISCRETA
# ═══════════════════════════════════════════════════════════════

def preparar_discreto(x, n_x, h, n_h, girar_h):
    """Prepara datos para la animacion de convolucion discreta."""
    if girar_h:
        s_fija, s_girar = x, h
    else:
        s_fija, s_girar = h, x

    lx  = len(s_fija)
    lh  = len(s_girar)
    pad = lh

    eje_k = np.arange(-pad, lx + pad)
    x_k   = np.concatenate((np.zeros(pad), s_fija, np.zeros(pad)))

    total_pasos = lx + lh - 1
    n_y_ini = n_x[0] + n_h[0]
    n_y     = np.arange(n_y_ini, n_y_ini + total_pasos)

    y_total = np.convolve(s_fija, s_girar)
    return eje_k, x_k, s_girar, n_y, y_total, total_pasos


# ═══════════════════════════════════════════════════════════════
#  FIGURAS — PUNTO 1
# ═══════════════════════════════════════════════════════════════

def figura_continuo(t_com, x_k, h_arr, t_y, y_total, paso, total_pasos,
                    t_x, x, t_h, h, girar_h, nombre_x, nombre_h):

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Proceso de Convolución — Tiempo Continuo",
                 fontsize=13, fontweight="bold")

    frac     = paso / max(total_pasos - 1, 1)
    idx_desp = int(frac * (len(t_com) - 1))
    t_actual = t_com[idx_desp]

    # ── 1: señal fija ────────────────────────────────────────
    ax1 = axes[0, 0]
    if girar_h:
        ax1.plot(t_x, x, "b", lw=1.8)
        ax1.set_title("x(t) — entrada (fija)")
    else:
        ax1.plot(t_h, h, "g", lw=1.8)
        ax1.set_title("h(t) — impulso (fija)")
    ax1.axhline(0, color="k", lw=0.6); ax1.grid(True, alpha=0.4); ax1.set_xlabel("t")

    # ── 2: señal girada deslizándose ─────────────────────────
    ax2 = axes[0, 1]
    senal_girar_arr = h_arr if girar_h else x_k
    h_girada = senal_girar_arr[::-1]
    t_centro = (t_com[0] + t_com[-1]) / 2
    t_giro   = t_com - t_centro + t_actual
    label_giro = "h(t−τ)" if girar_h else "x(t−τ)"
    ax2.plot(t_giro, h_girada, "g" if girar_h else "b", lw=1.8, label=label_giro)
    ax2.axvline(t_actual, color="r", lw=1.2, ls="--", label=f"t = {t_actual:.2f}")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_title(f"{'h' if girar_h else 'x'}(t−τ) deslizándose")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4); ax2.set_xlabel("τ")

    # ── 3: producto en el instante actual ────────────────────
    ax3 = axes[1, 0]
    h_despl = np.zeros(len(t_com))
    inicio  = idx_desp
    fin     = inicio + len(h_girada)
    if fin <= len(t_com):
        h_despl[inicio:fin] = h_girada
    elif inicio < len(t_com):
        h_despl[inicio:] = h_girada[:len(t_com) - inicio]
    producto = x_k * h_despl
    ax3.fill_between(t_com, 0, producto, alpha=0.55, color="orange")
    ax3.plot(t_com, producto, color="darkorange", lw=1.4)
    ax3.set_title(f"Producto x(τ)·h(t−τ),  t = {t_actual:.2f}")
    ax3.axhline(0, color="k", lw=0.6); ax3.grid(True, alpha=0.4); ax3.set_xlabel("τ")

    # ── 4: y(t) acumulado ────────────────────────────────────
    ax4 = axes[1, 1]
    step_size = max(1, len(t_y) // total_pasos)
    n_mostrar = min((paso + 1) * step_size, len(t_y))
    ax4.plot(t_y[:n_mostrar], y_total[:n_mostrar], "r", lw=2)
    ax4.set_title("y(t) = x(t) ∗ h(t)  acumulado")
    ax4.axhline(0, color="k", lw=0.6); ax4.grid(True, alpha=0.4); ax4.set_xlabel("t")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def figura_discreta(eje_k, x_k, senal_girar, n_y, y_total, paso, girar_h):

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Proceso de Convolución — Tiempo Discreto",
                 fontsize=13, fontweight="bold")

    lh      = len(senal_girar)
    n_actual = n_y[paso] if paso < len(n_y) else n_y[-1]

    # h[n-k] desplazado (mismo patron que Untitled20.ipynb del curso)
    ceros_izq = paso + 1
    ceros_der = max(0, len(x_k) - lh - ceros_izq)
    h_n_k = np.concatenate((
        np.zeros(ceros_izq),
        senal_girar[::-1],
        np.zeros(ceros_der)
    ))[:len(x_k)]
    if len(h_n_k) < len(x_k):
        h_n_k = np.append(h_n_k, np.zeros(len(x_k) - len(h_n_k)))

    v_k = x_k * h_n_k

    # y[n] acumulado hasta el paso actual
    y_acum = np.zeros(len(n_y))
    pasos_mostrar = min(paso + 1, len(y_total))
    y_acum[:pasos_mostrar] = y_total[:pasos_mostrar]

    # ── 1: x[k] fija ─────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.stem(eje_k, x_k, linefmt="b", markerfmt="bo", basefmt="k-")
    ax1.set_title("x[k] — señal fija")
    ax1.set_xlabel("k"); ax1.grid(True, alpha=0.4)

    # ── 2: h[n-k] o x[n-k] deslizándose ─────────────────────
    ax2 = axes[0, 1]
    label2 = f"h[n−k]" if girar_h else "x[n−k]"
    ax2.stem(eje_k, h_n_k, linefmt="g", markerfmt="gs", basefmt="k-")
    ax2.set_title(f"{label2}  (n = {n_actual})")
    ax2.set_xlabel("k"); ax2.grid(True, alpha=0.4)

    # ── 3: producto v[k] ──────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.stem(eje_k, v_k, linefmt="darkorange", markerfmt="D", basefmt="k-")
    ax3.set_title(f"v[k] = x[k]·h[n−k],   Σv[k] = {np.sum(v_k):.3f}")
    ax3.set_xlabel("k"); ax3.grid(True, alpha=0.4)

    # ── 4: y[n] acumulado ────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.stem(n_y, y_acum, linefmt="r", markerfmt="r^", basefmt="k-")
    ax4.set_title("y[n] = x[n] ∗ h[n]  acumulado")
    ax4.set_xlabel("n"); ax4.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def figura_senal_discreta(n_x, x, n_h, h, nombre):
    """Muestra x[n] y h[n] antes de iniciar la convolucion."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Señales discretas — {nombre}", fontsize=12, fontweight="bold")

    ax1.stem(n_x, x, linefmt="b", markerfmt="bo", basefmt="k-")
    ax1.set_title("x[n]"); ax1.set_xlabel("n"); ax1.grid(True, alpha=0.4)

    ax2.stem(n_h, h, linefmt="g", markerfmt="gs", basefmt="k-")
    ax2.set_title("h[n]"); ax2.set_xlabel("n"); ax2.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ═══════════════════════════════════════════════════════════════
#  FIGURAS — PUNTO 2
# ═══════════════════════════════════════════════════════════════

d2 = 0.01   # paso fino para Punto 2

def figura_punto2a():
    """
    h(t) = e^(-t/4)*u(t),  x(t) = e^(-4t/5)*(u(t+1)-u(t-5))
    Analitica:
      y = 0                                           t < -1
      y = (20/11)*(e^(11/20-t/4) - e^(-4t/5))        -1 <= t < 5
      y = (20/11)*(e^(11/20-t/4) - e^(-(t+11)/4))    t >= 5
    """
    th  = np.arange(0, 20 + d2, d2)
    h_t = np.exp(-th / 4)

    tx  = np.arange(-1, 5 + d2, d2)
    x_t = np.exp(-4*tx / 5)

    t1 = np.arange(-1, 5, d2);  t2 = np.arange(5, 20 + d2, d2)
    y1 = (20.0/11)*(np.exp(11.0/20 - t1/4) - np.exp(-4*t1/5))
    y2 = (20.0/11)*(np.exp(11.0/20 - t2/4) - np.exp(-(t2+11)/4))
    ty_mat = np.concatenate((t1, t2));  y_mat = np.concatenate((y1, y2))

    y_py  = np.convolve(x_t, h_t) * d2
    ty_py = np.arange(-1, -1 + len(y_py)*d2, d2)[:len(y_py)]

    return _figura3(tx, x_t, th[:500], h_t[:500], ty_mat, y_mat, ty_py, y_py,
                    "Ejercicio (a): h(t)=e^(−t/4)·u(t),   x(t)=e^(−4t/5)·(u(t+1)−u(t−5))")

def figura_punto2b():
    """
    h(t) = e^(-t/2)*u(t+1),  x(t)=e^(t/2) -4<t<0; e^(-t/2) 0<t<4
    Analitica:
      y = 0                             t < -5
      y = e^(t/2+1) - e^(-t/2-4)       -5 <= t < -1
      y = e^(-t/2)*(t+2) - e^(-(t/2+4)) -1 <= t < 3
      y = 5*e^(-t/2) - e^(-(t/2+4))    t >= 3
    """
    th  = np.arange(-1, 15 + d2, d2)
    h_t = np.exp(-th / 2)

    tx1 = np.arange(-4 + d2, 0, d2);  tx2 = np.arange(d2, 4, d2)
    tx  = np.concatenate((tx1, tx2))
    x_t = np.concatenate((np.exp(tx1/2), np.exp(-tx2/2)))

    ta = np.arange(-5, -1, d2);  tb = np.arange(-1, 3, d2);  tc = np.arange(3, 15+d2, d2)
    ya = np.exp(ta/2+1) - np.exp(-ta/2-4)
    yb = np.exp(-tb/2)*(tb+2) - np.exp(-(tb/2+4))
    yc = 5*np.exp(-tc/2) - np.exp(-(tc/2+4))
    ty_mat = np.concatenate((ta, tb, tc));  y_mat = np.concatenate((ya, yb, yc))

    y_py  = np.convolve(x_t, h_t) * d2
    ty_py = np.arange(tx[0]+th[0], tx[0]+th[0]+len(y_py)*d2, d2)[:len(y_py)]

    return _figura3(tx, x_t, th[:200], h_t[:200], ty_mat, y_mat, ty_py, y_py,
                    "Ejercicio (b): h(t)=e^(−t/2)·u(t+1),   x(t)=e^(t/2) o e^(−t/2) según intervalo")

def figura_punto2c():
    """
    h(t) = e^t*u(1-t),  x(t) = u(t+1)-u(t-4)
    Analitica:
      y = e^(t+1) - e^(t-4)   t <= 0
      y = e - e^(t-4)          0 < t <= 5
      y = 0                    t > 5
    """
    th  = np.arange(-8, 1 + d2, d2)
    h_t = np.exp(th)

    tx  = np.arange(-1, 4 + d2, d2)
    x_t = np.ones(len(tx))

    ta = np.arange(-8, 0 + d2, d2);  tb = np.arange(0 + d2, 5, d2)
    ya = np.exp(ta+1) - np.exp(ta-4)
    yb = np.e - np.exp(tb-4)
    ty_mat = np.concatenate((ta, tb));  y_mat = np.concatenate((ya, yb))

    y_py  = np.convolve(x_t, h_t) * d2
    ty_py = np.arange(tx[0]+th[0], tx[0]+th[0]+len(y_py)*d2, d2)[:len(y_py)]

    return _figura3(tx, x_t, th, h_t, ty_mat, y_mat, ty_py, y_py,
                    "Ejercicio (c): h(t)=e^t·u(1−t),   x(t)=u(t+1)−u(t−4)")

def _figura3(tx, x_t, th, h_t, ty_mat, y_mat, ty_py, y_py, titulo):
    """Figura de 3 subplots: x(t), h(t), comparacion y(t)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(titulo, fontsize=11, fontweight="bold")

    axes[0].plot(tx, x_t, "b", lw=1.8)
    axes[0].set_title("x(t) — señal de entrada")
    axes[0].axhline(0, color="k", lw=0.6); axes[0].grid(True, alpha=0.4); axes[0].set_xlabel("t")

    axes[1].plot(th, h_t, "g", lw=1.8)
    axes[1].set_title("h(t) — respuesta al impulso")
    axes[1].axhline(0, color="k", lw=0.6); axes[1].grid(True, alpha=0.4); axes[1].set_xlabel("t")

    axes[2].plot(ty_mat, y_mat, "r",   lw=2.2, label="y(t) analítica")
    axes[2].plot(ty_py,  y_py,  "k--", lw=1.5, label="y(t) con np.convolve")
    axes[2].set_title("y(t) — Comparación: analítica (rojo) vs np.convolve (negro)")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.4); axes[2].set_xlabel("t")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════
#  INTERFAZ STREAMLIT — PESTAÑAS
# ═══════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs([
    "📊 Punto 1 — Proceso de Convolución (paso a paso)",
    "📐 Punto 2 — Solución Matemática + np.convolve",
])


# ══════════════════════════════════════════════════════════════
#  PESTAÑA 1
# ══════════════════════════════════════════════════════════════
with tab1:

    # ── Sidebar (global en Streamlit) ─────────────────────────
    with st.sidebar:
        st.header("⚙️ Punto 1 — Controles")

        dominio = st.radio("**Dominio**", ["Tiempo Continuo", "Tiempo Discreto"])
        st.divider()

        if dominio == "Tiempo Continuo":
            nombre_x = st.selectbox(
                "**Señal x(t)**", list(SENALES_CONT.keys()), index=0)
            nombre_h = st.selectbox(
                "**Señal h(t)**", list(SENALES_CONT.keys()), index=1)
            nombre_par = list(SENALES_DISC.keys())[0]   # valor por defecto
        else:
            nombre_par = st.selectbox(
                "**Par de señales**", list(SENALES_DISC.keys()))
            nombre_x = list(SENALES_CONT.keys())[0]    # valor por defecto
            nombre_h = list(SENALES_CONT.keys())[1]    # valor por defecto

        girar_lbl = st.radio("**Función que se gira**",
                              ["h — impulso", "x — entrada"])
        girar_h = (girar_lbl == "h — impulso")
        st.divider()
        iniciar = st.button("▶️ Iniciar / Reiniciar", use_container_width=True)

    # ── Estado de sesión ──────────────────────────────────────
    for key, val in [("paso", 0), ("total_pasos", 1),
                     ("datos", None), ("config", {})]:
        if key not in st.session_state:
            st.session_state[key] = val

    config_actual = {
        "dominio": dominio,
        "senal_x": nombre_x,
        "senal_h": nombre_h,
        "par":     nombre_par,
        "girar_h": girar_h,
    }

    # ── Cargar datos al pulsar Iniciar o cambiar config ───────
    if iniciar or st.session_state.config != config_actual:
        st.session_state.paso   = 0
        st.session_state.config = config_actual

        if dominio == "Tiempo Continuo":
            t_x, x = SENALES_CONT[nombre_x]()
            t_h, h = SENALES_CONT[nombre_h]()
            t_com, x_k, h_arr, t_y, y_total = \
                preparar_continuo(t_x, x, t_h, h, girar_h)
            st.session_state.total_pasos = 60
            st.session_state.datos = dict(
                modo="continuo",
                t_com=t_com, x_k=x_k, h_arr=h_arr,
                t_y=t_y, y_total=y_total,
                t_x=t_x, x=x, t_h=t_h, h=h,
                girar_h=girar_h, nombre_x=nombre_x, nombre_h=nombre_h,
            )
        else:
            n_x, x, n_h, h = SENALES_DISC[nombre_par]()
            eje_k, x_k, sg, n_y, y_tot, total_p = \
                preparar_discreto(x, n_x, h, n_h, girar_h)
            st.session_state.total_pasos = total_p
            st.session_state.datos = dict(
                modo="discreto",
                n_x=n_x, x=x, n_h=n_h, h=h,
                eje_k=eje_k, x_k=x_k, sg=sg,
                n_y=n_y, y_total=y_tot,
                girar_h=girar_h, nombre_par=nombre_par,
            )

    # ── Mostrar señales discretas antes de iniciar ────────────
    if dominio == "Tiempo Discreto" and st.session_state.datos is not None:
        d = st.session_state.datos
        if d["modo"] == "discreto":
            st.markdown("#### Señales del par seleccionado")
            fig_prev = figura_senal_discreta(
                d["n_x"], d["x"], d["n_h"], d["h"], d["nombre_par"])
            st.pyplot(fig_prev); plt.close(fig_prev)
            st.divider()

    # ── Controles de paso ─────────────────────────────────────
    total = st.session_state.total_pasos
    c1, c2, c3, c4 = st.columns([1, 1, 3, 1])
    with c1:
        if st.button("⏮ Inicio"):
            st.session_state.paso = 0
    with c2:
        if st.button("◀ Anterior"):
            st.session_state.paso = max(0, st.session_state.paso - 1)
    with c4:
        if st.button("Siguiente ▶"):
            st.session_state.paso = min(total - 1, st.session_state.paso + 1)
    with c3:
        val_slider = st.slider("Paso", 0, max(total - 1, 1),
                               value=st.session_state.paso,
                               label_visibility="collapsed")
        st.session_state.paso = val_slider

    paso = st.session_state.paso
    st.caption(f"Paso {paso + 1} de {total}")

    # ── Figura principal ──────────────────────────────────────
    d = st.session_state.datos
    if d is None:
        st.info("Selecciona señales en el panel izquierdo y presiona **▶️ Iniciar**.")
    elif d["modo"] == "continuo":
        fig = figura_continuo(
            d["t_com"], d["x_k"], d["h_arr"], d["t_y"], d["y_total"],
            paso, total,
            d["t_x"], d["x"], d["t_h"], d["h"],
            d["girar_h"], d["nombre_x"], d["nombre_h"],
        )
        st.pyplot(fig); plt.close(fig)
    else:
        fig = figura_discreta(
            d["eje_k"], d["x_k"], d["sg"],
            d["n_y"], d["y_total"], paso, d["girar_h"],
        )
        st.pyplot(fig); plt.close(fig)

    st.divider()
    st.markdown(
        "**Cómo usar:** Selecciona dominio y señales → **▶️ Iniciar** "
        "→ avanza con **Siguiente ▶** o arrastra el slider."
    )


# ══════════════════════════════════════════════════════════════
#  PESTAÑA 2
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Punto 2 — Solución matemática y comparación con np.convolve")
    st.markdown(
        "Selecciona un ejercicio para ver: señales x(t) y h(t), "
        "resultado **analítico** por tramos (rojo) y resultado con "
        "**np.convolve** (negro punteado)."
    )
    st.divider()

    ej = st.radio(
        "**Ejercicio:**",
        ["(a)  h(t) = e^(−t/4)·u(t),    x(t) = e^(−4t/5)·(u(t+1) − u(t−5))",
         "(b)  h(t) = e^(−t/2)·u(t+1),  x(t) = e^(t/2) o e^(−t/2) según intervalo",
         "(c)  h(t) = e^t·u(1−t),        x(t) = u(t+1) − u(t−4)"],
    )

    if ej.startswith("(a)"):
        st.latex(r"""
y(t) = \begin{cases}
0 & t < -1 \\[4pt]
\dfrac{20}{11}\!\left(e^{\,\frac{11}{20}-\frac{t}{4}} - e^{-\frac{4t}{5}}\right)
  & -1 \leq t < 5 \\[8pt]
\dfrac{20}{11}\!\left(e^{\,\frac{11}{20}-\frac{t}{4}} - e^{-\frac{t+11}{4}}\right)
  & t \geq 5
\end{cases}
""")
        with st.spinner("Generando gráfica..."):
            fig = figura_punto2a()
        st.pyplot(fig); plt.close(fig)
        st.success("Las curvas coinciden. La leve diferencia en los bordes se debe a la discretización (Δt = 0.01).")

    elif ej.startswith("(b)"):
        st.latex(r"""
y(t) = \begin{cases}
0 & t < -5 \\[4pt]
e^{\,t/2+1} - e^{-t/2-4} & -5 \leq t < -1 \\[4pt]
e^{-t/2}(t+2) - e^{-(t/2+4)} & -1 \leq t < 3 \\[4pt]
5\,e^{-t/2} - e^{-(t/2+4)} & t \geq 3
\end{cases}
""")
        with st.spinner("Generando gráfica..."):
            fig = figura_punto2b()
        st.pyplot(fig); plt.close(fig)
        st.success("Las curvas coinciden. Las diferencias en extremos se deben al truncamiento finito del vector h(t).")

    else:
        st.latex(r"""
y(t) = \begin{cases}
e^{t+1} - e^{t-4} & t \leq 0 \\[4pt]
e - e^{t-4}        & 0 < t \leq 5 \\[4pt]
0                  & t > 5
\end{cases}
""")
        with st.spinner("Generando gráfica..."):
            fig = figura_punto2c()
        st.pyplot(fig); plt.close(fig)
        st.success("Concordancia excelente. La salida se anula exactamente en t = 5.")

    st.divider()
    st.warning(
        "⚠️ **Nota del laboratorio:** La solución **paso a paso** "
        "(desarrollo completo de las integrales) debe entregarse en un "
        "**documento aparte escrito a mano**, según el enunciado del Punto 3."
    )
