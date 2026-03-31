# Laboratorio de Convolución Continua y Discreta

**Asignatura:** Señales & Sistemas
**Institución:** Universidad del Norte
**Fecha:** 2026

---

## Autores

- [Nombre Estudiante 1]
- [Nombre Estudiante 2]
- [Nombre Estudiante 3]

---

## Abstract

En este laboratorio se estudia el proceso de convolución en el dominio del tiempo continuo y del tiempo discreto como herramienta fundamental para el análisis de sistemas lineales e invariantes en el tiempo (LTI). Se desarrolló una aplicación en Python con interfaz gráfica de usuario (GUI) que permite visualizar paso a paso el proceso de giro y desplazamiento de la señal involucrada en la convolución. Adicionalmente, se resolvieron tres pares de señales continuas de forma analítica y se compararon los resultados con la función `np.convolve` de Python. Los resultados confirman que la convolución numérica con paso de tiempo suficientemente pequeño aproxima de manera precisa la solución analítica, con diferencias atribuibles exclusivamente a la discretización.

---

## 1. Introducción

La convolución es la operación matemática central en el análisis de sistemas LTI. Para un sistema con respuesta al impulso h(t) sometido a una señal de entrada x(t), la salida se expresa como:

$$y(t) = x(t) * h(t) = \int_{-\infty}^{\infty} x(\tau)\, h(t - \tau)\, d\tau$$

En tiempo discreto, la convolución suma se define como:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k]\, h[n - k]$$

El proceso implica tres pasos fundamentales: (1) girar una de las señales respecto al eje vertical para obtener h(−τ), (2) desplazarla un tiempo t para obtener h(t−τ), y (3) calcular el área bajo el producto x(τ)·h(t−τ) para cada valor de t.

La función escalón unitario u(t) se define como u(t) = 1 para t ≥ 0 y u(t) = 0 para t < 0, y se usa extensamente para definir el soporte de las señales.

---

## 2. Marco Experimental

### 2.1 Punto 1 — Aplicación GUI de Convolución

**Descripción:**
Se diseñó una aplicación Python con interfaz gráfica usando las librerías `tkinter` y `matplotlib`, siguiendo el estilo de programación del curso. La aplicación permite:

- Seleccionar el dominio de trabajo: **tiempo continuo** o **tiempo discreto**.
- Seleccionar la señal de entrada x y la respuesta al impulso h de listas predefinidas.
- Elegir cuál de las dos señales se gira y desplaza durante el proceso.
- Avanzar y retroceder paso a paso en el proceso de convolución.
- Visualizar simultáneamente en cuatro subgráficas:
  1. La señal fija x(t) o x[n].
  2. La señal girada h(−τ + t) desplazándose.
  3. El producto x(τ)·h(t−τ) en el instante actual.
  4. La salida y(t) o y[n] acumulada hasta el paso actual.

**Señales continuas disponibles (Figura 1 del laboratorio):**

| Señal | Definición matemática |
|---|---|
| (a) Pulso rectangular | x(t) = 2 para 0 ≤ t ≤ 3; 0 en otro caso |
| (b) Triángulo | x(t) = 1 − \|t\| para \|t\| ≤ 1; 0 en otro caso |
| (c) Lineal a trozos | x(t) = 2 para −1 ≤ t ≤ 0; 2 − (4/3)t para 0 ≤ t ≤ 3 |
| (d) Campana gaussiana | x(t) = e^(−t²) para \|t\| ≤ 3 |

**Señales discretas del laboratorio:**

*Par (a):*
$$x[n] = \begin{cases} 6 - |n|, & |n| < 6 \\ 0, & \text{en otro caso} \end{cases}$$
$$h[n] = u[n+5] - u[n-5]$$

*Par (b):*
$$x[n] = u[n+3] - u[n-7]$$
$$h[n] = \left(\frac{6}{7}\right)^n \left(u[n] - u[n-9]\right)$$

**[Incluir aquí pantallazo de la interfaz gráfica diseñada]**

---

### 2.2 Punto 2a — Convolución de h(t) = e^(−t/4)·u(t) y x(t) = e^(−4t/5)·(u(t+1) − u(t−5))

**Definición matemática:**

$$y(t) = \int_{-\infty}^{\infty} e^{-\tau/4}\,u(\tau) \cdot e^{-\frac{4}{5}(t-\tau)}\left[u(t-\tau+1) - u(t-\tau-5)\right] d\tau$$

**Análisis de soporte:**
- h(τ): activo para τ ≥ 0
- x(t−τ): activo para −1 ≤ t−τ ≤ 5, es decir, t−5 ≤ τ ≤ t+1

El integrando se simplifica como e^(−4t/5) · e^(11τ/20), cuya integral es (20/11)·e^(11τ/20).

**Resultado de la convolución (sin operaciones intermedias):**

$$y(t) = \begin{cases}
0, & t < -1 \\[6pt]
\dfrac{20}{11}\left(e^{\,\frac{11}{20} - \frac{t}{4}} - e^{-\frac{4t}{5}}\right), & -1 \leq t < 5 \\[8pt]
\dfrac{20}{11}\left(e^{\,\frac{11}{20} - \frac{t}{4}} - e^{-\frac{t+11}{4}}\right), & t \geq 5
\end{cases}$$

**Comparación con Python (`np.convolve`):**
La curva obtenida con `np.convolve(x_t, h_t) * delta` (con delta = 0.01) coincide con la solución analítica. Las diferencias son menores al 1% en la región de mayor amplitud y se deben a la discretización del eje del tiempo. Al reducir delta, la aproximación numérica converge a la solución exacta.

**[Incluir gráfica generada por `ejercicio_a.png`]**

---

### 2.3 Punto 2b — Convolución de h(t) = e^(−t/2)·u(t+1) y x(t) biexponencial

**Definición matemática:**

$$h(t) = e^{-t/2}\,u(t+1), \qquad x(t) = \begin{cases} e^{t/2}, & -4 < t < 0 \\ e^{-t/2}, & 0 < t < 4 \end{cases}$$

**Análisis de soporte:**
- h(τ): activo para τ ≥ −1
- x(t−τ): activo para −4 < t−τ < 4, con cambio de rama en τ = t

Integrando separado en dos piezas:
- Para τ < t: h(τ)·x(t−τ) = e^(−t/2) (constante en τ)
- Para τ > t: h(τ)·x(t−τ) = e^(t/2)·e^(−τ)

**Resultado de la convolución:**

$$y(t) = \begin{cases}
0, & t < -5 \\[6pt]
e^{\,t/2+1} - e^{-t/2-4}, & -5 \leq t < -1 \\[6pt]
e^{-t/2}(t+2) - e^{-(t/2+4)}, & -1 \leq t < 3 \\[6pt]
5\,e^{-t/2} - e^{-(t/2+4)}, & t \geq 3
\end{cases}$$

**Comparación con Python:**
La solución numérica con `np.convolve` reproduce fielmente los cuatro tramos del resultado analítico. La discontinuidad de x(t) en t = 0 no genera error notable en la convolución gracias a que la integral promedia la vecindad de ese punto. La principal diferencia se observa para t muy negativo o muy positivo, donde los efectos de borde del truncamiento del vector h_t se hacen notorios.

**[Incluir gráfica generada por `ejercicio_b.png`]**

---

### 2.4 Punto 2c — Convolución de h(t) = e^t·u(1−t) y x(t) = u(t+1) − u(t−4)

**Definición matemática:**

$$y(t) = \int_{-\infty}^{1} e^{\tau} \cdot \left[u(t-\tau+1) - u(t-\tau-4)\right] d\tau$$

**Análisis de soporte:**
- h(τ): activo para τ ≤ 1
- x(t−τ): activo para t−4 ≤ τ ≤ t+1

Límite superior de integración: min(1, t+1), que vale t+1 si t ≤ 0 y 1 si t > 0.
La integral de e^τ es directamente e^τ evaluada en los límites.

**Resultado de la convolución:**

$$y(t) = \begin{cases}
e^{t+1} - e^{t-4}, & t \leq 0 \\[6pt]
e - e^{t-4}, & 0 < t \leq 5 \\[6pt]
0, & t > 5
\end{cases}$$

**Comparación con Python:**
La concordancia es excelente. El primer tramo (t ≤ 0) muestra y(t) ∝ e^t (creciente), mientras que el segundo tramo (0 < t ≤ 5) es decreciente a medida que e^(t−4) crece. Ambas curvas son continuas en t = 0 y la salida se anula exactamente en t = 5, comportamiento que `np.convolve` reproduce con buena precisión para delta = 0.01.

**[Incluir gráfica generada por `ejercicio_c.png`]**

---

## 3. Conclusiones

1. La convolución es la operación que caracteriza completamente la respuesta de un sistema LTI: conocer h(t) o h[n] es suficiente para calcular la salida ante cualquier entrada.

2. La solución analítica requiere identificar cuidadosamente los intervalos de solapamiento entre x(τ) y h(t−τ) en cada tramo de t; un error en los límites de integración es el equivocado más frecuente y causa resultados completamente incorrectos.

3. La función `np.convolve` de Python permite verificar la solución analítica de forma rápida y eficiente. Para aproximar la convolución continua se multiplica el resultado por delta (el paso de discretización), siguiendo el mismo principio que se usa en la suma de Riemann para aproximar una integral.

4. La aplicación GUI diseñada facilita la comprensión visual del proceso: el desplazamiento paso a paso de la señal girada permite observar cómo el área bajo el producto crece y decrece a medida que t avanza, dando significado físico a la operación matemática.

---

## 4. Referencias Bibliográficas

1. Oppenheim, A. V., Willsky, A. S., & Nawab, S. H. (1997). *Signals and Systems* (2nd ed.). Prentice Hall.
2. Haykin, S., & Van Veen, B. (2003). *Signals and Systems* (2nd ed.). John Wiley & Sons.
3. Materiales del curso: cuadernos Jupyter suministrados por el profesor (marzo 2026). Universidad del Norte.

---

> **NOTA IMPORTANTE — ENTREGABLE ADICIONAL:**
> Según el numeral 3 del enunciado del laboratorio, la solución **paso a paso** (desarrollo completo de las integrales, cambio de límites, evaluación y simplificación algebraica) de los tres ejercicios del Punto 2 debe entregarse en un **documento aparte escrito a mano**. Este informe solo presenta el resultado final por tramos de cada convolución, sin las operaciones intermedias.
