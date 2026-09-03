"""Parametros fisicos y adimensionales del pendulo de Wilberforce."""

import numpy as np

# 1. Parametros fisicos reales
m = 0.4905
k = 2.627
delta = 7.443 * (10**-4)
I = 1.39 * (10**-4)
epsilon = 9.27 * (10**-3)

z0_real = 0.1         # m
z0_dot_real = 0.0     # m/s
theta0_real = 0.0     # rad
theta0_dot_real = 0.0 # rad/s

# 2. Factores de normalizacion
omega_z = np.sqrt(k / m)
Z0 = z0_real if z0_real != 0 else 1.0
Theta0 = 1.0

# 3. Parametros adimensionales
K = (m * delta) / (k * I)
A = (epsilon * Theta0) / (2 * k * Z0)
B = (m * epsilon * Z0) / (2 * k * I * Theta0)

# 4. Condiciones iniciales adimensionales
u0 = z0_real / Z0
v0 = theta0_real / Theta0
u0_dot = z0_dot_real / (Z0 * omega_z)
v0_dot = theta0_dot_real / (Theta0 * omega_z)