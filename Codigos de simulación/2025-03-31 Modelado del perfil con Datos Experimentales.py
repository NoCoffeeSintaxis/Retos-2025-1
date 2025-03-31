import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

# Datos originales
dx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 20, 
               22, 23, 25, 27, 28, 30, 32, 33, 35, 37, 38, 40, 42, 43, 45, 47, 
               48, 50, 52, 53, 55, 57, 58, 60, 61, 62, 63, 64, 65])
dy = np.array([0.6, 0.8, 1.4, 2.8, 3.4, 3.8, 4.6, 5.1, 5.8, 6.2, 6.5, 7.0, 7.4, 
               7.7, 8.2, 9.3, 9.9, 10.3, 10.7, 10.9, 11.2, 11.7, 12.0, 12.2, 12.8, 
               13.0, 13.5, 13.9, 13.9, 14.4, 15.0, 15.1, 15.3, 15.6, 15.6, 15.7, 
               16.3, 16.5, 16.7, 16.9, 17.0, 17.3, 17.4, 17.5, 17.8, 18.2, 18.6])

# Interpolación con spline
spline = UnivariateSpline(dx, dy, s=1)
dx_full = np.linspace(0, 65, 200)
dy_spline = spline(dx_full)

# Definir la función suavizada
def smooth_cone_model(r, A, B, C, D):
    return A * np.sqrt(r + B) + C + D * r**2

params_smooth_cone, _ = curve_fit(smooth_cone_model, dx, dy, p0=[-5, 1, 10, 0.0005])

def refined_smooth_model(r, A, B, C, D):
    return smooth_cone_model(r, A, B, C, D)

dy_smooth = refined_smooth_model(dx_full, *params_smooth_cone)

# Suavizado en la unión con el perfil final
blend_zone_start, blend_zone_end = 60, 62
blend_indices = (dx_full >= blend_zone_start) & (dx_full <= blend_zone_end)
blend_weights = np.linspace(0, 1, np.sum(blend_indices))

dy_refined = dy_spline.copy()
dy_refined[blend_indices] = (1 - blend_weights) * dy_spline[blend_indices] + blend_weights * dy_smooth[blend_indices]

# Graficar resultado final
plt.figure(figsize=(8,6))
plt.plot(dx, dy, 'ro', label='Datos originales')
plt.plot(dx_full, dy_refined, 'c-', linewidth=2, label='Versión optimizada')
plt.axis('equal')
plt.xlabel("Distancia desde el centro (cm)")
plt.ylabel("Altura de la tela (cm)")
plt.title("Perfil final optimizado")
plt.legend()
plt.grid()
plt.show()
