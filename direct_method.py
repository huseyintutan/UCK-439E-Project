import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Uçak parametreleri (B737-800)
C_fcr = 0.92958
C_f1 = 0.70057
C_f2 = 1068.1
C_Tcr = 0.95
C_Tc1 = 146590
C_Tc2 = 53872
C_Tc3 = 3.0453e-11
C_d0 = 0.025452
k = 0.035815
S = 124.65

# Rüzgar katsayıları
wind_coeffs_x = [-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001]
wind_coeffs_y = [-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227]

def wind_speed(lam, phi, coeffs):
    return (coeffs[0] + coeffs[1]*lam + coeffs[2]*phi + coeffs[3]*lam*phi +
            coeffs[4]*lam**2 + coeffs[5]*phi**2 + coeffs[6]*lam**2*phi +
            coeffs[7]*lam*phi**2 + coeffs[8]*lam**2*phi**2)

def aircraft_dynamics(t, state, gamma, mu, delta):
    x, y, h, v, psi, m = state
    rho = 1.225 * (1 - 2.2257e-5 * h) ** 4.2586
    CL = (2 * m * 9.81) / (rho * S * v**2 * np.cos(mu))
    CD = C_d0 + k * CL**2
    Tmax = C_Tcr * (C_Tc1 - 3.28 * h / C_Tc2 + C_Tc3 * (3.28 * h)**2)
    eta = C_f1 / 60000 * (1 + 1.943 * v / C_f2)
    f = delta * Tmax * eta * C_fcr

    Wx = wind_speed(x, y, wind_coeffs_x)
    Wy = wind_speed(x, y, wind_coeffs_y)

    dxdt = v * np.cos(psi) * np.cos(gamma) + Wx
    dydt = v * np.sin(psi) * np.cos(gamma) + Wy
    dhdt = v * np.sin(gamma)
    dvdt = (Tmax * delta / m) - 9.81 * np.sin(gamma) - (CD * S * rho * v**2) / (2 * m)
    dpsidt = (CL * S * rho * v / (2 * m)) * (np.sin(mu) / np.cos(gamma))
    dmdt = -f

    return [dxdt, dydt, dhdt, dvdt, dpsidt, dmdt]

def objective_function(t, state, delta):
    m = state[5]
    return 0.05 + delta * C_fcr * C_f1 * Tmax / (60000 * (1 + 1.943 * v / C_f2))

# Uçuş 1 için başlangıç ve hedef koşulları
lambda_0, phi_0, h0, v0, psi_0, m0 = 2, 40, 8000, 220, 0, 68000
lambda_d, phi_d, hd = 32, 52, 9000

initial_conditions = [lambda_0, phi_0, h0, v0, psi_0, m0]

# Kontrol girdileri (optimize edilecek)
gamma = 0.1
mu = 0.1
delta = 0.5

# Simülasyon için zaman aralığı
t_span = [0, 5000]  # Gerekirse ayarlayın
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(aircraft_dynamics, t_span, initial_conditions, args=(gamma, mu, delta), t_eval=t_eval)

x, y, h, v, psi, m = sol.y
time = sol.t

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectory in x-y Plane')

plt.subplot(2, 2, 2)
plt.plot(time, h)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs Time')

plt.subplot(2, 2, 3)
plt.plot(time, v)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed vs Time')

plt.subplot(2, 2, 4)
plt.plot(time, m)
plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.title('Mass vs Time')

plt.tight_layout()
plt.show()

# Optimize edilecek fonksiyon
def to_optimize(controls):
    gamma, mu, delta = controls
    sol = solve_ivp(aircraft_dynamics, t_span, initial_conditions, args=(gamma, mu, delta), t_eval=t_eval)
    final_state = sol.y[:, -1]
    return objective_function(t_span[1], final_state, delta)

initial_guess = [0.1, 0.1, 0.5]

result = minimize(to_optimize, initial_guess, bounds=[(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (0, 1)])
optimal_controls = result.x
print("Optimal controls:", optimal_controls)
