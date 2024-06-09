import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Uçak ve çevre parametreleri
g = 9.81  # Yerçekimi ivmesi
rho0 = 1.225  # Deniz seviyesindeki hava yoğunluğu
cfcr = 0.92958
cf1 = 0.70057
cf2 = 1068.1
ctcr = 0.95
ctc1 = 146590
ctc2 = 53872
ctc3 = 3.0453e-11
cd0 = 0.025452
k = 0.035815
S = 124.65

# Rüzgar katsayıları (x ve y yönleri için)
wind_coeffs_x = [-21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001]
wind_coeffs_y = [-65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227]

# Giriş değerlerini [-1, 1] aralığına normalize etme fonksiyonu
def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

# x yönündeki rüzgar hızını hesaplama fonksiyonu
def wind_speed_x(lam, phi):
    lam_norm = normalize(lam, -180, 180)  # Boylamı normalize et
    phi_norm = normalize(phi, -90, 90)   # Enlemi normalize et
    poly = np.poly1d(wind_coeffs_x[::-1]) # Katsayılardan polinom oluştur
    return poly([lam_norm, phi_norm])[0] # Rüzgar hızını hesapla

# y yönündeki rüzgar hızını hesaplama fonksiyonu
def wind_speed_y(lam, phi):
    lam_norm = normalize(lam, -180, 180)
    phi_norm = normalize(phi, -90, 90)
    poly = np.poly1d(wind_coeffs_y[::-1])
    return poly([lam_norm, phi_norm])[0]

# Belirli bir yükseklikteki hava yoğunluğunu hesaplama fonksiyonu
def air_density(h):
    return rho0 * (1 - 2.2557e-5 * h) ** 4.2558

# Uçak dinamiklerini modelleyen fonksiyon
def aircraft_dynamics(t, state, u):
    x, y, h, v, psi, m = state  # Durum değişkenleri
    gamma, mu, delta = u       # Kontrol girdileri

    rho = air_density(h)       # Hava yoğunluğu
    Wx = wind_speed_x(x, y)    # x yönündeki rüzgar hızı
    Wy = wind_speed_y(x, y)    # y yönündeki rüzgar hızı

    # Durum denklemleri (hareket denklemleri)
    dxdt = v * np.cos(psi) * np.cos(gamma) + Wx
    dydt = v * np.sin(psi) * np.cos(gamma) + Wy
    dhdt = v * np.sin(gamma)

    # İtme, yakıt tüketimi ve aerodinamik hesaplamaları
    Thrust_max = ctcr * ctc1 * (1 - 3.28 * h / ctc2 + ctc3 * (3.28 * h) ** 2)
    eta = cf1 / 60000 * (1 + 1.943 * v / cf2)
    f = delta * Thrust_max * eta * cfcr
    CL = 2 * m * g / (rho * S * v ** 2 * np.cos(mu))
    CD = cd0 + k * CL ** 2

    # Kalan durum denklemleri
    dvdt = Thrust_max * delta / m - g * np.sin(gamma) - CD * S * rho * v ** 2 / (2 * m)
    dpsidt = (CL * S * rho * v * np.sin(mu)) / (2 * m * np.cos(gamma))
    dmdt = -f

    return np.array([dxdt, dydt, dhdt, dvdt, dpsidt, dmdt])

def cost_function(u_flat, initial_state, time_span, posf):
    N = len(u_flat) // 3
    u = u_flat.reshape((N, 3))
    dt = time_span[1] - time_span[0]
    state = initial_state
    cost = 0
    
    for i in range(N):
        u_current = u[i]
        sol = solve_ivp(lambda t, x: aircraft_dynamics(t, x, u_current), [0, dt], state, method='RK45')
        state = sol.y[:, -1]
        cost += np.linalg.norm(state[:3] - posf) ** 2 + 0.05 * dt + state[-1] * dt
        
    return cost

def optimize_trajectory(initial_state, time_span, posf):
    N = len(time_span) - 1
    u_guess = np.zeros((N, 3))  
    bounds = [(-np.pi/2, np.pi/2), (-np.pi/4, np.pi/4), (0, 1)] * N
    result = minimize(cost_function, u_guess.flatten(), args=(initial_state, time_span, posf), bounds=bounds, method='SLSQP')
    return result.x.reshape((N, 3))

def simulate_flight(initial_state, u_optimal, time_span):
    dt = time_span[1] - time_span[0]
    states = np.zeros((len(time_span), len(initial_state)))
    states[0] = initial_state
    
    for i in range(1, len(time_span)):
        u_current = u_optimal[i-1]
        sol = solve_ivp(lambda t, x: aircraft_dynamics(t, x, u_current), [0, dt], states[i-1], method='RK45')
        states[i] = sol.y[:, -1]
        
    return states

def main():
    flight_plans = [
        {"initial": [2, 40, 8000, 220, 0, 68000], "final": [32, 52, 9000]},
        {"initial": [30, 50, 8000, 200, 225, 58000], "final": [20, 50, 7000]},
        {"initial": [10, 40, 7000, 210, 0, 61000], "final": [10, 53, 7000]},
    ]

    time_span = np.linspace(0, 3600, 361)

    for i, plan in enumerate(flight_plans):
        initial_state = plan["initial"]
        posf = plan["final"]

        u_optimal = optimize_trajectory(initial_state, time_span, posf)
        states = simulate_flight(initial_state, u_optimal, time_span)

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(time_span, states[:, 0], label="x")
        plt.plot(time_span, states[:, 1], label="y")
        plt.plot(time_span, states[:, 2], label="h")
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.legend()
        plt.title(f"Flight {i+1} - Position")

        plt.subplot(3, 1, 2)
        plt.plot(time_span, states[:, 3], label="v")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.legend()
        plt.title(f"Flight {i+1} - Speed")

        plt.subplot(3, 1, 3)
        plt.plot(time_span, states[:, 4], label="ψ")
        plt.plot(time_span, states[:, 5], label="m")
        plt.xlabel("Time (s)")
        plt.ylabel("Heading and Mass")
        plt.legend()
        plt.title(f"Flight {i+1} - Heading and Mass")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
