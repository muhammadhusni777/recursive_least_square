import numpy as np

# Inisialisasi parameter
lambda_ = 0.95  # Forgetting Factor

# Estimasi parameter awal untuk b0, b1, dan -a1. NIM : 23223303. a = 0; b =3
a1 = 0
b0 = 3
b1 = 0

theta_hat = [b0, b1, -a1]#np.zeros((3, 1))  

P = 10000 * np.eye(3)  # Matriks kovariansi awal
N = 1000  # Jumlah iterasi
noise_variance = 0.5**2  # Varians noise

# Input sistem
u = np.zeros(N)
u[50:] = 1  # Impuls pada t=50

# Output sistem
y = np.zeros(N)

# Simulasi RLS dengan input impuls
for t in range(1, N):
    phi = np.array([[u[t]], [u[t-1]], [y[t-1]]])  # Vektor regressor
    K = P.dot(phi) / (lambda_ + phi.T.dot(P).dot(phi))  # Gain RLS
    e_t = np.random.normal(0, np.sqrt(noise_variance))  # White noise
    print(e_t)
    y[t] = -1.5 * y[t-1] + 2 * u[t-1] + e_t  # Update output dengan white noise
    theta_hat = theta_hat + K * (y[t] - phi.T.dot(theta_hat))  # Update estimasi parameter
    #print(theta_hat[0], theta_hat[1], theta_hat[2])
    P = (P - K.dot(phi.T).dot(P)) / lambda_  # Update matriks kovariansi

# Hasil estimasi parameter
print("Estimasi parameter setelah 100 iterasi:")
print(theta_hat)
