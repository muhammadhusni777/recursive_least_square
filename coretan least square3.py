import numpy as np
import matplotlib.pyplot as plt

# Inisialisasi parameter
lambda_ = 0.96  # Forgetting Factor

# Estimasi parameter awal untuk b0, b1, dan -a1. NIM : 23223303. a = 0; b =3
a1 = 0
b0 = 3
b1 = 0

t_memory = []
a_memory = []
b_memory = []

theta_hat = [b0, b1, -a1]

P = 10000 * np.eye(3)  # Matriks kovariansi awal
print(P)
N = 100  # Jumlah iterasi
noise_variance = 0.8**2  # Varians noise

# Input sistem
u = np.zeros(N)
u[50:] = 1  # Impuls pada t=50

# Output sistem
y = np.zeros(N)

# Simulasi RLS dengan input impuls
for t in range(1, N):
    if (t==1):
        print(theta_hat)
    
    phi = np.array([[u[t]], [u[t-1]], [y[t-1]]])  # Vektor regressor
    K = P.dot(phi) / (lambda_ + phi.T.dot(P).dot(phi))  # Gain RLS
    e_t = np.random.normal(0, np.sqrt(noise_variance))  # White noise
    
    #print(e_t)
    y[t] = -1.5 * y[t-1] + 2 * u[t-1] + e_t  # Update output dengan white noise
    theta_hat = theta_hat + K * (y[t] - phi.T.dot(theta_hat))  # Update estimasi parameter
    #print(theta_hat[1][1])
    #a_memory = np.append(a_memory, theta_hat[2], axis = 0)
    
    P = (P - K.dot(phi.T).dot(P)) / lambda_  # Update matriks kovariansi

    t_memory.append(t)
    a_memory.append(theta_hat[1][2])
    b_memory.append(theta_hat[2][0])  

# Hasil estimasi parameter
'''
print("Estimasi parameter setelah 100 iterasi:")
print(theta_hat)
print(a_memory)
print(t)
'''

# Plot the fitted line
plt.plot(t_memory, a_memory, color='red', label='a')
plt.plot(t_memory, b_memory, color='blue', label='b')

plt.legend()
plt.show()

