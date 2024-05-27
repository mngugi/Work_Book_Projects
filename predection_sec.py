import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the model
alpha = 0.01    # Rate of new zero-day vulnerability discovery
beta = 0.1     # Rate of exploitation of vulnerabilities
gamma = 0.05   # Random factor for unpredictability
delta = 0.05   # Rate at which vulnerabilities are patched
p = 1.1        # Non-linear growth parameter
iterations = 1000  # Number of iterations for the simulation

# Initialize arrays to store the simulation results
D = np.zeros(iterations + 1)  # Array to store the discovered zero-day vulnerabilities
S = np.zeros(iterations + 1)  # Array to store the number of vulnerable systems
E = np.zeros(iterations + 1)  # Array to store the exploitation events
epsilon = np.random.randn(iterations + 1)  # Array of random values for randomness

# Set initial conditions for the simulation
D[0] = 1  # Initial zero-day vulnerabilities discovered
S[0] = 1  # Initial number of vulnerable systems
E[0] = 0  # Initial exploitation event

# Simulation loop
for n in range(iterations):
    E[n+1] = np.random.choice([0, 1], p=[0.95, 0.05])  # Randomly determine if an exploitation event occurs
    D[n+1] = D[n]**p + alpha - delta * E[n] + gamma * epsilon[n]  # Update discovered vulnerabilities with non-linear growth
    S[n+1] = S[n] + beta * E[n] - delta * E[n]  # Update vulnerable systems
    
    # Print the values at each iteration
    print(f"Iteration {n+1}: E[n+1] = {E[n+1]}, D[n+1] = {D[n+1]}, S[n+1] = {S[n+1]}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(D, label='Discovered Zero-Day Vulnerabilities (D)')
plt.plot(S, label='Vulnerable Systems (S)')
plt.plot(E, label='Exploitation Events (E)')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.title('Zero-Day Vulnerability Prediction Model with Non-linear Growth')
plt.show()
