import numpy as np
import matplotlib.pyplot as plt

# Create a time vector
t = np.linspace(0, 4, 1000)  # Adjust the time range and resolution as needed

# Define the function f(t)
f_t = t * (t >= 0) - 2 * (t - 1) * (t >= 1) + 2 * (t - 2) * (t >= 2) - 2 * (t - 3) * (t >= 3) + 2 * (t - 4) * (t >= 4) - 2 * (t - 5) * (t >= 5) + 2 * (t - 6) * (t >= 6)

# Plot the function f(t)
plt.figure(figsize=(8, 4))
plt.plot(t, f_t, label='f(t)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Ramp?')
plt.grid(True)
plt.legend()
plt.show()
