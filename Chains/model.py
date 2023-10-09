"""Model of chain swimming."""

import numpy as np
import matplotlib.pyplot as plt

THICKNESS = 20e-3
PITCH_NORMAL = 3.37
RADIUS_NORMAL = 0.47
LENGTH = 8.5
ETA = 11.5e-3 #Pa.s Ficoll not dialized
ASPECT_RATIO = 2 #0.18 or 2 : 0.18 Lighthill, 2 Gray and Hancock
LONG_AXIS = 1.4  #semi long axis of ellipse
SMALL_AXIS = 0.5

def calculate_velocity(n: int, alpha: float) -> float:
    """Calculate the velocity of the chain."""
    A0 = (4 * np.pi * ETA * LONG_AXIS * n) / (np.log(2 * n * LONG_AXIS / SMALL_AXIS) - 0.5)

    L = LENGTH + 2 * alpha * (n - 1) * LONG_AXIS

    kn = 8 * np.pi * ETA / (2 * np.log(2 * PITCH_NORMAL / THICKNESS) + 1)
    kt = 4 * np.pi * ETA / (2 * np.log(2 * PITCH_NORMAL / THICKNESS) - 1)

    psi = np.arctan(ASPECT_RATIO * np.pi * RADIUS_NORMAL / PITCH_NORMAL)
    
    Af = L * (kn * np.sin(psi) ** 2 + kt * np.cos(psi) ** 2) / np.cos(psi)
    Bf = L * PITCH_NORMAL * np.sin(psi) ** 2 * (kn - kt) / (2 * np.pi * np.cos(psi))

    return Bf / (Af + A0)

res = {}
alpha_val = [0, 1]

plt.figure()

for alpha in alpha_val:
    vel = []
    for n in range(1, 10):
        vel.append(calculate_velocity(n, alpha) / calculate_velocity(1, 1))
    res[alpha] = vel
    plt.plot(range(1, 10), vel, "o", label=str(alpha))
plt.ylabel("V / Vsingle")
plt.xlabel("Chain length")
plt.legend(title=r"$\alpha$")
plt.show(block=True)