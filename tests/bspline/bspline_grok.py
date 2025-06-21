import numpy as np
from scipy.interpolate import BSpline
from scipy.special import roots_legendre
from scipy.linalg import eigh

def generate_exponential_knots(num_interior_knots, r_min, r_max):
    knots = [r_min * (r_max / r_min)**(i / num_interior_knots) for i in range(num_interior_knots)]
    return knots

def get_quad_points(interval, Nle):
    a, b = interval
    x, w = roots_legendre(Nle)
    x = (x + 1) * (b - a) / 2 + a
    w = w * (b - a) / 2
    return x, w

def MsC_potential_vec(alpha: float, r: np.ndarray) -> np.ndarray:
    """Calculates the 1D Morse-Coulomb potential for an array of positions r."""
    D = 1 / alpha
    beta = 1 / (alpha * np.sqrt(2))
    
    coulomb = -1 / np.sqrt(r * r + alpha * alpha)
    morse = D * (np.exp(-2 * beta * r) - 2 * np.exp(-beta * r))
    
    return np.where(r > 0, coulomb, morse)

n = 1000  # Reduced for demonstration
k = 3   # Reduced for demonstration
rmin = 1
rmax = 2300.0
Nle = 20
alpha_value = 0.001

num_interior_knots = n - k
t_interior = generate_exponential_knots(num_interior_knots, rmin, rmax)
t = [rmin]*(k+1) + list(t_interior) + [rmax]*(k+1)

intervals = [(t[m], t[m+1]) for m in range(len(t)-1)]
all_x, all_w = [], []
for interval in intervals:
    x, w = get_quad_points(interval, Nle)
    all_x.extend(x)
    all_w.extend(w)

c = [0]*n
bspline_obj = BSpline(t, c, k)
B = bspline_obj.design_matrix(all_x).toarray()
derivative_bspline_obj = bspline_obj.derivative()
dB = derivative_bspline_obj.design_matrix(all_x).toarray()

V_all_x = MsC_potential_vec(alpha_value, np.array(all_x)-2)
S = np.dot(B.T, np.dot(np.diag(all_w), B))
HKinetic = np.dot(dB.T, np.dot(np.diag(all_w), dB))
HPotential = np.dot(B.T, np.dot(np.diag(all_w), (V_all_x[:, None] * B)))
HP = HKinetic + HPotential

eigenvalues, eigenvectors = eigh(a=HP, b=S)
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

for i in range(n):
    norm = np.sqrt(np.dot(eigenvectors[:, i].T @ S, eigenvectors[:, i]))
    eigenvectors[:, i] /= norm

print("Eigenvalues:", eigenvalues)