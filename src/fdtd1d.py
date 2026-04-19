import numpy as np
import matplotlib.pyplot as plt

C = 1.0

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

def panel_transfer_matrix(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    omega = 2.0 * np.pi * freq
    eps_c = eps_r - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(mu_r * eps_c)
    eta = np.sqrt(mu_r / eps_c)

    gd = gamma * d
    ch = np.cosh(gd)
    sh = np.sinh(gd)

    Phi = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi[:, 0, 0] = ch
    Phi[:, 0, 1] = eta * sh
    Phi[:, 1, 0] = sh / eta
    Phi[:, 1, 1] = ch
    return Phi

# ── BONUS: multilayer panel ──
def stack_transfer_matrix(freq, layers):
    freq = np.atleast_1d(np.asarray(freq, dtype=complex))
    Phi_total = np.zeros((len(freq), 2, 2), dtype=complex)
    Phi_total[:, 0, 0] = 1.0
    Phi_total[:, 1, 1] = 1.0
    for layer in layers:
        Phi_i = panel_transfer_matrix(
            freq, d=layer['d'],
            eps_r=layer.get('eps_r', 1.0),
            sigma=layer.get('sigma', 0.0),
            mu_r=layer.get('mu_r', 1.0),
        )
        Phi_new = np.zeros_like(Phi_total)
        Phi_new[:, 0, 0] = Phi_total[:, 0, 0]*Phi_i[:, 0, 0] + Phi_total[:, 0, 1]*Phi_i[:, 1, 0]
        Phi_new[:, 0, 1] = Phi_total[:, 0, 0]*Phi_i[:, 0, 1] + Phi_total[:, 0, 1]*Phi_i[:, 1, 1]
        Phi_new[:, 1, 0] = Phi_total[:, 1, 0]*Phi_i[:, 0, 0] + Phi_total[:, 1, 1]*Phi_i[:, 1, 0]
        Phi_new[:, 1, 1] = Phi_total[:, 1, 0]*Phi_i[:, 0, 1] + Phi_total[:, 1, 1]*Phi_i[:, 1, 1]
        Phi_total = Phi_new
    return Phi_total
# ── END BONUS ──

def RT_from_transfer_matrix(Phi):
    A, B, C_, D = Phi[:, 0, 0], Phi[:, 0, 1], Phi[:, 1, 0], Phi[:, 1, 1]
    denom = A + B + C_ + D
    return (A + B - C_ - D) / denom, 2.0 / denom

def reflection_transmission(freq, d, eps_r=1.0, sigma=0.0, mu_r=1.0):
    return RT_from_transfer_matrix(panel_transfer_matrix(freq, d, eps_r, sigma, mu_r))

class FDTD1D:
    mu0 = 1.0
    eps0 = 1.0

    def __init__(self, x, boundaries=None, x_o=None, pert=None, pert_dir=False,
                 n_pml=10, pml_order=3, pml_R0=1e-6):
        self.x_physical = x
        self.dx = x[1] - x[0]
        self.dt = self.dx / C
        self.N_physical = len(x)
        self.boundaries = boundaries
        self.x_o = x_o
        self.pert = pert
        self.pert_dir = pert_dir

        # PML parameters
        self.n_pml = n_pml
        self.pml_order = pml_order
        self.pml_R0 = pml_R0

        # Check if PML is used
        self.use_pml_left = boundaries is not None and boundaries[0] == 'PML'
        self.use_pml_right = boundaries is not None and boundaries[1] == 'PML'

        # Extend domain if PML is used
        self.Npml_left = n_pml if self.use_pml_left else 0
        self.Npml_right = n_pml if self.use_pml_right else 0

        # Build extended grid
        if self.use_pml_left or self.use_pml_right:
            x_left = np.array([])
            x_right = np.array([])
            if self.use_pml_left:
                x_left = x[0] - self.dx * np.arange(self.Npml_left, 0, -1)
            if self.use_pml_right:
                x_right = x[-1] + self.dx * np.arange(1, self.Npml_right + 1)
            self.x = np.concatenate([x_left, x, x_right])
        else:
            self.x = x

        self.xH = (self.x[1:] + self.x[:-1]) / 2.0
        self.N = len(self.x)

        # Fields
        self.e = np.zeros(self.N)
        self.h = np.zeros(self.N - 1)
        self.t = 0.0

        # Material properties
        self.sig = np.zeros(self.N)
        self.sig_H = np.zeros(self.N - 1)  # sigma for H (staggered)
        self.eps_r = np.ones(self.N)
        self.eps = self.eps0 * self.eps_r

        # Setup PML conductivity profile
        if self.use_pml_left or self.use_pml_right:
            self._setup_pml()

        # Field recording probes
        self.probes = []

    def _setup_pml(self):
        """Configure PML conductivity profile."""
        eta0 = np.sqrt(self.mu0 / self.eps0)
        d_pml = self.n_pml * self.dx
        sigma_max = (self.pml_order + 1) * np.log(1.0 / self.pml_R0) / (2.0 * eta0 * d_pml)

        if self.use_pml_left:
            for i in range(self.Npml_left):
                dist = (self.Npml_left - i) * self.dx
                self.sig[i] = sigma_max * (dist / d_pml) ** self.pml_order
            for i in range(self.Npml_left):
                dist = (self.Npml_left - i - 0.5) * self.dx
                if dist > 0:
                    self.sig_H[i] = sigma_max * (dist / d_pml) ** self.pml_order

        if self.use_pml_right:
            idx_start = self.N - self.Npml_right
            for i in range(self.Npml_right):
                dist = (i + 1) * self.dx
                self.sig[idx_start + i] = sigma_max * (dist / d_pml) ** self.pml_order
            idx_start_H = self.N - 1 - self.Npml_right
            for i in range(self.Npml_right):
                dist = (i + 0.5) * self.dx
                if dist > 0:
                    self.sig_H[idx_start_H + i] = sigma_max * (dist / d_pml) ** self.pml_order

    def add_probe(self, x_pos, record_e=True, record_h=True):
        idx_e = np.argmin(np.abs(self.x - x_pos))
        idx_h = np.argmin(np.abs(self.xH - x_pos)) if len(self.xH) > 0 else 0
        self.probes.append({
            'x': x_pos,
            'idx_e': idx_e,
            'idx_h': idx_h,
            'record_e': record_e,
            'record_h': record_h,
            'e_history': [],
            'h_history': []
        })

    def clear_probes(self):
        for p in self.probes:
            p['e_history'] = []
            p['h_history'] = []

    def get_probe_data(self):
        return [
            {
                'x': p['x'],
                'e': np.array(p['e_history']),
                'h': np.array(p['h_history'])
            } for p in self.probes
        ]

    def load_initial_field(self, e0):
        """Load initial E field. e0 should be defined on the physical domain."""
        if self.use_pml_left or self.use_pml_right:
            self.e[self.Npml_left:self.Npml_left + self.N_physical] = e0.copy()
        else:
            self.e = e0.copy()

    def _step(self):
        r = self.dt / self.dx
        self.eps = self.eps0 * self.eps_r

        # Update coefficients for E
        ca = (2 * self.eps - self.sig * self.dt) / (2 * self.eps + self.sig * self.dt)
        cb = (2 * self.dt / self.dx) / (2 * self.eps + self.sig * self.dt)

        # Update coefficients for H (with magnetic conductivity for PML)
        sig_H_star = self.sig_H * (self.mu0 / self.eps0)
        da = (2 * self.mu0 - sig_H_star * self.dt) / (2 * self.mu0 + sig_H_star * self.dt)
        db = (2 * self.dt / self.dx) / (2 * self.mu0 + sig_H_star * self.dt)

        if self.boundaries is not None:
            if self.boundaries[0] == 'mur':
                e_old_left_0 = self.e[0]
                e_old_left_1 = self.e[1]
            if self.boundaries[1] == 'mur':
                e_old_right_0 = self.e[-1]
                e_old_right_1 = self.e[-2]

        if self.pert_dir and self.pert is not None and self.x_o is not None and self.t != 0.0:
            idx = np.argmin(np.abs(self.xH - self.x_o))
            self.h[idx] += self.pert(self.t)

        # Update E
        self.e[1:-1] = ca[1:-1] * self.e[1:-1] - cb[1:-1] * (self.h[1:] - self.h[:-1])

        # Apply boundary conditions
        if self.boundaries is not None:
            if self.boundaries[0] == 'PEC':
                self.e[0] = 0.0
            elif self.boundaries[0] == 'PML':
                self.e[0] = 0.0
            elif self.boundaries[0] == 'periodic':
                self.e[0] = ca[0] * self.e[0] - cb[0] * (self.h[0] - self.h[-1])
                self.e[-1] = self.e[0]
            elif self.boundaries[0] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[0] = e_old_left_1 + mur_coeff * (self.e[1] - e_old_left_0)
            elif self.boundaries[0] == 'PMC':
                self.e[0] -= 2 * r * self.h[0]

            if self.boundaries[1] == 'PEC':
                self.e[-1] = 0.0
            elif self.boundaries[1] == 'PML':
                self.e[-1] = 0.0
            elif self.boundaries[1] == 'mur':
                mur_coeff = (C * self.dt - self.dx) / (C * self.dt + self.dx)
                self.e[-1] = e_old_right_1 + mur_coeff * (self.e[-2] - e_old_right_0)
            elif self.boundaries[1] == 'PMC':
                self.e[-1] += 2 * r * self.h[-1]

        # Source perturbation
        if self.pert is not None and self.x_o is not None:
            idx = np.argmin(np.abs(self.x - self.x_o))
            self.e[idx] += self.pert(self.t + self.dt/2 + self.dx/2/C)

        # Update H (with PML coefficients)
        self.h = da * self.h - db * (self.e[1:] - self.e[:-1])

        for p in self.probes:
            if p['record_e']:
                p['e_history'].append(self.e[p['idx_e']])
            if p['record_h']:
                p['h_history'].append(self.h[p['idx_h']])

        self.t += self.dt

    def run_until(self, t_final):
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
        self.t = t_final

    def get_e(self):
        """Return E field on the physical domain only."""
        if self.use_pml_left or self.use_pml_right:
            return self.e[self.Npml_left:self.Npml_left + self.N_physical].copy()
        return self.e.copy()

    def get_h(self):
        """Return H field on the physical domain only."""
        if self.use_pml_left or self.use_pml_right:
            return self.h[self.Npml_left:self.Npml_left + self.N_physical - 1].copy()
        return self.h.copy()

    def get_e_full(self):
        """Return E field including PML regions."""
        return self.e.copy()

    def get_h_full(self):
        """Return H field including PML regions."""
        return self.h.copy()
