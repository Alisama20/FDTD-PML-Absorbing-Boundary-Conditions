import numpy as np

C = 1.0  

def gaussian2d(x, y, x0, y0, sigma):
    return np.exp(-0.5 * (((x - x0)**2 + (y - y0)**2) / sigma**2))


class FDTD2D:    
    mu0 = 1.0
    eps0 = 1.0
    
    def __init__(self, x, y, boundaries=None, n_pml=10, pml_order=3, pml_R0=1e-6):
        self.x_physical = x
        self.y_physical = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dt = 1.0 / (C * np.sqrt(1.0/self.dx**2 + 1.0/self.dy**2))
        
        self.Nx_physical = len(x)
        self.Ny_physical = len(y)
        
        self.boundaries = boundaries if boundaries else ('PEC', 'PEC', 'PEC', 'PEC')
        
        self.n_pml = n_pml
        self.pml_order = pml_order
        self.pml_R0 = pml_R0

        self.use_pml_left = self.boundaries[0] == 'PML'
        self.use_pml_right = self.boundaries[1] == 'PML'
        self.use_pml_bottom = self.boundaries[2] == 'PML'
        self.use_pml_top = self.boundaries[3] == 'PML'

        self.Npml_left = n_pml if self.use_pml_left else 0
        self.Npml_right = n_pml if self.use_pml_right else 0
        self.Npml_bottom = n_pml if self.use_pml_bottom else 0
        self.Npml_top = n_pml if self.use_pml_top else 0

        self.Nx = self.Nx_physical + self.Npml_left + self.Npml_right
        self.Ny = self.Ny_physical + self.Npml_bottom + self.Npml_top
        
        self._build_extended_grid()
        
        self.Ez = np.zeros((self.Nx, self.Ny))
        self.Hx = np.zeros((self.Nx, self.Ny - 1))
        self.Hy = np.zeros((self.Nx - 1, self.Ny))

        self.sig = np.zeros((self.Nx, self.Ny))
        self.sig_Hx = np.zeros((self.Nx, self.Ny - 1))
        self.sig_Hy = np.zeros((self.Nx - 1, self.Ny))
        self.eps_r = np.ones((self.Nx, self.Ny))

        if any([self.use_pml_left, self.use_pml_right, 
                self.use_pml_bottom, self.use_pml_top]):
            self._setup_pml()
        
        self.t = 0.0
    
    def _build_extended_grid(self):
        x_left = np.array([])
        x_right = np.array([])
        if self.use_pml_left:
            x_left = self.x_physical[0] - self.dx * np.arange(self.Npml_left, 0, -1)
        if self.use_pml_right:
            x_right = self.x_physical[-1] + self.dx * np.arange(1, self.Npml_right + 1)
        self.x = np.concatenate([x_left, self.x_physical, x_right])

        y_bottom = np.array([])
        y_top = np.array([])
        if self.use_pml_bottom:
            y_bottom = self.y_physical[0] - self.dy * np.arange(self.Npml_bottom, 0, -1)
        if self.use_pml_top:
            y_top = self.y_physical[-1] + self.dy * np.arange(1, self.Npml_top + 1)
        self.y = np.concatenate([y_bottom, self.y_physical, y_top])
    
    def _setup_pml(self):
        eta0 = np.sqrt(self.mu0 / self.eps0)

        if self.use_pml_left or self.use_pml_right:
            d_pml_x = self.n_pml * self.dx
            sigma_max_x = (self.pml_order + 1) * np.log(1.0 / self.pml_R0) / (2.0 * eta0 * d_pml_x)
        else:
            sigma_max_x = 0.0
            d_pml_x = 1.0
            
        if self.use_pml_bottom or self.use_pml_top:
            d_pml_y = self.n_pml * self.dy
            sigma_max_y = (self.pml_order + 1) * np.log(1.0 / self.pml_R0) / (2.0 * eta0 * d_pml_y)
        else:
            sigma_max_y = 0.0
            d_pml_y = 1.0

        i_left = self.Npml_left
        i_right = self.Npml_left + self.Nx_physical
        j_bottom = self.Npml_bottom
        j_top = self.Npml_bottom + self.Ny_physical

        for i in range(self.Nx):
            for j in range(self.Ny):
                sx, sy = 0.0, 0.0
                
                if i < i_left:
                    dist = (i_left - i) * self.dx
                    sx = sigma_max_x * (dist / d_pml_x) ** self.pml_order
                elif i >= i_right:
                    dist = (i - i_right + 1) * self.dx
                    sx = sigma_max_x * (dist / d_pml_x) ** self.pml_order
                
                if j < j_bottom:
                    dist = (j_bottom - j) * self.dy
                    sy = sigma_max_y * (dist / d_pml_y) ** self.pml_order
                elif j >= j_top:
                    dist = (j - j_top + 1) * self.dy
                    sy = sigma_max_y * (dist / d_pml_y) ** self.pml_order
                
                self.sig[i, j] = sx + sy
        
        for i in range(self.Nx):
            for j in range(self.Ny - 1):
                sy = 0.0
                j_pos = j + 0.5
                
                if j_pos < j_bottom:
                    dist = (j_bottom - j_pos) * self.dy
                    sy = sigma_max_y * (dist / d_pml_y) ** self.pml_order
                elif j_pos >= j_top:
                    dist = (j_pos - j_top + 1) * self.dy
                    sy = sigma_max_y * (dist / d_pml_y) ** self.pml_order
                
                self.sig_Hx[i, j] = sy
        
        for i in range(self.Nx - 1):
            for j in range(self.Ny):
                sx = 0.0
                i_pos = i + 0.5
                
                if i_pos < i_left:
                    dist = (i_left - i_pos) * self.dx
                    sx = sigma_max_x * (dist / d_pml_x) ** self.pml_order
                elif i_pos >= i_right:
                    dist = (i_pos - i_right + 1) * self.dx
                    sx = sigma_max_x * (dist / d_pml_x) ** self.pml_order
                
                self.sig_Hy[i, j] = sx
    
    def load_initial_field(self, Ez0): 
        i_start = self.Npml_left
        j_start = self.Npml_bottom
        self.Ez[i_start:i_start + self.Nx_physical, 
                j_start:j_start + self.Ny_physical] = Ez0.copy()
    
    def _step(self):
        eps = self.eps0 * self.eps_r
        ca = (2 * eps - self.sig * self.dt) / (2 * eps + self.sig * self.dt)
        cb = (2 * self.dt) / (2 * eps + self.sig * self.dt)

        sig_Hx_star = self.sig_Hx * (self.mu0 / self.eps0)
        da_x = (2 * self.mu0 - sig_Hx_star * self.dt) / (2 * self.mu0 + sig_Hx_star * self.dt)
        db_x = (2 * self.dt) / (2 * self.mu0 + sig_Hx_star * self.dt)

        sig_Hy_star = self.sig_Hy * (self.mu0 / self.eps0)
        da_y = (2 * self.mu0 - sig_Hy_star * self.dt) / (2 * self.mu0 + sig_Hy_star * self.dt)
        db_y = (2 * self.dt) / (2 * self.mu0 + sig_Hy_star * self.dt)

        self.Hx = da_x * self.Hx - db_x / self.dy * (self.Ez[:, 1:] - self.Ez[:, :-1])

        self.Hy = da_y * self.Hy + db_y / self.dx * (self.Ez[1:, :] - self.Ez[:-1, :])

        dHy_dx = (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) / self.dx
        dHx_dy = (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) / self.dy
        self.Ez[1:-1, 1:-1] = (ca[1:-1, 1:-1] * self.Ez[1:-1, 1:-1] + 
                               cb[1:-1, 1:-1] * (dHy_dx - dHx_dy))

        self.Ez[0, :] = 0.0
        self.Ez[-1, :] = 0.0
        self.Ez[:, 0] = 0.0
        self.Ez[:, -1] = 0.0
        
        self.t += self.dt
    
    def add_source(self, i, j, value):
        i_idx = self.Npml_left + i
        j_idx = self.Npml_bottom + j
        self.Ez[i_idx, j_idx] += value
    
    def run_until(self, t_final):
        n_steps = round((t_final - self.t) / self.dt)
        for _ in range(n_steps):
            self._step()
        self.t = t_final
    
    def run_steps(self, n_steps, source_func=None, source_pos=None):
        for _ in range(n_steps):
            if source_func is not None and source_pos is not None:
                self.add_source(source_pos[0], source_pos[1], source_func(self.t))
            self._step()
    
    def get_Ez(self):
        i_start = self.Npml_left
        j_start = self.Npml_bottom
        return self.Ez[i_start:i_start + self.Nx_physical,
                       j_start:j_start + self.Ny_physical].copy()
    
    def get_Ez_full(self):
        return self.Ez.copy()
    
    def get_Hx(self):
        i_start = self.Npml_left
        j_start = self.Npml_bottom
        return self.Hx[i_start:i_start + self.Nx_physical,
                       j_start:j_start + self.Ny_physical - 1].copy()
    
    def get_Hy(self):
        i_start = self.Npml_left
        j_start = self.Npml_bottom
        return self.Hy[i_start:i_start + self.Nx_physical - 1,
                       j_start:j_start + self.Ny_physical].copy()
