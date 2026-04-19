import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D, C, gaussian, panel_transfer_matrix, RT_from_transfer_matrix, stack_transfer_matrix

def test_fdtd_solves_basic_propagation():
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)

    t_final = 0.2
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()

    e_expected = 0.5 * gaussian(x, -t_final*C, sigma) \
     + 0.5 * gaussian(x, t_final*C, sigma)

    plt.plot(x, e_solved)
    plt.plot(x, e_expected)

    assert np.corrcoef(e_solved, e_expected)[0,1] > 0.99

def test_fdtd_PEC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('PEC', 'PEC')
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = - initial_e
    h_expected = np.zeros_like(h_solved)

    assert np.corrcoef(e_solved, e_expected)[0,1] > 0.99
    assert np.allclose(h_solved, h_expected, atol=0.01)

def test_fdtd_periodic_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    boundaries = ('periodic', 'periodic')
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    initial_h = np.zeros_like(initial_e[:-1])

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = initial_e
    h_expected = initial_h

    assert np.corrcoef(e_solved, e_expected)[0, 1] > 0.99
    assert np.allclose(h_solved, h_expected, atol=1e-10)

def test_fdtd_PMC_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('PMC', 'PMC')
    x0 = 0.0
    sigma = 0.05
    initial_h = gaussian(xH, x0, sigma)
    initial_e = np.zeros_like(x)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    L = xMax - xMin
    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    h_expected = - initial_h
    e_expected = np.zeros_like(e_solved)

    assert np.corrcoef(h_solved, h_expected)[0,1] > 0.99
    assert np.max(np.abs(e_solved - e_expected)) < 1e-8

def test_fdtd_perturbation():
    xMax = 1
    xMin = -1
    L = xMax - xMin
    x = np.linspace(xMin,xMax,201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur','mur') 
    x_o = 0
    initial_e = np.zeros_like(x)

    def my_pert(t):
        return np.sin(2*np.pi*2*C/L*t)

    fdtd = FDTD1D(x,boundaries,x_o,pert = lambda t: my_pert(t))
    fdtd.load_initial_field(initial_e)
    t_final = L / (2*C) 
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = my_pert(L/C/2 - np.abs(x)/C)/2
    h_expected = my_pert(L/C/2 - xH/C + fdtd.dt/2)/2

    assert np.allclose(e_solved, e_expected, atol=1e-1)
    assert np.allclose(h_solved, h_expected, atol=1e-1)

def test_fdtd_eh_perturbation():
    xMax = 1
    xMin = -1
    L = xMax - xMin
    x = np.linspace(xMin,xMax,201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur','mur') 
    x_o = 0
    initial_e = np.zeros_like(x)

    def my_pert(t):
        return gaussian(t, 0.1, 0.02)
    fdtd = FDTD1D(x,boundaries,x_o,pert = lambda t: my_pert(t), pert_dir = True)
    fdtd.load_initial_field(initial_e)
    t_final = L / (2*C) 
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    e_expected = my_pert(L/C/2 - np.abs(x)/C)
    h_expected = my_pert(L/C/2 - xH/C + fdtd.dt/2)

    e_expected[x<0.1] = 0
    h_expected[xH<0.1] = 0

    assert np.allclose(e_solved, e_expected, atol=1e-3)
    assert np.allclose(h_solved, h_expected, atol=1e-3)

def test_fdtd_mur_boundary_conditions():
    xMax = 1
    xMin = -1
    x = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    x0 = 0.0
    sigma = 0.05

    initial_e = gaussian(x, x0, sigma)
    initial_h = -gaussian(xH, x0, sigma)

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    t_final = 1.2
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    assert np.allclose(e_solved, 0.0, atol=1e-2)
    assert np.allclose(h_solved, 0.0, atol=1e-2)

def test_fdtd_dissipative_exact():
    xMax = 1.0
    xMin = -1.0
    L = xMax - xMin
    x = np.linspace(xMin, xMax, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('periodic', 'periodic')

    x0 = 0.0
    k = np.pi  
    initial_e = np.sin((x-x0)*k)
    initial_h = np.zeros_like(initial_e[:-1])

    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()

    sigma_val = 1.0
    eps_r_val = 2.0

    fdtd.sig = np.full_like(x, sigma_val)       
    fdtd.eps_r = np.full_like(x, eps_r_val)
    fdtd.eps = fdtd.eps0 * fdtd.eps_r

    gamma = sigma_val / (2 * fdtd.eps0 * eps_r_val)
    w0_sq = (k**2) / (fdtd.mu0 * fdtd.eps0 * eps_r_val)
    wd = np.sqrt(w0_sq - gamma**2)

    t_final = L / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    t_func_E = np.exp(-gamma * t_final) * (np.cos(wd * t_final) - (gamma / wd) * np.sin(wd * t_final))
    t_func_H = np.exp(-gamma * t_final) * (k / (fdtd.mu0 * wd)) * np.sin(wd * t_final)

    e_expected = t_func_E * np.sin(k * (x - x0))
    h_expected = -t_func_H * np.cos(k * (xH - x0)) 

    assert np.allclose(e_solved, e_expected, atol=1e-2)
    assert np.allclose(h_solved, h_expected, atol=1e-2)

def test_fdtd_conductive_panel_reflection():
    L = 4.0
    N = 4001
    x = np.linspace(0, L, N)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    pulse_x0 = 0.8
    sigma = 0.06

    panel_center = L / 2
    panel_d = 0.2
    panel_left = panel_center - panel_d / 2
    panel_right = panel_center + panel_d / 2
    eps_r_panel = 4.0
    sigma_panel = 0.5 

    obs_left_idx = np.argmin(np.abs(x - (panel_left - 0.4)))
    obs_right_idx = np.argmin(np.abs(x - (panel_right + 0.4)))

    initial_e = gaussian(x, pulse_x0, sigma)
    initial_h = gaussian(xH, pulse_x0, sigma)

    t_final = 2.5 * L

    # Simulation WITH panel
    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()
    fdtd.eps_r = np.where((x >= panel_left)  & (x  <= panel_right), eps_r_panel, 1.0)
    fdtd.sig = np.where((x  >= panel_left)  & (x  <= panel_right), sigma_panel, 0.0)

    n_steps = round(t_final / fdtd.dt)
    E_left_panel = np.zeros(n_steps)
    E_right_panel = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd._step()
        E_left_panel[i] = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # Reference simulation WITHOUT panel (free space)
    fdtd_ref = FDTD1D(x, boundaries)
    fdtd_ref.load_initial_field(initial_e)
    fdtd_ref.h = initial_h.copy()

    E_left_ref = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i] = fdtd_ref.e[obs_left_idx]
        E_right_ref[i] = fdtd_ref.e[obs_right_idx]

    # Extract R(f), T(f) via FFT
    dt = fdtd.dt
    E_ref_fft = np.fft.rfft(E_left_panel - E_left_ref)
    E_trans_fft = np.fft.rfft(E_right_panel)
    E_inc_fft = np.fft.rfft(E_right_ref)
    freq = np.fft.rfftfreq(n_steps, d=dt)

    valid = np.abs(E_inc_fft)  > 1e-10 * np.max(np.abs(E_inc_fft))
    R_fdtd = np.zeros_like(freq, dtype=complex)
    T_fdtd = np.zeros_like(freq, dtype=complex)
    R_fdtd[valid] = E_ref_fft[valid] / E_inc_fft[valid]
    T_fdtd[valid] = E_trans_fft[valid] / E_inc_fft[valid]

    # Analytical R(f), T(f) via Transfer Matrix
    f_bw = 1.0 / (2.0 * np.pi * sigma)
    band = (freq  > 0.1)  & (freq  < 1.5 * f_bw)

    Phi = panel_transfer_matrix(freq[band], panel_d, eps_r_panel, sigma_panel)
    R_anal, T_anal = RT_from_transfer_matrix(Phi)

    assert np.corrcoef(np.abs(R_fdtd[band]), np.abs(R_anal))[0, 1]  > 0.99
    assert np.corrcoef(np.abs(T_fdtd[band]), np.abs(T_anal))[0, 1]  > 0.99

def test_fdtd_multilayer_panel_reflection():
    L = 4.0
    N = 4001
    x = np.linspace(0, L, N)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('mur', 'mur')
    pulse_x0 = 0.8
    sigma = 0.06

    # Define 3-layer stack centered at L/2
    layers = [
        {'d': 0.08, 'eps_r': 4.0, 'sigma': 0.0},
        {'d': 0.04, 'eps_r': 1.0, 'sigma': 0.5},
        {'d': 0.08, 'eps_r': 4.0, 'sigma': 0.0},
    ]
    total_d = sum(lay['d'] for lay in layers)
    panel_center = L / 2
    panel_left = panel_center - total_d / 2
    panel_right = panel_center + total_d / 2
 
    obs_left_idx = np.argmin(np.abs(x - (panel_left - 0.4)))
    obs_right_idx = np.argmin(np.abs(x - (panel_right + 0.4)))

    initial_e = gaussian(x, pulse_x0, sigma)
    initial_h = gaussian(xH, pulse_x0, sigma)

    t_final = 2.5 * L

    # Build spatially-varying eps_r and sigma for the multilayer
    eps_r_arr = np.ones_like(x)
    sig_arr = np.zeros_like(x)
    edge = panel_left
    for lay in layers:
        mask = (x  >= edge)  & (x  < edge + lay['d'])
        eps_r_arr[mask] = lay['eps_r']
        sig_arr[mask] = lay['sigma']
        edge += lay['d']

    # Simulation WITH multilayer panel
    fdtd = FDTD1D(x, boundaries)
    fdtd.load_initial_field(initial_e)
    fdtd.h = initial_h.copy()
    fdtd.eps_r = eps_r_arr
    fdtd.sig = sig_arr

    n_steps = round(t_final / fdtd.dt)
    E_left_panel = np.zeros(n_steps)
    E_right_panel =  np.zeros(n_steps)
    for i in range(n_steps):
        fdtd._step()
        E_left_panel[i] = fdtd.e[obs_left_idx]
        E_right_panel[i] = fdtd.e[obs_right_idx]

    # Reference simulation WITHOUT  panel (free space)
    fdtd_ref = FDTD1D(x, boundaries)
    fdtd_ref.load_initial_field(initial_e)
    fdtd_ref.h = initial_h.copy()

    E_left_ref = np.zeros(n_steps)
    E_right_ref = np.zeros(n_steps)
    for i in range(n_steps):
        fdtd_ref._step()
        E_left_ref[i] = fdtd_ref.e[obs_left_idx]
        E_right_ref[i] = fdtd_ref.e[obs_right_idx]

    # Extract R(f), T(f) via FFT
    dt = fdtd.dt
    E_ref_fft = np.fft.rfft(E_left_panel - E_left_ref)
    E_trans_fft = np.fft.rfft(E_right_panel)
    E_inc_fft = np.fft.rfft(E_right_ref)
    freq = np.fft.rfftfreq(n_steps, d=dt)

    valid  = np.abs(E_inc_fft)  > 1e-10 * np.max(np.abs(E_inc_fft))
    R_fdtd = np.zeros_like(freq, dtype=complex)
    T_fdtd = np.zeros_like(freq, dtype=complex)
    R_fdtd[valid] = E_ref_fft[valid] / E_inc_fft[valid]
    T_fdtd[valid] = E_trans_fft[valid] / E_inc_fft[valid]

    # Analytical R(f), T(f) via stack transfer matrix
    f_bw = 1.0 / (2.0 * np.pi * sigma)
    band = (freq  > 0.1)  & (freq  < 1.5 * f_bw)

    Phi = stack_transfer_matrix(freq[band], layers)
    R_anal, T_anal = RT_from_transfer_matrix(Phi)

    assert np.corrcoef(np.abs(R_fdtd[band]), np.abs(R_anal))[0, 1]  > 0.90
    assert np.corrcoef(np.abs(T_fdtd[band]), np.abs(T_anal))[0, 1]  > 0.90

def test_fdtd_probes_record_correctly():
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)

    fdtd = FDTD1D(x, boundaries=('mur', 'mur'))
    fdtd.load_initial_field(initial_e)

    probe_x = 0.3
    fdtd.add_probe(probe_x, record_e=True, record_h=True)

    t_final = 0.2
    fdtd.run_until(t_final)

    data = fdtd.get_probe_data()
    assert len(data) == 1
    p = data[0]

    n_steps = round(t_final / fdtd.dt)
    assert len(p['e']) == n_steps
    assert len(p['h']) == n_steps

    idx_e = np.argmin(np.abs(x - probe_x))
    idx_h = np.argmin(np.abs((x[1:] + x[:-1]) / 2.0 - probe_x))
    assert np.isclose(p['e'][-1], fdtd.e[idx_e])
    assert np.isclose(p['h'][-1], fdtd.h[idx_h])

if __name__ == "__main__":
    pytest.main([__file__])