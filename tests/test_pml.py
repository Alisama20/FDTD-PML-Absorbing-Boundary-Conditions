import numpy as np
import pytest
from fdtd1d import FDTD1D, C, gaussian


def test_pml_absorbs_left_traveling_wave():
    x = np.linspace(-1.0, 1.0, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('PML', 'PML')

    x0 = 0.0
    sigma = 0.05
    n_pml = 15
    initial_e = gaussian(x, x0, sigma)
    initial_h = -gaussian(xH, x0, sigma)

    fdtd = FDTD1D(x, boundaries, n_pml=n_pml, pml_order=3, pml_R0=1e-6)
    fdtd.load_initial_field(initial_e)
    fdtd.h[fdtd.Npml_left:fdtd.Npml_left + len(initial_h)] = initial_h.copy()

    dx = x[1] - x[0]
    d_pml = n_pml * dx
    # time for wave to travel from center to nearest boundary + cross PML twice (safety)
    t_final = ((x[-1] - x[0]) / 2.0 + 2.0 * d_pml) / C
    fdtd.run_until(t_final)
    
    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()
    
    assert np.max(np.abs(e_solved)) < 0.05
    assert np.max(np.abs(h_solved)) < 0.05


def test_pml_absorbs_right_traveling_wave():
    x = np.linspace(-1.0, 1.0, 201)
    xH = (x[1:] + x[:-1]) / 2.0
    boundaries = ('PML', 'PML')

    x0 = 0.0
    sigma = 0.05
    n_pml = 15
    initial_e = gaussian(x, x0, sigma)
    initial_h = gaussian(xH, x0, sigma)

    fdtd = FDTD1D(x, boundaries, n_pml=n_pml, pml_order=3, pml_R0=1e-6)
    fdtd.load_initial_field(initial_e)
    fdtd.h[fdtd.Npml_left:fdtd.Npml_left + len(initial_h)] = initial_h.copy()

    dx = x[1] - x[0]
    d_pml = n_pml * dx
    # time for wave to travel from center to nearest boundary + cross PML twice (safety)
    t_final = ((x[-1] - x[0]) / 2.0 + 2.0 * d_pml) / C
    fdtd.run_until(t_final)
    
    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()
    
    assert np.max(np.abs(e_solved)) < 0.05
    assert np.max(np.abs(h_solved)) < 0.05




def test_pml_one_sided():
    x = np.linspace(-1.0, 1.0, 201)
    xH = (x[1:] + x[:-1]) / 2.0

    x0 = -0.5
    sigma = 0.05
    n_pml = 15
    initial_e = gaussian(x, x0, sigma)
    initial_h = -gaussian(xH, x0, sigma)

    fdtd = FDTD1D(x, ('PML', 'PEC'), n_pml=n_pml)
    fdtd.load_initial_field(initial_e)
    fdtd.h[fdtd.Npml_left:fdtd.Npml_left + len(initial_h)] = initial_h.copy()

    dx = x[1] - x[0]
    d_pml = n_pml * dx
    # time for wave to travel from x0 to left boundary + cross PML twice (safety)
    t_final = (abs(x0 - x[0]) + 2.0 * d_pml) / C
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()
    h_solved = fdtd.get_h()

    assert np.max(np.abs(e_solved)) < 0.05
    assert np.max(np.abs(h_solved)) < 0.05




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
