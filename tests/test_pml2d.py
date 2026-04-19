import numpy as np
import pytest
from fdtd2d import FDTD2D, gaussian2d, C


def test_fdtd2d_basic_propagation():
    x = np.linspace(-1.0, 1.0, 101)
    y = np.linspace(-1.0, 1.0, 101)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    initial_Ez = gaussian2d(X, Y, 0.0, 0.0, 0.1)
    
    fdtd = FDTD2D(x, y, boundaries=('PEC', 'PEC', 'PEC', 'PEC'))
    fdtd.load_initial_field(initial_Ez)

    # run for 10% of the time it takes a wave to cross the full domain diagonally
    t_final = 0.1 * np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2) / C
    fdtd.run_until(t_final)
    
    Ez = fdtd.get_Ez()
    
    assert Ez.shape == (len(x), len(y))
    assert np.max(np.abs(Ez)) < 1.0


def test_pml2d_absorbs_wave():
    x = np.linspace(-1.0, 1.0, 101)
    y = np.linspace(-1.0, 1.0, 101)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    initial_Ez = gaussian2d(X, Y, 0.0, 0.0, 0.1)
    
    n_pml = 15
    fdtd = FDTD2D(x, y, boundaries=('PML', 'PML', 'PML', 'PML'),
                  n_pml=n_pml, pml_order=3, pml_R0=1e-6)
    fdtd.load_initial_field(initial_Ez)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    d_pml = n_pml * max(dx, dy)
    # diagonal distance from center to corner + cross PML twice (safety)
    half_diagonal = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2) / 2.0
    t_final = (half_diagonal + 2.0 * d_pml) / C
    fdtd.run_until(t_final)
    
    Ez = fdtd.get_Ez()

    max_field = np.max(np.abs(Ez))
    assert max_field < 0.05



def test_pml2d_sigma_profile():
    x = np.linspace(-1.0, 1.0, 51)
    y = np.linspace(-1.0, 1.0, 51)
    n_pml = 5
    
    fdtd = FDTD2D(x, y, boundaries=('PML', 'PML', 'PML', 'PML'), n_pml=n_pml)
    
    physical_sigma = fdtd.sig[n_pml:-n_pml, n_pml:-n_pml]
    assert np.allclose(physical_sigma, 0.0)

    left_pml = fdtd.sig[:n_pml, n_pml:-n_pml]
    right_pml = fdtd.sig[-n_pml:, n_pml:-n_pml]
    bottom_pml = fdtd.sig[n_pml:-n_pml, :n_pml]
    top_pml = fdtd.sig[n_pml:-n_pml, -n_pml:]
    
    assert np.all(left_pml > 0)
    assert np.all(right_pml > 0)
    assert np.all(bottom_pml > 0)
    assert np.all(top_pml > 0)


def test_pml2d_corners_have_combined_sigma():
    x = np.linspace(-1.0, 1.0, 51)
    y = np.linspace(-1.0, 1.0, 51)
    n_pml = 5
    
    fdtd = FDTD2D(x, y, boundaries=('PML', 'PML', 'PML', 'PML'), n_pml=n_pml)
    
    pml_mid = n_pml // 2  # midpoint depth inside the PML

    # corner: same PML depth in both x and y → sx + sy
    corner_sigma      = fdtd.sig[pml_mid, pml_mid]
    # left strip center: same PML depth in x, center of physical domain in y → only sx
    left_edge_sigma   = fdtd.sig[pml_mid, n_pml + len(y) // 2]
    # bottom strip center: center of physical domain in x, same PML depth in y → only sy
    bottom_edge_sigma = fdtd.sig[n_pml + len(x) // 2, pml_mid]
    
    assert corner_sigma > left_edge_sigma
    assert corner_sigma > bottom_edge_sigma




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
