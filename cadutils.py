import scipy.sparse.linalg as spla
import scipy.sparse.linalg.dsolve as dsolve
import pyvista as pv
import numpy as np
import os
import torch

from dl4to.problem import Problem
from dl4to.datasets import SELTODataset

dsolve.use_solver(useUmfpack=False)

def density2stl(density, out_name, spacing=(1,1,1), threshold=0.5, smooth_iters=100):
    if isinstance(density, torch.Tensor):
        density = density.detach().cpu().numpy()

    
    if len(density.shape) == 4:
        density = density[0, 0]
    elif len(density.shape) == 3:  
        density = density[0]

    # uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = np.array(density.shape) + 1
    grid.spacing = spacing
    grid.cell_data["scalar_field"] = density.flatten(order="F")
    
    # get surface
    surf = grid.threshold(threshold, scalars='scalar_field').extract_surface()
    
    # smooth for conversion
    if smooth_iters > 0:
        surf = surf.smooth_taubin(n_iter=smooth_iters)
    
    # save as stl
    surf.save(out_name)
    print(f"saved {out_name}")


def stl2density(stl_path, grid_size=(64, 64, 64), spacing=(1, 1, 1)):
    ## TODO
    pass


def load_density(name='disc_complex', idx=0):
    if name in ['disc_complex', 'disc_simple', ]:
        dataset = SELTODataset("../dl4to_dataset", name=name)
        problem, _ = dataset[idx]
        return problem.Ω_design
    else:
        density = stl2density(name)
        return density
        
def load_force_and_dirichlet(shape, force_location='top', force_direction=(0, 0, -1), force_magnitude=4e7, fixed_location='bottom', Ω_design=None):
    F = torch.zeros(3, *shape)
    Ω_dirichlet = torch.zeros(3, *shape)

    locations = {
        'top': (slice(None), slice(None), -1),
        'bottom': (slice(None), slice(None), 0),
        'left': (0, slice(None), slice(None)),
        'right': (-1, slice(None), slice(None)),
        'front': (slice(None), 0, slice(None)),
        'back': (slice(None), -1, slice(None))
    }
    
    # design mask
    if Ω_design is not None:
        design_mask = (Ω_design[0] != 0) 
    else:
        design_mask = torch.ones(shape, dtype=torch.bool)
    
    if force_location in locations:
        idx = locations[force_location]
        force = torch.tensor(force_direction, dtype=torch.float) * force_magnitude
        
        force_mask = design_mask[idx]
        F[0, idx[0], idx[1], idx[2]] = force[0] * force_mask
        F[1, idx[0], idx[1], idx[2]] = force[1] * force_mask
        F[2, idx[0], idx[1], idx[2]] = force[2] * force_mask

    # boundary conditions
    if fixed_location in locations:
        idx = locations[fixed_location]
        fixed_mask = design_mask[idx]
        Ω_dirichlet[0, idx[0], idx[1], idx[2]] = fixed_mask
        Ω_dirichlet[1, idx[0], idx[1], idx[2]] = fixed_mask
        Ω_dirichlet[2, idx[0], idx[1], idx[2]] = fixed_mask
    
    return F, Ω_dirichlet


def construct_custom_problem(density, material_props, spacing=(1.0, 1.0, 1.0)):
    
    shape = density.shape[1:]  
    Ω_design = density.clone()  

    # forces
    F, Ω_dirichlet = load_force_and_dirichlet(shape, Ω_design=Ω_design)
    
    # material properties
    E = material_props['E']
    nu = material_props['nu']
    sig_ys = material_props['sig_ys']

    h = torch.tensor(spacing, dtype=torch.float)

    # create problem 
    problem = Problem(
        E=E,
        ν=nu,
        σ_ys=sig_ys,
        h=h,
        Ω_dirichlet=Ω_dirichlet,
        Ω_design=Ω_design,
        F=F,
        restrict_density_for_voxels_with_applied_forces=True
    )
    
    return problem

def viz(to_viz, display=True):
  camera_position = (0, 0.06, 0.12)
  for item in to_viz:
    item.plot(camera_position=camera_position,
              use_pyvista=True,
              smooth_iters=100,
              window_size=(600,600),
              display=display)

def main():

    density = load_density('disc_complex')
    #density = load_density('cad_files/disc_complex/disc_complex_initial_0.stl')

    material_props = {
        'E': 7e10,
        'nu': 0.3,
        'sig_ys': 4.5e8
    }

    problem = construct_custom_problem(density, material_props)
    
    viz([problem])

if __name__ == "__main__":
    main()
    
