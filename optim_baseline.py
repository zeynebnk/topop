import scipy.sparse.linalg as spla
import scipy.sparse.linalg.dsolve as dsolve
import pyvista as pv
import numpy as np
import os
import torch
import argparse

dsolve.use_solver(useUmfpack=False, assumeSortedIndices=True)

from dl4to.criteria import Compliance, VolumeConstraint
from dl4to.topo_solvers import SIMP
from dl4to.pde import FDM
from dl4to.problem import Problem
from dl4to.datasets import SELTODataset

from cadutils import *
# from custom_pen import PEN

def simp_solve(problem, criterion, iters=3, lr=5e-1):
    """Solve the topology optimization problem using SIMP method."""
    simp = SIMP(
        criterion=criterion,
        binarizer_steepening_factor=1.02,
        n_iterations=iters,
        lr=lr,
    )

    solution = simp(problem)
    return solution

def main(args):
    density = load_density(args.base_mesh)
    material_props = eval(args.material_props)
    material_props = {
        'E': float(material_props['E']),
        'nu': float(material_props['nu']),
        'sig_ys': float(material_props['sig_ys'])
    }
    
    problem = construct_custom_problem(density, material_props)
    pde = FDM(padding_depth=0)
    problem.pde_solver = pde
    

    criterion = Compliance() + VolumeConstraint(max_volume_fraction=0.12, threshold_fct='relu')

    solution = simp_solve(problem, criterion)
    sol_density = solution.θ.cpu().detach().numpy()
    viz([problem, solution])

    density2stl(sol_density, args.out_name, spacing=problem.h, threshold=0.5, smooth_iters=100)

'''

def pen_solve(problem, criterion, iters=10, lr=5e-1):
    pen = PEN(
        criterion=criterion,
        binarizer_steepening_factor=1.02,
        n_iterations=iters,
        lr=lr,
    )

  solution = pen(problem)
  return solution
  
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_mesh", type=str, default="sphere_simple")
    parser.add_argument("--material_props", type=str, default="{'E': 7e10, 'nu': 0.3, 'sig_ys': 4.5e8}")
    parser.add_argument("--out_name", type=str, default="output.stl")
    parser.add_argument("--viz", type=bool, default=True)
    args = parser.parse_args() 

    main(args) 
