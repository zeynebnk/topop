import scipy.sparse.linalg as spla
import scipy.sparse.linalg.dsolve as dsolve
import pyvista as pv
import numpy as np
import os
import torch

# Initialize the solver properly
dsolve.use_solver(useUmfpack=False)

from dl4to.datasets import SELTODataset

def save_as_stl(density, filename, spacing=(1,1,1), threshold=0.5, smooth_iters=100):
    """Save a density field as an STL file."""
    # Remove the first dimension (batch dimension) if it exists
    if len(density.shape) == 4:
        density = density[0]  # Take the first batch element
    
    # Create a uniform grid
    grid = pv.UniformGrid()
    grid.dimensions = np.array(density.shape) + 1
    grid.spacing = spacing
    grid.cell_data["scalar_field"] = density.flatten(order="F")
    
    # Extract surface
    surf = grid.threshold(threshold, scalars='scalar_field').extract_surface()
    
    # Smooth the surface
    if smooth_iters > 0:
        surf = surf.smooth_taubin(n_iter=smooth_iters)
    
    # Save as STL
    surf.save(filename)
    print(f"Saved {filename}")

def process_dataset(dataset_name, output_dir, n_samples=1):
    """Process a dataset and save initial forms and ground truth solutions."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nProcessing {dataset_name} problems...")
    dataset = SELTODataset("../dl4to_dataset", name=dataset_name)
    
    # Process each sample
    for i in range(min(n_samples, len(dataset))):
        try:
            # Load problem and ground truth
            problem, gt_solution = dataset[i]
            
            # Save design space (Ω_design)
            design_space = (problem.Ω_design.squeeze() != 0).cpu().detach().numpy() #problem.Ω_design.cpu().detach().numpy()
            
            
            # Save initial design space with adjusted threshold
            save_as_stl(
                design_space,
                os.path.join(output_dir, f'{dataset_name}_initial_{i}.stl'),
                spacing=problem.h,
                threshold=0.99  # Higher threshold to capture only the solid regions
            )
            
            # Save ground truth solution
            gt_density = gt_solution.get_θ(binary=True).cpu().detach().numpy()
            save_as_stl(
                gt_density,
                os.path.join(output_dir, f'{dataset_name}_ground_truth_{i}.stl'),
                spacing=problem.h,
                threshold=0.5
            )
            
        except Exception as e:
            print(f"Error processing sample {i} from {dataset_name}: {str(e)}")

def main():
    # List of datasets to process
    datasets = [
        'disc_simple',
        'disc_complex',
        'sphere_simple'
    ]
    
    # Process each dataset
    for dataset_name in datasets:
        output_dir = os.path.join('cad_files', dataset_name)
        process_dataset(dataset_name, output_dir, n_samples=10)

if __name__ == "__main__":
    main() 
