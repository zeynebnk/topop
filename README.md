# topop
topological optimization 

### inputs - problem space
material_props (scalar): 
E (Young's modulus)
ν (Poisson's ratio)
σ_ys (Yield stress)

base model (n_x, n_y, n_z)
dirichlet boundary conditions (1, $n_x$, $n_y$, $n_z$) -> 'locks' [default = base]
external forces (3, n_x, n_y, n_z) -> load forces (dir, mag) [default = faces/top]


### inputs - optim crit
e.g. compliance, volume, stress, etc 
supports custom (unsupervised physics-based)

### inputs - optim settings
hyperparams (default rn)

### TO
in -> CAD/.stl
TO type (baseline, ver, ...)
out -> CAD/.stl

