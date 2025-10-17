import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
import os
from srcTopDown.helper_functions.element_erosion import element_erosion_3D_PBC
from srcTopDown.helper_functions.nodes_periodic import nodesPeriodic3D
from srcTopDown.helper_functions.gmsh.parser_3D import parse_msh
from srcTopDown.helper_functions.newton_raphson import newton_raphson_solve
from srcTopDown.helper_functions.element_erosion import element_erosion_multiplemat
import srcTopDown.plot_functions.RVE_plot_3d as pf
# mesh
# ----

# define mesh
print("running a GooseFEM static example...")
mesh_file_name = "i.geo_composite_cube.msh2"
json_file_name = "i.geo_element_sets.json"
curr_dir = os.path.dirname(os.path.abspath(__file__))
mesh_file_path = os.path.join(curr_dir, mesh_file_name)
json_file_path = os.path.join(curr_dir, json_file_name)

if os.path.exists(mesh_file_path) and os.path.exists(json_file_path):
    print(f"Parsing mesh: {mesh_file_path}")
    mesh = parse_msh( 
        msh_filepath=mesh_file_path, json_filepath=json_file_path
    )
else:
    print(f"Error: Mesh file not found at {mesh_file_path}")

nr_materials = 3
ndim = 3

conn = np.empty(nr_materials, dtype=object)
nelem = np.empty(nr_materials, dtype=object)
elem = np.empty(nr_materials, dtype=object)
elem0 = np.empty(nr_materials, dtype=object)
mat = np.empty(nr_materials, dtype=object)
fe = np.empty(nr_materials, dtype=object)
ue = np.empty(nr_materials, dtype=object)
coore = np.empty(nr_materials, dtype=object)
Ke = np.empty(nr_materials, dtype=object)
I2 = np.empty(nr_materials, dtype=object)
damage_prev = np.empty(nr_materials, dtype=object)

# mesh definition, displacement, external forces
coor = mesh["coor"]
conn[0] = mesh["conn_matrix"]
conn[1] = mesh["conn_interface"]
conn[2] = mesh["conn_particle"]
# connChz = np.vstack((connCohParticle, connCohInterface))

dofs = mesh["dofs"]
nelem[0] = len(conn[0])
nelem[1] = len(conn[1])
nelem[2] = len(conn[2])

# create control nodes
control = GooseFEM.Tyings.Control(coor, dofs)

# add control nodes
coor = control.coor

# list of prescribed DOFs (fixed node + control nodes)
iip = np.concatenate(
    (
    dofs[mesh['corner_froBotLft'], 0],
    dofs[mesh['corner_froBotLft'], 1],
    dofs[mesh['corner_froBotLft'], 2],    
    control.controlDofs[0],
    control.controlDofs[1],
    control.controlDofs[2]
    )
)

# ----- NODES PERIODIC?? -----
tyinglist = nodesPeriodic3D(mesh)
# ----- NODES PERIODIC?? -----


# initialize my periodic boundary condition class
periodicity = GooseFEM.Tyings.Periodic(coor, control.dofs, control.controlDofs, tyinglist, iip)
dofs = periodicity.dofs

# simulation variables
# --------------------

# vector definition
vector = GooseFEM.VectorPartitionedTyings(dofs, periodicity.Cdu, periodicity.Cdp, periodicity.Cdi)

# element definition
for i in range(nr_materials):
    elem[i] = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, conn[i]))
    elem0[i] = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, conn[i]))
nipBulk = elem[0].nip

# nodal quantities
disp = np.zeros_like(coor)
du = np.zeros_like(coor)  # iterative displacement update
fint = np.zeros_like(coor)  # internal force
fext = np.zeros_like(coor)  # external force


# DOF values
Fext = np.zeros([periodicity.nni])
Fint = np.zeros([periodicity.nni])

# material definition
# -------------------
# matrix material
mat[0] = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem[0], nipBulk])*170,
    G=np.ones([nelem[0], nipBulk])*80,
    tauy0=np.ones([nelem[0], nipBulk])*1.0,
    H=np.ones([nelem[0], nipBulk])*1.0,
    D1=np.ones([nelem[0], nipBulk])*0.1,
    D2=np.ones([nelem[0], nipBulk])*0.2,
    D3=np.ones([nelem[0], nipBulk])*-1.7    
    )

# interface material
mat[1] = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem[1], nipBulk])*70,
    G=np.ones([nelem[1], nipBulk])*30,
    tauy0=np.ones([nelem[1], nipBulk])*100.0,
    H=np.ones([nelem[1], nipBulk])*1.0,
    D1=np.ones([nelem[1], nipBulk])*0.1,
    D2=np.ones([nelem[1], nipBulk])*0.2,
    D3=np.ones([nelem[1], nipBulk])*-1.7    
    )

# particle material
mat[2] = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem[2], nipBulk])*250,
    G=np.ones([nelem[2], nipBulk])*110,
    tauy0=np.ones([nelem[2], nipBulk])*10.0,
    H=np.ones([nelem[2], nipBulk])*1.0,
    D1=np.ones([nelem[2], nipBulk])*0.1,
    D2=np.ones([nelem[2], nipBulk])*0.2,
    D3=np.ones([nelem[2], nipBulk])*-1.7    
    )
# allocate system matrix
K = GooseFEM.MatrixPartitionedTyings(dofs, periodicity.Cdu, periodicity.Cdp)
Solver = GooseFEM.MatrixPartitionedTyingsSolver()

K.clear()
for i in range(nr_materials):
    I2[i] = GMatTensor.Cartesian3d.Array2d(mat[i].shape).I2  
    ue[i] = vector.AsElement(disp, conn[i])
    coore[i] = vector.AsElement(coor, conn[i]) 
    elem[i].gradN_vector((ue[i]), mat[i].F)
    mat[i].F += I2[i]
    mat[i].refresh()
    fe[i] = elem[i].Int_gradN_dot_tensor2_dV(mat[i].Sig)
    Ke[i] = elem[i].Int_gradN_dot_tensor4_dot_gradNT_dV(mat[i].C)
    K.assemble(Ke[i], conn[i])
K.finalize()

# pre-process
# plot
# ----
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Plot result")
parser.add_argument("--save", action="store_true", help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])
if args.plot:    
    import matplotlib.pyplot as plt
    plt.style.use(["goose", "goose-latex"])
    pf.plot_materials(coor, conn, mat, args, labels=["Matrix", "Interface", "Particle"])


# internal force of the right hand side per element and assembly
fint = vector.AssembleNode(fe[0], conn[0])
for i in range(1,nr_materials):
    fint += vector.AssembleNode(fe[i], conn[i])

# initial residual
fres = fext - fint




# solve
# -----
ninc = 701
max_iter = 50
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

# ue = vector.AsElement(disp)
# du = np.zeros_like(disp)
initial_guess = np.zeros_like(disp)
total_increment = np.zeros_like(disp)
elem_failed_prev = np.empty(nr_materials, dtype=object)
to_be_deleted = np.empty(nr_materials, dtype=object)
damage_prev = np.empty(nr_materials, dtype=object)

for j in range(nr_materials):
    elem_failed_prev[j] = set()
    to_be_deleted[j] = []

# deformation gradient
F = np.array(
        [
            [1.0 + (0.1/ninc), 0.0, 0.0],
            [0.0, 1.0 / (1.0 + (0.1/ninc)), 0.0],
            [0.0, 0.0, 1.0]
        ]
    )

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    #damage_prev = mat.D_damage.copy()
    #disp, elem, mat = element_erosion_3D_PBC(Solver, vector, conn, mat, damage_prev, elem, elem0, fe,                                                                      
    #                                                                fext, disp, elem_to_delete, K, fe, I2, coor)
    converged = False

    for i in range(nr_materials):
        mat[i].increment()
        damage_prev[i] = mat[i].D_damage.copy()

    disp += initial_guess

    disp, elem, mat, converged, total_increment, res, iter = newton_raphson_solve(disp, initial_guess, max_iter, control, vector, conn, elem0, elem, mat,
                                                                       ue, fe, fint, Ke, K, fext, Fext, Fint, Solver, du, F, I2, coor, RES_TOL=1e-06)
    
    if converged:
        # accumulate strains and stresses
        epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat[0].F), axis=1)))
        sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat[0].Sig, axis=1)))
        print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res}")
        initial_guess = 0.3 * total_increment
        continue
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")

# post-processing
if args.plot:
    plot_data = pf.prepare_plot_data(elem, mat, conn, coor, disp)
    pf.plot_3d(plot_data, "stress", args)
    pf.plot_3d(plot_data, "strain", args)
    pf.plot_3d(plot_data, "damage", args)
    pf.plot_3d(plot_data, "triaxiality", args)