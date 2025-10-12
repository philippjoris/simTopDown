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
    tauy0=np.ones([nelem[0], nipBulk])*10.0,
    H=np.ones([nelem[0], nipBulk])*1.0,
    D1=np.ones([nelem[0], nipBulk])*0.1,
    D2=np.ones([nelem[0], nipBulk])*0.2,
    D3=np.ones([nelem[0], nipBulk])*-1.7    
    )

# interface material
mat[1] = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem[1], nipBulk])*170,
    G=np.ones([nelem[1], nipBulk])*80,
    tauy0=np.ones([nelem[1], nipBulk])*10.0,
    H=np.ones([nelem[1], nipBulk])*1.0,
    D1=np.ones([nelem[1], nipBulk])*0.1,
    D2=np.ones([nelem[1], nipBulk])*0.2,
    D3=np.ones([nelem[1], nipBulk])*-1.7    
    )

# particle material
mat[2] = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem[2], nipBulk])*170,
    G=np.ones([nelem[2], nipBulk])*80,
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

for j in range(nr_materials):
    elem_failed_prev[j] = set()
    to_be_deleted[j] = []

# deformation gradient
F = np.array(
        [
            [1.0 + (0.10/ninc), 0.0, 0.0],
            [0.0, 1.0 / (1.0 + (0.10/ninc)), 0.0],
            [0.0, 0.0, 1.0]
        ]
    )

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    if len(failed_elem) > 65:
        break
    #    damage_prev = mat.D_damage.copy()
    #    elem_to_delete = {100}
    #    disp, elem, mat = element_erosion_3D_PBC(Solver, vector, conn, mat, damage_prev, elem, elem0, fe,                                                                      
    #                                                                    fext, disp, elem_to_delete, K, fe, I2, coor)
    converged = False

    mat.increment()

    disp += initial_guess
    total_increment = initial_guess.copy()
    damage_prev = mat.D_damage.copy()
    for iter in range(max_iter):  
        # deformation gradient
        K.clear()
        for i in range(nr_materials):
            ue[i] = vector.AsElement(disp, conn[i]) 
            elem0[i].symGradN_vector(ue[i], mat[i].F)
            mat[i].F += I2[i]
            mat[i].refresh()
            fe[i] = elem[i].Int_gradN_dot_tensor2_dV(mat[i].Sig)
            Ke[i] = elem[i].Int_gradN_dot_tensor4_dot_gradNT_dV(mat[i].C) 
            K.assemble(Ke[i], conn[i]) 
        K.finalize(stabilize=True)

        fint = vector.AssembleNode(fe[0], conn[0])
        for i in range(1,nr_materials):
            fint += vector.AssembleNode(fe[i], conn[i])

        fres = -fint

        if iter > 0:
            # - internal/external force as DOFs (account for periodicity)
            vector.asDofs_i(fext, Fext)
            vector.asDofs_i(fint, Fint)
            # - extract reaction force
            vector.copy_p(Fint, Fext)
            # - norm of the residual and the reaction force
            nfres = np.sum(np.abs(Fext - Fint))
            nfext = np.sum(np.abs(Fext))
            # - relative residual, for convergence check
            if nfext:
                res = nfres / nfext
            else:
                res = nfres

            if res < 1e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res}")
                converged = True
                break

        du.fill(0.0)

        # initialise displacement update
        if iter == 0:
            du[control.controlNodes, 0] = (F[0,:] - np.eye(3)[0, :]) 
            du[control.controlNodes, 1] = (F[1,:] - np.eye(3)[1, :])  
            du[control.controlNodes, 2] = (F[2,:] - np.eye(3)[2, :])              

        # solve
        Solver.solve(K, fres, du)
        
        # add delta u
        disp += du
        total_increment += du

        for i in range(nr_materials):
            elem[i].update_x(vector.AsElement(coor + disp, conn[i]))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat[0].F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat[0].Sig, axis=1)))
    
    if converged:
         # print(total_increment)
         initial_guess = 0.3 * total_increment
         continue
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# plot
# ----
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Plot result")
parser.add_argument("--save", action="store_true", help="Save plot (plot not shown)")
args = parser.parse_args(sys.argv[1:])

if args.plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter

    plt.style.use(["goose", "goose-latex"])

    # Average equivalent stress per element
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq_av = GMat.Sigeq(Sigav)
    epseq_av = GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1))
    damage_av = np.average(mat.D_damage, axis=1)
    sig_triax_av = np.average(mat.Sig_triax, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    norm = plt.Normalize(vmin=sigeq_av.min(), vmax=sigeq_av.max())
    colors = cmap(norm(sigeq_av))

    faces = [
    [0, 1, 2, 3],  # bottom face
    [4, 5, 6, 7],  # top face
    [0, 1, 5, 4],  # front face
    [1, 2, 6, 5],  # right face
    [2, 3, 7, 6],  # back face
    [3, 0, 4, 7]   # left face
    ]

    # Plot deformed mesh with colors
    for i, element in enumerate(conn):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in conn:
        verts = np.array(coor[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[[0, 0, 0, 0]],  # Fully transparent
                                    edgecolors='k',
                                    linewidths=0.5,
                                    linestyles='dashed')
            ax.add_collection3d(poly)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(sigeq_av)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent stress")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save:
        fig.savefig('fixed-disp_contour_sig.pdf')
    else:
        plt.show()

    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    norm = plt.Normalize(vmin=epseq_av.min(), vmax=epseq_av.max())
    colors = cmap(norm(epseq_av))
    

    # Plot deformed mesh with colors
    for i, element in enumerate(conn):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in conn:
        verts = np.array(coor[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[[0, 0, 0, 0]],  # Fully transparent
                                    edgecolors='k',
                                    linewidths=0.5,
                                    linestyles='dashed')
            ax.add_collection3d(poly)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(epseq_av)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent strain")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save:
        fig.savefig('fixed-disp_contour_eps.pdf')
    else:
        plt.show()
    plt.close(fig)

    # ------------ damage plot -------------        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    norm = plt.Normalize(vmin=damage_av.min(), vmax=damage_av.max())
    colors = cmap(norm(damage_av))
    

    # Plot deformed mesh with colors
    for i, element in enumerate(conn):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in conn:
        verts = np.array(coor[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[[0, 0, 0, 0]],  # Fully transparent
                                    edgecolors='k',
                                    linewidths=0.5,
                                    linestyles='dashed')
            ax.add_collection3d(poly)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(damage_av)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Damage")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save:
        fig.savefig('fixed-disp_contour_damage.pdf')
    else:
        plt.show()
    plt.close(fig)    

# ------------ stress triaxiality plot -------------        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    norm = plt.Normalize(vmin=sig_triax_av.min(), vmax=sig_triax_av.max())
    colors = cmap(norm(sig_triax_av))
    

    # Plot deformed mesh with colors
    for i, element in enumerate(conn):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in conn:
        verts = np.array(coor[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[[0, 0, 0, 0]],  # Fully transparent
                                    edgecolors='k',
                                    linewidths=0.5,
                                    linestyles='dashed')
            ax.add_collection3d(poly)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(sig_triax_av)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="stress traixility")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])  # Ensure scaling works properly

    # Optional save or show
    if args.save:
        fig.savefig('fixed-disp_contour_triaxiality.pdf')
    else:
        plt.show()
    plt.close(fig)     