import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMatElastoPlast
import GooseFEM
import numpy as np
import os

# include functions from srcTopDown package
from srcTopDown.helper_functions.gmsh.parser_3D_interface  import parse_msh

# mesh
# ----
mesh_file_name = "i.geo.msh2"
curr_dir = os.path.dirname(os.path.abspath(__file__))
mesh_file_path = os.path.join(curr_dir, mesh_file_name)

if os.path.exists(mesh_file_path):
    print(f"Parsing mesh: {mesh_file_path}")
    mesh = parse_msh( 
        msh_filepath=mesh_file_path
    )
else:
    print(f"Error: Mesh file not found at {mesh_file_path}")

# mesh definition, displacement, external forces
coor = mesh["coor"]
connBulk = mesh["conn_bulk"]
connCohPart = mesh["conn_CohesiveParticle"]
connCohInt = mesh["conn_CohesiveInterface"]

# mesh dimensions
dofs = mesh["dofs"]
nelemBulk = len(connBulk)
nelemCohPart = len(connCohPart)
nelemCohInt = len(connCohInt)
ndim = 3

disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        dofs[mesh["UpperRightSurface"], 2],
        dofs[mesh["LowerRightSurface"], 2],
        dofs[mesh["UpperLeftSurface"], 0],
        dofs[mesh["UpperLeftSurface"], 1],
        dofs[mesh["UpperLeftSurface"], 2],
        dofs[mesh["LowerLeftSurface"], 0],
        dofs[mesh["LowerLeftSurface"], 1],
        dofs[mesh["LowerLeftSurface"], 2]        
    )
)

# simulation variables
# --------------------
# vector definition
vector = GooseFEM.VectorPartitioned(dofs, iip)

# allocate system matrix
K = GooseFEM.MatrixPartitioned(dofs, iip)
Solver = GooseFEM.MatrixPartitionedSolver()

# element definition
elemBulk = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, connBulk))
elemBulk0 = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, connBulk))
elemCohPart = GooseFEM.Element.Cohesive8.Quadrature(vector.AsElement(coor, connCohPart))
elemCohInt = GooseFEM.Element.Cohesive8.Quadrature(vector.AsElement(coor, connCohInt))

nipBulk = elemBulk.nip
nipCohPart = elemCohPart.nip
nipCohInt = elemCohInt.nip

# material definition
# -------------------

# mat = GMat.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))
matBulk = GMatElastoPlast.LinearHardening2d(
    K=np.ones([nelemBulk, nipBulk])*170,
    G=np.ones([nelemBulk, nipBulk])*80,
    tauy0=np.ones([nelemBulk, nipBulk])*10.0,
    H=np.ones([nelemBulk, nipBulk])*1.0,
)

# Cohesive zone material initialization for bulk Interface
matCohInt = GooseFEM.ConstitutiveModels.CohesiveBilinear3d(
    Kn=np.ones([nelemCohInt, nipCohInt])*30.0,
    Kt=np.ones([nelemCohInt, nipCohInt])*30.0,
    delta0=np.ones([nelemCohInt, nipCohInt])*0.02,
    deltafrac=np.ones([nelemCohInt, nipCohInt])*0.4,
    beta=np.ones([nelemCohInt, nipCohInt])*1.0,
    eta = np.ones([nelemCohInt, nipCohInt])*5e-03
    )    

# # Cohesive zone material initialization for Particle Interface
# matCohPart = GooseFEM.ConstitutiveModels.CohesiveBilinear3d(
#     Kn=np.ones([nelemCohPart, nipCohPart])*30.0,
#     Kt=np.ones([nelemCohPart, nipCohPart])*30.0,
#     delta0=np.ones([nelemCohPart, nipCohPart])*0.02,
#     deltafrac=np.ones([nelemCohPart, nipCohPart])*0.2,
#     beta=np.ones([nelemCohPart, nipCohPart])*1.0,
#     eta = np.ones([nelemCohPart, nipCohPart])*5e-03
#     )      

# solve
# -----
I2 = GMatTensor.Cartesian3d.Array2d(matBulk.shape).I2 
# strain
ueBulk = vector.AsElement(disp, connBulk)
# ueCohPart = vector.AsElement(disp, connCohPart)
ueCohInt = vector.AsElement(disp, connCohInt)
cooreBulk = vector.AsElement(coor, connBulk)
# cooreCohPart = vector.AsElement(coor, connCohPart)
cooreCohInt = vector.AsElement(coor, connCohInt)
elemBulk0.gradN_vector(ueBulk, matBulk.F)
matBulk.F += I2
matBulk.refresh()

# internal force of the right hand side per element and assembly
feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
# feCohPart = elemCohPart.Int_N_dot_traction_dL(matCohPart.T)
feCohInt = elemCohInt.Int_N_dot_traction_dA(matCohInt.T)
fint = vector.AssembleNode(feBulk, connBulk)
# fint += vector.AssembleNode(feCohPart, connCohPart)
fint += vector.AssembleNode(feCohInt, connCohInt)

# initial element tangential stiffness matrix incorporating the geometrical and material stiffness matrix
KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)
# KeCohPart = elemCohPart.Int_BT_D_B_dL(matCohPart.C)
KeCohInt = elemCohInt.Int_BT_D_B_dA(matCohInt.C)

K.clear()
K.assemble(KeBulk, connBulk)
# K.assemble(KeCohPart, connCohPart)
K.assemble(KeCohInt, connCohInt)
K.finalize()

# residual
fres = fext - fint

# solve
# -----
ninc = 2001
max_iter = 200
tangent = True 

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

du = np.zeros_like(disp)

total_time = 1.0 # pseudo time
dt = total_time / ninc # pseudo time increment

residual_history = []

initial_guess = np.zeros_like(disp)

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    if ilam % 50 == 0:
        print(matCohInt.Damage[:8])
        # print(matCohPart.Damage[:8])

    disp[mesh["UpperRightSurface"], 2] += (+0.5/ninc)
    disp[mesh["LowerRightSurface"], 2] += (-0.5/ninc)
    disp[mesh["UpperLeftSurface"], 0] = 0.0  
    disp[mesh["UpperLeftSurface"], 1] = 0.0  
    disp[mesh["UpperLeftSurface"], 2] = 0.0 
    disp[mesh["LowerLeftSurface"], 0] = 0.0  
    disp[mesh["LowerLeftSurface"], 1] = 0.0  
    disp[mesh["LowerLeftSurface"], 2] = 0.0 

    # convergence flag
    converged = False

    matBulk.increment()
    # matCohPart.increment()
    matCohInt.increment()

    # impose initial guess on unknown displacements
    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))

    total_increment = initial_guess.copy()
    for iter in range(max_iter): 
        # update element wise displacments
        ueBulk = vector.AsElement(disp, connBulk) 
        # ueCohPart = vector.AsElement(disp, connCohPart)
        ueCohInt = vector.AsElement(disp, connCohInt)

        # update deformation gradient F
        elemBulk.symGradN_vector(ueBulk, matBulk.F)
        matBulk.F += I2
        matBulk.refresh()

        # update nodal displacements of cohesive zone
        # elemCohPart.relative_disp(ueCohPart, matCohPart.delta, matCohPart.ori)
        elemCohInt.relative_disp(ueCohInt, matCohInt.delta, matCohInt.ori)        
        # matCohPart.refresh(dt)
        matCohInt.refresh(dt)
  
        # update internal forces and assemble
        feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)

        # update tractions of cohesive zone
        # feCohPart = elemCohPart.Int_N_dot_traction_dL(matCohPart.T)
        feCohInt = elemCohInt.Int_N_dot_traction_dA(matCohInt.T)

        fint = vector.AssembleNode(feBulk, connBulk)
        # fint += vector.AssembleNode(feCohPart, connCohPart)
        fint += vector.AssembleNode(feCohInt, connCohInt)

        # update stiffness matrix
        KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)        
        # KeCohPart = elemCohPart.Int_BT_D_B_dL(matCohPart.C)
        KeCohInt = elemCohInt.Int_BT_D_B_dA(matCohInt.C)

        K.clear()
        K.assemble(KeBulk, connBulk)
        # K.assemble(KeCohPart, connCohPart)
        K.assemble(KeCohInt, connCohInt)
        K.finalize()

        fres = -fint

        if iter > 0:
            # residual 
            fres_u = -vector.AsDofs_u(fint)
            
            res_norm = np.linalg.norm(fres_u) 

            residual_history.append(res_norm)

            if res_norm < 1e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
                converged = True
                break


        # solve
        Solver.solve(K, fres, du)

        # add newly found delta_u to total increment
        total_increment += du

        # update displacement vector
        disp += du

        # update shape functions
        # elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
        # lemCohPart.update_x(vector.AsElement(coor + disp, connCohPart))
        elemCohInt.update_x(vector.AsElement(coor + disp, connCohInt))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1)))
    sigeq[ilam] = np.average(GMatElastoPlast.Sigeq(np.average(matBulk.Sig, axis=1)))
    
    if converged:
        initial_guess = total_increment
    if not converged:
        print(f"WARNING: not converge at displacement {0.4*ilam/ninc} but no Runtime termination.")
        break
        # raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# ------------
# strain
print(disp)
# print(matCohPart.Damage)


elemBulk0.symGradN_vector(ueBulk, matBulk.F)
# elemCohPart.relative_disp(ueCohPart, matCohPart.delta, matCohPart.ori)
elemCohInt.relative_disp(ueCohInt, matCohInt.delta, matCohInt.ori)
matBulk.refresh()  
# matCohPart.refresh(dt)
matCohInt.refresh(dt)


feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
# feCohPart = elemCohPart.Int_N_dot_traction_dL(matCohPart.T)
feCohInt = elemCohInt.Int_N_dot_traction_dA(matCohInt.T)
# internal force
fint = vector.AssembleNode(feBulk, connBulk)
# fint += vector.AssembleNode(feCohPart, connCohPart)
fint += vector.AssembleNode(feCohInt, connCohInt)

# apply reaction force
fres_u = -vector.AsDofs_u(fint)
            
res_norm = np.linalg.norm(fres_u)

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
    dV = elemBulk.AsTensor(2, elemBulk.dV)
    Sigav = np.average(matBulk.Sig, weights=dV, axis=1)
    sigeq_av = GMatElastoPlast.Sigeq(Sigav)
    epseq_av = GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    # if all values are equal, extend range a bit:
    vmax = sigeq_av.max()
    vmin = 0

    norm = Normalize(vmin=vmin, vmax=vmax)
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
    for i, element in enumerate(connBulk):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    # for element in connBulk:
    #     verts = np.array(coor[element]) 
    #     for face in faces:
    #         face_verts = verts[face].tolist()
    #         poly = Poly3DCollection([face_verts],
    #                                 facecolors=[[0, 0, 0, 0]],  # Fully transparent
    #                                 edgecolors='k',
    #                                 linewidths=0.5,
    #                                 linestyles='dashed')
    #         ax.add_collection3d(poly)

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
    # if all values are equal, extend range a bit:
    vmax = epseq_av.max()
    vmin = 0

    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(epseq_av))
    
    faces = [
    [0, 1, 2, 3],  # bottom face
    [4, 5, 6, 7],  # top face
    [0, 1, 5, 4],  # front face
    [1, 2, 6, 5],  # right face
    [2, 3, 7, 6],  # back face
    [3, 0, 4, 7]   # left face
    ]

    # Plot deformed mesh with colors
    for i, element in enumerate(connBulk):
        verts = np.array(coor[element] + disp[element]) 
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts], facecolors=[colors[i]], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(poly)

    # Plot undeformed mesh with dashed edges and transparent faces
    for element in connBulk:
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
