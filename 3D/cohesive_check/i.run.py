import argparse
import sys

import GMatTensor.Cartesian2d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMatElastoPlast
import GooseFEM
import numpy as np
import random
#
# Example with 2D varying microstructure and fixed disp
#
# mesh
# ----

# define mesh
print("running a GooseFEM static example...")
mesh = GooseFEM.MeshCohesive.Hex8.RegularCohesive(1, 1, 1, 1, 1.0)

# mesh dimensions
# nelem = mesh.nelem
nneBulk = mesh.nne_bulk
nelemBulk = mesh.nelem_bulk
nneChz = mesh.nne_cohesive
nelemChz = mesh.nelem_cohesive
ndim = mesh.ndim


# mesh definition, displacement, external forces
coor = mesh.coor
connBulk = mesh.conn_bulk
connChz = mesh.conn_cohesive
dofs = mesh.dofs

disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        dofs[mesh.nodesTopFace, 2],
        dofs[mesh.nodesBottomFace, 2],
        dofs[mesh.nodesBottomFace, 1],
        dofs[mesh.nodesBottomFace, 0]
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

# TO DO
elemChz = GooseFEM.Element.Cohesive8.Quadrature(vector.AsElement(coor, connChz))

nipBulk = elemBulk.nip
nipChz = elemChz.nip

# material definition
# -------------------
# Bulk material initialization
matBulk = GMatElastoPlast.LinearHardening2d(
    K=np.ones([nelemBulk, nipBulk])*170,
    G=np.ones([nelemBulk, nipBulk])*80,
    tauy0=np.ones([nelemBulk, nipBulk])*50,
    H=np.ones([nelemBulk, nipBulk])*1.0)


# Cohesive zone material initialization
matChz = GooseFEM.ConstitutiveModels.CohesiveBilinear3d(
    Kn=np.ones([nelemChz, nipChz])*30,
    Kt=np.ones([nelemChz, nipChz])*30,
    delta0=np.ones([nelemChz, nipChz])*0.02,
    deltafrac=np.ones([nelemChz, nipChz])*0.1,
    beta=np.ones([nelemChz, nipChz])*1.0,
    eta = np.ones([nelemChz, nipChz])*5e-03
    )    


I2 = GMatTensor.Cartesian3d.Array2d(matBulk.shape).I2  
# simulation variables
# --------------------
ueBulk = vector.AsElement(disp, connBulk)
ueChz = vector.AsElement(disp, connChz)
cooreBulk = vector.AsElement(coor, connBulk)
cooreChz = vector.AsElement(coor, connChz)
elemBulk0.gradN_vector((cooreBulk + ueBulk), matBulk.F)
matBulk.F += I2  # Add out-of-plane stretch = 1.0 (identity)
matBulk.refresh()

# internal force of the right hand side per element and assembly
feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
feChz = elemChz.Int_N_dot_traction_dA(matChz.T)
fint = vector.AssembleNode(feBulk, connBulk)
fint += vector.AssembleNode(feChz, connChz)

# initial element tangential stiffness matrix incorporating the geometrical and material stiffness matrix
KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)
KeChz = elemChz.Int_BT_D_B_dA(matChz.C)

K.clear()
K.assemble(KeBulk, connBulk)
K.assemble(KeChz, connChz)
K.finalize()

# initial residual
fres = fext - fint

# solve
# -----
ninc = 501
max_iter = 50
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

du = np.zeros_like(disp)
du_last = np.zeros_like(vector.AsDofs_u(disp))

initial_guess = np.zeros_like(disp)

total_time = 1.0 # pseudo time
dt = total_time / ninc # pseudo time increment

residual_history = []

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):

    # empty displacement update
    # du.fill(0.0)
    # update displacement
    disp[mesh.nodesTopFace, 2] += (+0.09/ninc)
    disp[mesh.nodesBottomFace, 0] = 0.0  # not strictly needed: default == 0
    disp[mesh.nodesBottomFace, 1] = 0.0  # not strictly needed: default == 0
    disp[mesh.nodesBottomFace, 2] = 0.0  # not strictly needed: default == 0
    

    # convergence flag
    converged = False
    
    matBulk.increment()
    matChz.increment()

    # elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
    # elemChz.update_x(vector.AsElement(coor + disp, connChz))

    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))

    total_increment = initial_guess.copy()  
    for iter in range(max_iter): 
        # update element wise displacments
        ueBulk = vector.AsElement(disp, connBulk) 
        ueChz = vector.AsElement(disp, connChz)

        # update deformation gradient F
        elemBulk0.symGradN_vector(ueBulk, matBulk.F)
        matBulk.F += I2
        matBulk.refresh()

        # update nodal displacements of cohesive zone
        elemChz.relative_disp(ueChz, matChz.delta, matChz.ori)
        matChz.refresh(dt)
  
        # update internal forces and assemble
        feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)

        # update tractions of cohesive zone
        feChz = elemChz.Int_N_dot_traction_dA(matChz.T)

        fint = vector.AssembleNode(feBulk, connBulk)
        fint += vector.AssembleNode(feChz, connChz)

        # update stiffness matrix
        KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)
        KeChz = elemChz.Int_BT_D_B_dA(matChz.C)

        K.clear()
        K.assemble(KeBulk, connBulk)
        K.assemble(KeChz, connChz)
        K.finalize()

        fres = -fint

        if iter > 0:
            # residual 
            fres_u = -vector.AsDofs_u(fint)
            
            res_norm = np.linalg.norm(fres_u) 

            residual_history.append(res_norm) 

            if res_norm < 5e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
                converged = True
                break

            # solve
            # du.fill(0.0)

        Solver.solve(K, fres, du)

        # add newly found delta_u to total increment
        total_increment += du

        # update displacement vector
        disp += du

        # update shape functions
        # elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
        elemChz.update_x(vector.AsElement(coor + disp, connChz))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1)))
    sigeq[ilam] = np.average(GMatElastoPlast.Sigeq(np.average(matBulk.Sig, axis=1)))
    
    if converged:
        # for r, val in enumerate(residual_history):
        #     print(f"Residual at iter {r}: {val}")
        # residual_history.clear()
        initial_guess = 1.0 * total_increment
        continue
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# ------------
# strain
print(disp)
elemBulk0.symGradN_vector(ueBulk, matBulk.F)
elemChz.relative_disp(ueChz, matChz.delta, matChz.ori)
matBulk.F += I2
matBulk.refresh()  
matChz.refresh(dt)


feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
feChz = elemChz.Int_N_dot_traction_dA(matChz.T)
# internal force
elemBulk.int_gradN_dot_tensor2_dV(matBulk.Sig, feBulk)
elemChz.int_N_dot_traction_dA(matChz.T, feChz)
fint = vector.AssembleNode(feBulk, connBulk)
fint += vector.AssembleNode(feChz, connChz)

# apply reaction force
fres_u = -vector.AsDofs_u(fint)
            
res_norm = np.linalg.norm(fres_u)
# print residual
# assert np.isclose(res_norm, 0,  atol=1e-6)

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
    conn_all = np.concatenate((connBulk, connChz), axis=0)

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
    for element in conn_all:
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
    for element in conn_all:
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