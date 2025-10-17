import argparse
import sys
import os

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
import random
from srcTopDown.helper_functions.element_erosion import element_erosion_3D_PBC
from srcTopDown.helper_functions.gmsh.parser_3D import parse_msh
import srcTopDown.plot_functions.RVE_plot_3d as pf

# mesh
# ----
print("running a GooseFEM static example...")
mesh = GooseFEM.Mesh.Hex8.Regular(10, 10, 10)

# mesh dimensions
nelem = mesh.nelem
nne = mesh.nne
ndim = mesh.ndim

# mesh definition
coor = mesh.coor
conn = mesh.conn
dofs = mesh.dofs
disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        dofs[mesh.nodesRightFace, 0],
        dofs[mesh.nodesLeftFace, 0], 
        dofs[mesh.nodesBottomFace, 1], 
        dofs[mesh.nodesFrontFace, 2]
    )
)

# vector definition
vector = GooseFEM.VectorPartitioned(dofs, iip)

# allocate system matrix
K = GooseFEM.MatrixPartitioned(dofs, iip)
Solver = GooseFEM.MatrixPartitionedSolver()

# simulation variables
# --------------------
# element definition
elem = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, conn))
nipBulk = elem.nip
# matrix material
mat = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem, nipBulk])*170,
    G=np.ones([nelem, nipBulk])*80,
    tauy0=np.ones([nelem, nipBulk])*10.0,
    H=np.ones([nelem, nipBulk])*1.0,
    D1=np.ones([nelem, nipBulk])*0.1,
    D2=np.ones([nelem, nipBulk])*0.2,
    D3=np.ones([nelem, nipBulk])*-1.7    
    )

K.clear()
I2 = GMatTensor.Cartesian3d.Array2d(mat.shape).I2  
ue = vector.AsElement(disp, conn)
coore = vector.AsElement(coor, conn) 
elem.gradN_vector((ue), mat.F)
mat.F += I2
mat.refresh()
fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
K.assemble(Ke, conn)
K.finalize()

# internal force of the right hand side per element and assembly
fint = vector.AssembleNode(fe, conn)

# initial residual
fres = fext - fint


# -------------------- PLOT UNDEFORMED MESH ------------------------------------
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

    # Use a colormap with distinct colors
    cmap = plt.colormaps["tab10"]

    faces = [
        [0, 1, 2, 3],  # bottom face
        [4, 5, 6, 7],  # top face
        [0, 1, 5, 4],  # front face
        [1, 2, 6, 5],  # right face
        [2, 3, 7, 6],  # back face
        [3, 0, 4, 7]   # left face
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Loop over material groups
    for i, element in enumerate(conn):
        color = cmap(i / 1)  # assign one color per material group
        verts = np.array(coor[element])
        for face in faces:
            face_verts = verts[face].tolist()
            poly = Poly3DCollection([face_verts],
                                    facecolors=[color],
                                    edgecolor="k",
                                    linewidths=0.5)
            ax.add_collection3d(poly)

    # Labels and scaling
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.auto_scale_xyz(coor[:, 0], coor[:, 1], coor[:, 2])

    # Legend: one entry per material
    labels = ["Matrix", "Interface", "Particle"]
    handles = [plt.Rectangle((0,0),1,1,color=cmap(i/len(labels))) for i in range(len(labels))]
    ax.legend(handles, labels,
            loc="center left",      
            bbox_to_anchor=(-0.2, 0.0)) 

    fig.savefig('undeformed_mesh.pdf')
# -----------------------------------------------------------------------------------

# solve
# -----
ninc = 1001
max_iter = 20
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

du = np.zeros_like(disp)


residual_history = []

initial_guess = np.zeros_like(disp)

elem_failed_prev = []
to_be_deleted = []

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    # update displacement
    # disp[mesh["face_rgt"], 1] = 0.0
    disp[mesh.nodesRightFace, 0] += 2.0/ninc  
    disp[mesh.nodesLeftFace, 0] = 0.0  
    disp[mesh.nodesBottomFace, 1] = 0.0  
    disp[mesh.nodesFrontFace, 2] = 0.0  
    
    mat.increment()

    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))
    
    total_increment = initial_guess.copy()
    converged = False

    for iter in range(max_iter): 
        # update element wise displacments
        K.clear()
        ue = vector.AsElement(disp, conn) 
        elem.symGradN_vector(ue, mat.F)
        mat.F += I2
        mat.refresh()
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C) 
        K.assemble(Ke, conn) 
        K.finalize(stabilize=True)

        fint = vector.AssembleNode(fe, conn)

        fres = -fint

        if iter > 0:
            # residual 
            fres_u = -vector.AsDofs_u(fint)
            
            res_norm = np.linalg.norm(fres_u) 

            residual_history.append(res_norm)

            if res_norm < 1e-07:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
                converged = True                
                break

        Solver.solve(K, fres, du)

        # add newly found delta_u to total increment
        total_increment += du

        # update displacement vector
        disp += du

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    
    if converged:
        residual_history.clear()
        initial_guess = 1.0 * total_increment
        continue
    if not converged:
        print (f"WARNING: Increment {ilam}/{ninc} did not converged!")
        break


# post-process
# plot
# ----

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