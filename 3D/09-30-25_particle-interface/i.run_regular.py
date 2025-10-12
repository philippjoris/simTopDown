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

# mesh
# ----
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

disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        dofs[mesh["RightSurface"], 0],
        dofs[mesh["RightSurface"], 2],        
        dofs[mesh["LeftSurface"], 0],
        dofs[mesh["LeftSurface"], 1],
        dofs[mesh["LeftSurface"], 2]
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
for i in range(nr_materials):
    elem[i] = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, conn[i]))
nipBulk = elem[0].nip

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
    for i, (m, elements) in enumerate(zip(mat, conn)):
        color = cmap(i / len(mat))  # assign one color per material group
        for element in elements:    # elements belonging to this material
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
elem_failed_prev = np.empty(nr_materials, dtype=object)
to_be_deleted = np.empty(nr_materials, dtype=object)

for j in range(nr_materials):
    elem_failed_prev[j] = set()
    to_be_deleted[j] = []

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    # if ilam > 1190:
    #    break
    # update displacement
    disp[mesh["RightSurface"], 0] -= 0.0
    disp[mesh["RightSurface"], 2] -= 2.0/ninc  
    disp[mesh["LeftSurface"], 0] = 0.0  
    disp[mesh["LeftSurface"], 1] = 0.0  
    disp[mesh["LeftSurface"], 2] = 0.0
    
    for i in range(nr_materials):
        mat[i].increment()

    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))
    
    for i in range(len(damage_prev)):
        damage_prev[i] = mat[i].D_damage.copy()
    total_increment = initial_guess.copy()
    converged = False

    for iter in range(max_iter): 
        # update element wise displacments
        K.clear()
        for i in range(nr_materials):
            ue[i] = vector.AsElement(disp, conn[i]) 
            elem[i].symGradN_vector(ue[i], mat[i].F)
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
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat[0].F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat[0].Sig, axis=1)))
    
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
    # Collect per-material arrays
    Sigav_all = []
    sigeq_av_all = []
    epseq_av_all = []
    damage_av_all = []
    conn_all = []

    # Average equivalent stress per element
    for e, m, c in zip(elem, mat, conn):
        dV = e.AsTensor(2, e.dV)

        # stresses
        Sigav = np.average(m.Sig, weights=dV, axis=1)
        Sigav_all.append(Sigav)
        sigeq_av_all.append(GMat.Sigeq(Sigav))

        # damage
        damage_av_all.append(np.average(m.D_damage, axis=1))

        # strains
        epseq_av_all.append(
            GMat.Epseq(np.average(GMat.Strain(m.F), axis=1))
        )

        # connectivity
        conn_all.append(c)

    # Concatenate across materials
    sigeq_av_all = np.concatenate(sigeq_av_all, axis=0)
    epseq_av_all = np.concatenate(epseq_av_all, axis=0)
    damage_av_all = np.concatenate(damage_av_all, axis=0)
    conn_all = np.concatenate(conn_all, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values for colormap
    cmap = plt.colormaps["jet"]
    norm = plt.Normalize(vmin=sigeq_av_all.min(), vmax=sigeq_av_all.max())
    colors = cmap(norm(sigeq_av_all))

    faces = [
    [0, 1, 2, 3],  # bottom face
    [4, 5, 6, 7],  # top face
    [0, 1, 5, 4],  # front face
    [1, 2, 6, 5],  # right face
    [2, 3, 7, 6],  # back face
    [3, 0, 4, 7]   # left face
    ]

    # Plot deformed mesh with colors
    for i, element in enumerate(conn_all):
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
    mappable.set_array(sigeq_av_all)
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
    norm = plt.Normalize(vmin=epseq_av_all.min(), vmax=epseq_av_all.max())
    colors = cmap(norm(epseq_av_all))
    

    # Plot deformed mesh with colors
    for i, element in enumerate(conn_all):
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
    mappable.set_array(epseq_av_all)
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
    norm = plt.Normalize(vmin=damage_av_all.min(), vmax=damage_av_all.max())
    colors = cmap(norm(damage_av_all))
    

    # Plot deformed mesh with colors
    for i, element in enumerate(conn_all):
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
    mappable.set_array(damage_av_all)
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
