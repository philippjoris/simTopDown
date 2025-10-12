import argparse
import sys
import os

import GMatTensor.Cartesian2d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMatElastoPlast
import GooseFEM
import numpy as np
import random
from scipy.sparse.linalg import eigs

# include functions from srcTopDown package
from srcTopDown.helper_functions.gmsh.parser_2D_particleBulk import parse_msh
from srcTopDown.helper_functions.element_erosion import element_erosion_multiplemat

#
# Example with 2D geometry with hole and horizontal cohesive zone
#

# mesh
# ----
mesh_file_name = "i.geo_w_particle.msh2"
curr_dir = os.path.dirname(os.path.abspath(__file__))
mesh_file_path = os.path.join(curr_dir, mesh_file_name)

if os.path.exists(mesh_file_path):
    print(f"Parsing mesh: {mesh_file_path}")
    mesh = parse_msh( 
        msh_filepath=mesh_file_path
    )
else:
    print(f"Error: Mesh file not found at {mesh_file_path}")

nr_materials = 2
ndim = 2

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
conn[1] = mesh["conn_particle"]
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
        dofs[mesh["TopLine"][:-1], 1],
        dofs[mesh["BottomLine"][:-1], 1],
        dofs[mesh["RightLine"][0:1], 0],
        dofs[mesh["RightLine"][0:1], 1],
        dofs[mesh["RightLine"][1:], 1]
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
for i in range(nr_materials):
    elem[i] = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, conn[i]))
nipBulk = elem[0].nip

# material definition
# -------------------
mat[0] = GMatElastoPlast.LinearHardeningDamage2d(
    K=np.ones([nelem[0], nipBulk])*170,
    G=np.ones([nelem[0], nipBulk])*80,
    tauy0=np.ones([nelem[0], nipBulk])*10.0,
    H=np.ones([nelem[0], nipBulk])*1.0,
    D1=np.ones([nelem[0], nipBulk])*0.1,
    D2=np.ones([nelem[0], nipBulk])*0.2,
    D3=np.ones([nelem[0], nipBulk])*-1.7    
    )

mat[1] = GMatElastoPlast.LinearHardeningDamage2d(
    K=np.ones([nelem[1], nipBulk])*1700,
    G=np.ones([nelem[1], nipBulk])*800,
    tauy0=np.ones([nelem[1], nipBulk])*100.0,
    H=np.ones([nelem[1], nipBulk])*1.0,
    D1=np.ones([nelem[1], nipBulk])*0.1,
    D2=np.ones([nelem[1], nipBulk])*0.2,
    D3=np.ones([nelem[1], nipBulk])*-1.7    
    )

K.clear()
for i in range(nr_materials):
    I2[i] = GMatTensor.Cartesian3d.Array2d(mat[i].shape).I2  
    ue[i] = vector.AsElement(disp, conn[i])
    coore[i] = vector.AsElement(coor, conn[i]) 
    elem[i].gradN_vector((coore[i] + ue[i]), mat[i].F)
    mat[i].F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
    mat[i].refresh()
    fe[i] = elem[i].Int_gradN_dot_tensor2_dV(mat[i].Sig)
    Ke[i] = elem[i].Int_gradN_dot_tensor4_dot_gradNT_dV(mat[i].C)
    K.assemble(Ke[i], conn[i])
K.finalize()
# simulation variables
# --------------------

# internal force of the right hand side per element and assembly
fint = vector.AssembleNode(fe[0], conn[0])
for i in range(1,nr_materials):
    fint += vector.AssembleNode(fe[i], conn[i])

# initial residual
fres = fext - fint

# solve
# -----
ninc = 4000
max_iter = 20
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

du = np.zeros_like(disp)

total_time = 1.0 # pseudo time

dt = total_time / ninc # pseudo time increment

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
    for i, node in enumerate(mesh["TopLine"]):
        factor = 1.0 - (coor[node][0] - coor[mesh["TopLine"][0]][0]) / (coor[mesh["TopLine"][-1]][0] - coor[mesh["TopLine"][0]][0])
        disp[node, 1] += (+0.5/ninc) * factor
    for i, node in enumerate(mesh["BottomLine"]):
        factor = 1.0 - (coor[node][0] - coor[mesh["BottomLine"][0]][0]) / (coor[mesh["BottomLine"][-1]][0] - coor[mesh["BottomLine"][0]][0])
        disp[node, 1] += (-0.5/ninc) * factor 

    disp[mesh["RightLine"][0:1], 0] = 0.0  # not strictly needed: default == 0
    disp[mesh["RightLine"][0:1], 1] = 0.0  # not strictly needed: default == 0
    disp[mesh["RightLine"][1:], 1] = 0.0  # not strictly needed: default == 0

    for i in range(nr_materials):
        mat[i].increment()

    # elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
    # elemChz.update_x(vector.AsElement(coor + disp, connChz))

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
                for i in range(nr_materials):
                    failure = False
                    if (np.amax(mat[i].D_damage)) > 1:
                        failure = True
                    if failure:
                        elem_failed = set()
                        elem_failed.update(elem_failed_prev[i])
                        # delete elements with damage > 1 and append to vector
                        
                        for k, obj in enumerate(mat[i].D_damage):
                            if any(IP >= 1 for IP in obj):
                                elem_failed.add(k)
                        
                        newly_failed = elem_failed - elem_failed_prev[i]
                        elem_failed_prev[i] = elem_failed.copy()

                        to_be_deleted[i].extend(list(newly_failed))

                        if to_be_deleted[i]:
                            print(f"INFO: Element(s) failed: {to_be_deleted[i]}.")
                            elem_to_delete = to_be_deleted[i].pop(0)
                            print( f"INFO: Deleting element {elem_to_delete}.")
                            disp, elem, mat = element_erosion_multiplemat(Solver, vector, conn, mat, damage_prev, elem, fe,                                                                      
                                                                        fext, disp, elem_to_delete, K, fe, I2, i)
                        damage_prev[i] = mat[i].D_damage.copy()
                break
        Solver.solve(K, fres, du)

        # add newly found delta_u to total increment
        total_increment += du

        # update displacement vector
        disp += du

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(mat[0].F), axis=1)))
    sigeq[ilam] = np.average(GMatElastoPlast.Sigeq(np.average(mat[0].Sig, axis=1)))
    
    if converged:
        residual_history.clear()
        initial_guess = 1.0 * total_increment
        continue
    if not converged:
        print (f"WARNING: Increment {ilam}/{ninc} did not converged!")
        break


# post-process
# ------------
# strain

for i in range(nr_materials):
    ue[i] = vector.AsElement(disp, conn[i]) 
    elem[i].symGradN_vector(ue[i], mat[i].F)
    mat[i].F += I2[i]
    mat[i].refresh()
    fe[i] = elem[i].Int_gradN_dot_tensor2_dV(mat[i].Sig)

fint = vector.AssembleNode(fe[0], conn[0])
for i in range(1,nr_materials):
    fint += vector.AssembleNode(fe[i], conn[i])

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
    import GooseMPL as gplt
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plt.style.use(["goose", "goose-latex"])

    # Average equivalent stress per element
    # Collect per-material arrays
    Sigav_all = []
    sigeq_av_all = []
    epseq_av_all = []
    damage_av_all = []
    conn_all = []

    for e, m, c in zip(elem, mat, conn):
        dV = e.AsTensor(2, e.dV)

        # stresses
        Sigav = np.average(m.Sig, weights=dV, axis=1)
        Sigav_all.append(Sigav)
        sigeq_av_all.append(GMatElastoPlast.Sigeq(Sigav))

        # damage
        damage_av_all.append(np.average(m.D_damage, axis=1))

        # strains
        epseq_av_all.append(
            GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(m.F), axis=1))
        )

        # connectivity
        conn_all.append(c)

    # Concatenate across materials
    sigeq_av_all = np.concatenate(sigeq_av_all, axis=0)
    epseq_av_all = np.concatenate(epseq_av_all, axis=0)
    damage_av_all = np.concatenate(damage_av_all, axis=0)
    conn_all = np.concatenate(conn_all, axis=0)
    # plot stress
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn_all, cindex=sigeq_av_all, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(sigeq_av_all)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent mises stress")

    # optional save
    if args.save:
        fig.savefig('BcTopDown_contour_sig.pdf')
    else:
        plt.show()

    plt.close(fig)

    # plot strain
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn_all, cindex=epseq_av_all, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(epseq_av_all)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent strain")

    # optional save
    if args.save:
        fig.savefig('BcTopDown_contour_eps.pdf')
    else:
        plt.show()

    plt.close(fig)

    # plot Damage
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn_all, cindex=damage_av_all, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(damage_av_all)
    ax.set_xlim( 0.1, 6.)
    ax.set_ylim(-5.0, 5.0)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Damage")

    # optional save
    if args.save:
        fig.savefig('BcTopDown_damage.pdf')
    else:
        plt.show()

    plt.close(fig)    

    # plot
    fig, ax = plt.subplots()
    ax.plot(epseq, sigeq, c="r", label=r"LinearHardening")

    # optional save
    if args.save:
        fig.savefig('BcTopDown_sig-eps.pdf')
    else:
        plt.show()

    plt.close(fig)