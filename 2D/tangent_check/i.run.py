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
mesh = GooseFEM.MeshCohesive.Quad4.RegularCohesive(1, 1, 1, 1.0)

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
        dofs[mesh.nodesTopEdge, 1],
        dofs[mesh.nodesBottomEdge, 1],
        dofs[mesh.nodesBottomEdge, 0]
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
elemBulk = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, connBulk))
elemBulk0 = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, connBulk))

# TO DO
elemChz = GooseFEM.Element.Cohesive4.Quadrature(vector.AsElement(coor, connChz))

nipBulk = elemBulk.nip
nipChz = elemChz.nip

# material definition
# -------------------
def randomizeMicrostr(nelem, nip, fraction_soft, value_hard, value_soft):
    array = np.ones([nelem, nip])*value_hard
    nsoft = round(fraction_soft * nelem)
    softelem = random.sample(range(nelem), k=nsoft)
    for elem in softelem:
        array[elem] *= (value_soft/value_hard)
    return array
# -------------------
# mat = GMat.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))
# tauy0 = randomizeMicrostr(nelem, nip, 0.7, .600, .200)
matBulk = GMatElastoPlast.LinearHardening2d(
    K=np.ones([nelemBulk, nipBulk])*170,
    G=np.ones([nelemBulk, nipBulk])*80,
    tauy0=np.ones([nelemBulk, nipBulk])*50,
    H=np.ones([nelemBulk, nipBulk])*1.0)


# Cohesive zone material initialization
matChz = GooseFEM.ConstitutiveModels.CohesiveBilinear2d(
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
matBulk.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
matBulk.refresh()

# internal force of the right hand side per element and assembly
feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
feChz = elemChz.Int_N_dot_traction_dL(matChz.T)
fint = vector.AssembleNode(feBulk, connBulk)
fint += vector.AssembleNode(feChz, connChz)

# initial element tangential stiffness matrix incorporating the geometrical and material stiffness matrix
KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)
KeChz = elemChz.Int_BT_D_B_dL(matChz.C)

K.clear()
K.assemble(KeBulk, connBulk)
K.assemble(KeChz, connChz)
K.finalize()

# initial residual
fres = fext - fint

# solve
# -----
ninc = 5001
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
    du.fill(0.0)
    # update displacement
    du[mesh.nodesTopEdge, 1] = (+0.1/ninc)
    du[mesh.nodesTopEdge, 0] = 0.0  # not strictly needed: default == 0
    du[mesh.nodesBottomEdge, 0] = 0.0  # not strictly needed: default == 0
    du[mesh.nodesBottomEdge, 1] = 0.0  # not strictly needed: default == 0
    

    # convergence flag
    converged = False
    
    matBulk.increment()
    matChz.increment()

    elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
    elemChz.update_x(vector.AsElement(coor + disp, connChz))

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
        feChz = elemChz.Int_N_dot_traction_dL(matChz.T)

        fint = vector.AssembleNode(feBulk, connBulk)
        fint += vector.AssembleNode(feChz, connChz)

        # update stiffness matrix
        KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)
        KeChz = elemChz.Int_BT_D_B_dL(matChz.C)

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
            du.fill(0.0)

        Solver.solve(K, fres, du)

        # update displacement vector
        disp += du

        # update shape functions
        elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
        elemChz.update_x(vector.AsElement(coor + disp, connChz))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1)))
    sigeq[ilam] = np.average(GMatElastoPlast.Sigeq(np.average(matBulk.Sig, axis=1)))
    
    if converged:
        # for r, val in enumerate(residual_history):
        #     print(f"Residual at iter {r}: {val}")
        # residual_history.clear()
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
feChz = elemChz.Int_N_dot_traction_dL(matChz.T)
# internal force
elemBulk.int_gradN_dot_tensor2_dV(matBulk.Sig, feBulk)
elemChz.int_N_dot_traction_dL(matChz.T, feChz)
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
    import GooseMPL as gplt
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plt.style.use(["goose", "goose-latex"])

    # Average equivalent stress per element
    dV = elemBulk.AsTensor(2, elemBulk.dV)
    Sigav = np.average(matBulk.Sig, weights=dV, axis=1)
    sigeq_av = GMatElastoPlast.Sigeq(Sigav)
    damageBulk_av = np.zeros_like(sigeq_av)
    damageChz_av = np.average(matChz.Damage, axis = 1)
    # Average eq. strain per element
    epseq_av = GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1))

    #combine connectivity
    conn_all = np.concatenate((connBulk, connChz), axis=0)
    damage_all = np.concatenate((damageBulk_av, damageChz_av), axis=0)

    # plot stress
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=connBulk, cindex=sigeq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(sigeq_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent mises stress")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_contour_sig.pdf')
    else:
        plt.show()

    plt.close(fig)

    # plot strain
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=connBulk, cindex=epseq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(epseq_av)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Equivalent strain")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_contour_eps.pdf')
    else:
        plt.show()

    plt.close(fig)

    # plot Damage
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn_all, cindex=damage_all, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    
    # Add colorbar
    mappable = ScalarMappable(norm=plt.Normalize(), cmap=plt.colormaps["jet"])
    mappable.set_array(damage_all)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Damage")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_damage.pdf')
    else:
        plt.show()

    plt.close(fig)    

    # plot
    fig, ax = plt.subplots()
    ax.plot(epseq, sigeq, c="r", label=r"LinearHardening")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_sig-eps.pdf')
    else:
        plt.show()

    plt.close(fig)