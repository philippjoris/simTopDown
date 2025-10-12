import argparse
import sys
import os

import GMatTensor.Cartesian2d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMatElastoPlast
import GooseFEM
import numpy as np
import random

# include functions from srcTopDown package
from srcTopDown.helper_functions.gmsh.parser_2D import parse_msh

#
# Example with 2D geometry with hole and horizontal cohesive zone
#

# mesh
# ----
mesh_file_name = "i.geo_complex.msh"
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
# connChz = np.vstack((connCohParticle, connCohInterface))

dofs = mesh["dofs"]
nelemBulk = len(connBulk)
nelemCohPart = len(connCohPart)
nelemCohInt = len(connCohInt)
ndim = 2

disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        dofs[mesh["LeftUpperLine"], 1],
        dofs[mesh["LeftLowerLine"], 1],
        # dofs[mesh["ParticleLine"], 0],
        # dofs[mesh["ParticleLine"], 1],
        # dofs[mesh["BottomLine"][1:-2], 1],
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
elemBulk = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, connBulk))
elemBulk0 = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, connBulk))
elemCohPart = GooseFEM.Element.Cohesive4.Quadrature(vector.AsElement(coor, connCohPart))
elemCohInt = GooseFEM.Element.Cohesive4.Quadrature(vector.AsElement(coor, connCohInt))

nipBulk = elemBulk.nip
nipCohPart = elemCohPart.nip
nipCohInt = elemCohInt.nip

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
matBulk = GMatElastoPlast.LinearHardeningDamage2d(
    K=np.ones([nelemBulk, nipBulk])*170,
    G=np.ones([nelemBulk, nipBulk])*80,
    tauy0=np.ones([nelemBulk, nipBulk])*10.0,
    H=np.ones([nelemBulk, nipBulk])*1.0,
    D1=np.ones([nelemBulk, nipBulk])*0.1,
    D2=np.ones([nelemBulk, nipBulk])*0.2,
    D3=np.ones([nelemBulk, nipBulk])*-1.7    
    )

# Cohesive zone material initialization for bulk Interface
matCohInt = GooseFEM.ConstitutiveModels.CohesiveBilinear2d(
    Kn=np.ones([nelemCohInt, nipCohInt])*30.0,
    Kt=np.ones([nelemCohInt, nipCohInt])*30.0,
    delta0=np.ones([nelemCohInt, nipCohInt])*0.02,
    deltafrac=np.ones([nelemCohInt, nipCohInt])*0.2,
    beta=np.ones([nelemCohInt, nipCohInt])*1.0,
    eta = np.ones([nelemCohInt, nipCohInt])*5e-03
    )    

# Cohesive zone material initialization for Particle Interface
matCohPart = GooseFEM.ConstitutiveModels.CohesiveBilinear2d(
    Kn=np.ones([nelemCohPart, nipCohPart])*30.0,
    Kt=np.ones([nelemCohPart, nipCohPart])*30.0,
    delta0=np.ones([nelemCohPart, nipCohPart])*0.02,
    deltafrac=np.ones([nelemCohPart, nipCohPart])*0.2,
    beta=np.ones([nelemCohPart, nipCohPart])*1.0,
    eta = np.ones([nelemCohPart, nipCohPart])*5e-03
    )      


I2 = GMatTensor.Cartesian3d.Array2d(matBulk.shape).I2  
# simulation variables
# --------------------
ueBulk = vector.AsElement(disp, connBulk)
ueCohPart = vector.AsElement(disp, connCohPart)
ueCohInt = vector.AsElement(disp, connCohInt)
cooreBulk = vector.AsElement(coor, connBulk)
cooreCohPart = vector.AsElement(coor, connCohPart)
cooreCohInt = vector.AsElement(coor, connCohInt)
elemBulk0.gradN_vector((cooreBulk + ueBulk), matBulk.F)
matBulk.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
matBulk.refresh()

# internal force of the right hand side per element and assembly
feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
feCohPart = elemCohPart.Int_N_dot_traction_dL(matCohPart.T)
feCohInt = elemCohInt.Int_N_dot_traction_dL(matCohInt.T)
fint = vector.AssembleNode(feBulk, connBulk)
fint += vector.AssembleNode(feCohPart, connCohPart)
fint += vector.AssembleNode(feCohInt, connCohInt)

# initial element tangential stiffness matrix incorporating the geometrical and material stiffness matrix
KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)
KeCohPart = elemCohPart.Int_BT_D_B_dL(matCohPart.C)
KeCohInt = elemCohInt.Int_BT_D_B_dL(matCohInt.C)

K.clear()
K.assemble(KeBulk, connBulk)
K.assemble(KeCohPart, connCohPart)
K.assemble(KeCohInt, connCohInt)
K.finalize()

# initial residual
fres = fext - fint

# solve
# -----
ninc = 10001
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
        print(matCohPart.Damage[:8])

    # empty displacement update
    # du.fill(0.0)
    # update displacement
    disp[mesh["LeftUpperLine"], 1] += (+0.7/ninc)
    disp[mesh["LeftLowerLine"], 1] += (-0.7/ninc)

    # disp[mesh["ParticleLine"], 0] = 0.0  
    # disp[mesh["ParticleLine"], 1] = 0.0 
    # disp[mesh["BottomLine"][1:-2], 1] = 0.0
    disp[mesh["RightLine"][0:1], 0] = 0.0  # not strictly needed: default == 0
    disp[mesh["RightLine"][0:1], 1] = 0.0  # not strictly needed: default == 0
    disp[mesh["RightLine"][1:], 1] = 0.0  # not strictly needed: default == 0
    

    # convergence flag
    converged = False
    
    matBulk.increment()
    matCohPart.increment()
    matCohInt.increment()

    # elemBulk.update_x(vector.AsElement(coor + disp, connBulk))
    # elemChz.update_x(vector.AsElement(coor + disp, connChz))

    disp = vector.NodeFromPartitioned(vector.AsDofs_u(disp) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(disp))

    total_increment = initial_guess.copy()  
    for iter in range(max_iter): 
        # update element wise displacments
        ueBulk = vector.AsElement(disp, connBulk) 
        ueCohPart = vector.AsElement(disp, connCohPart)
        ueCohInt = vector.AsElement(disp, connCohInt)

        # update deformation gradient F
        elemBulk.symGradN_vector(ueBulk, matBulk.F)
        matBulk.F += I2
        matBulk.refresh()

        # update nodal displacements of cohesive zone
        elemCohPart.relative_disp(ueCohPart, matCohPart.delta, matCohPart.ori)
        elemCohInt.relative_disp(ueCohInt, matCohInt.delta, matCohInt.ori)        
        matCohPart.refresh(dt)
        matCohInt.refresh(dt)
  
        # update internal forces and assemble
        feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)

        # update tractions of cohesive zone
        feCohPart = elemCohPart.Int_N_dot_traction_dL(matCohPart.T)
        feCohInt = elemCohInt.Int_N_dot_traction_dL(matCohInt.T)

        fint = vector.AssembleNode(feBulk, connBulk)
        fint += vector.AssembleNode(feCohPart, connCohPart)
        fint += vector.AssembleNode(feCohInt, connCohInt)

        # update stiffness matrix
        KeBulk = elemBulk.Int_gradN_dot_tensor4_dot_gradNT_dV(matBulk.C)        
        KeCohPart = elemCohPart.Int_BT_D_B_dL(matCohPart.C)
        KeCohInt = elemCohInt.Int_BT_D_B_dL(matCohInt.C)

        K.clear()
        K.assemble(KeBulk, connBulk)
        K.assemble(KeCohPart, connCohPart)
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
        elemCohPart.update_x(vector.AsElement(coor + disp, connCohPart))
        elemCohInt.update_x(vector.AsElement(coor + disp, connCohInt))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1)))
    sigeq[ilam] = np.average(GMatElastoPlast.Sigeq(np.average(matBulk.Sig, axis=1)))
    
    if converged:
        #for r, val in enumerate(residual_history):
        #    print(f"Residual at iter {r}: {val}")
        residual_history.clear()
        initial_guess = 1.0 * total_increment
        continue
    if not converged:
        print (f"WARNING: Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
        break
        #raise RuntimeError(f"Load step {ilam} failed to converge.")


# post-process
# ------------
# strain
print(matCohInt.Damage)
print(matCohPart.Damage)


elemBulk0.symGradN_vector(ueBulk, matBulk.F)
elemCohPart.relative_disp(ueCohPart, matCohPart.delta, matCohPart.ori)
elemCohInt.relative_disp(ueCohInt, matCohInt.delta, matCohInt.ori)
matBulk.F += I2
matBulk.refresh()  
matCohPart.refresh(dt)
matCohInt.refresh(dt)


feBulk = elemBulk.Int_gradN_dot_tensor2_dV(matBulk.Sig)
feCohPart = elemCohPart.Int_N_dot_traction_dL(matCohPart.T)
feCohInt = elemCohInt.Int_N_dot_traction_dL(matCohInt.T)
# internal force
fint = vector.AssembleNode(feBulk, connBulk)
fint += vector.AssembleNode(feCohPart, connCohPart)
fint += vector.AssembleNode(feCohInt, connCohInt)

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
    damageCohPart_av = np.average(matCohPart.Damage, axis = 1)
    damageCohInt_av = np.average(matCohInt.Damage, axis = 1)
    # Average eq. strain per element
    epseq_av = GMatElastoPlast.Epseq(np.average(GMatElastoPlast.Strain(matBulk.F), axis=1))

    #combine connectivity
    conn_all = np.concatenate((connBulk, connCohPart, connCohInt), axis=0)
    damage_all = np.concatenate((damageBulk_av, damageCohPart_av, damageCohInt_av), axis=0)

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
    ax.set_xlim(0.2, 4)
    ax.set_ylim(-1.0, 1.0)
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