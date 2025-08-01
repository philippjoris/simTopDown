import argparse
import sys
import os


import GMatTensor.Cartesian2d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
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
mesh = GooseFEM.Mesh.Quad4.Regular(30, 10)

# mesh dimensions
nelem = mesh.nelem
nne = mesh.nne
ndim = mesh.ndim

# mesh definition, displacement, external forces
coor = mesh.coor
conn = mesh.conn
dofs = mesh.dofs
disp = np.zeros_like(coor)
fext = np.zeros_like(coor)

# list of prescribed DOFs
iip = np.concatenate(
    (
        dofs[mesh.nodesRightEdge, 0],
        dofs[mesh.nodesRightEdge, 1],
        dofs[mesh.nodesLeftEdge, 1]
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
elem = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, conn))
elem0 = GooseFEM.Element.Quad4.QuadraturePlanar(vector.AsElement(coor, conn))
nip = elem.nip

# material definition
# -------------------
# mat = GMat.Elastic2d(K=np.ones([nelem, nip]), G=np.ones([nelem, nip]))
mat = GMat.LinearHardening2d(K=np.ones([nelem, nip])*170, G=np.ones([nelem, nip])*80, tauy0=np.ones([nelem, nip])*.200, H=np.ones([nelem, nip])*3)

# simulation variables
# --------------------
ue = vector.AsElement(disp, conn)
coore = vector.AsElement(coor, conn)
elem.gradN_vector((coore + ue), mat.F)
mat.F[:, :, 2, 2] = 1.0  # Add out-of-plane stretch = 1.0 (identity)
mat.refresh()

# internal force of the right hand side per element and assembly
fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
fint = vector.AssembleNode(fe, conn)

# initial element tangential stiffness matrix incorporating the geometrical and material stiffness matrix
Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
K.clear()
K.assemble(Ke, conn)
K.finalize()
# initial residual
fres = fext - fint

# solve
# -----
ninc = 1501
max_iter = 70
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

du = np.zeros_like(disp)

residual_history = []

initial_guess = np.zeros_like(disp)

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):
    # store old displacement
    xp = vector.AsDofs_p(disp).copy()
    du.fill(0.0)

    # update displacement
    du[mesh.nodesLeftEdge, 1] = (+2.0/ninc)
    du[mesh.nodesRightEdge, 0] = 0.0  # not strictly needed: default == 0
    du[mesh.nodesRightEdge, 1] = 0.0  # not strictly needed: default == 0

    # convergence flag
    converged = False

    mat.increment()

    elem.update_x(vector.AsElement(coor + disp, conn))
    
    # impose initial guess on unknown displacements
    du = vector.NodeFromPartitioned(vector.AsDofs_u(du) + vector.AsDofs_u(initial_guess), vector.AsDofs_p(du))
    disp += du

    total_increment = initial_guess.copy()
    for iter in range(max_iter): 
        # update element wise displacments
        ue = vector.AsElement(disp, conn)

        # update deformation gradient F
        elem0.symGradN_vector(ue, mat.F)
        mat.F += GMatTensor.Cartesian3d.Array2d(mat.shape).I2 
        mat.refresh()  
  
        # update internal forces and assemble
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe, conn) 

        # update stiffness matrix        
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.clear()
        K.assemble(Ke, conn)
        K.finalize()

        fres = -fint
        # residual
        if iter > 0: 
            fres_u = -vector.AsDofs_u(fint)
            
            res_norm = np.linalg.norm(fres_u) 
            # print (f"Iter {iter}, Residual = {res_norm}")
            if (res_norm) < 1e-06:
                print (f"Increment {ilam}/{ninc} converged at Iter {iter}, Residual = {res_norm}")
                converged = True
                break
            
            #fill
        du.fill(0.0)

        # solve
        Solver.solve(K, fres, du)
        
        # add newly found delta_u to total increment
        total_increment += du
        
        # update displacement vector
        disp += du

        elem.update_x(vector.AsElement(coor+disp, conn))

    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))

    if converged:
        initial_guess = 1.0 * total_increment
        continue
    if not converged:
        raise RuntimeError(f"Load step {ilam} failed to converge.")

print(disp)
# post-process
# ------------
# strain
elem0.symGradN_vector(ue, mat.F)
mat.F += GMatTensor.Cartesian3d.Array2d(mat.shape).I2 
mat.refresh()  

# internal force
elem.int_gradN_dot_tensor2_dV(mat.Sig, fe)
vector.assembleNode(fe, conn, fint)

# apply reaction force
vector.copy_p(fint, fext)

# residual
fres = vector.AsDofs_u(fext) - vector.AsDofs_u(fint)
# print residual
# assert np.isclose(np.sum(np.abs(fres)) / np.sum(np.abs(vector.AsDofs_u(fext))), 0,  atol=1e-6)
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
    dV = elem.AsTensor(2, elem.dV)
    Sigav = np.average(mat.Sig, weights=dV, axis=1)
    sigeq_av = GMat.Sigeq(Sigav)
    
    # Average eq. strain per element
    epseq_av = GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    # ax.set_xlim(-7,90)
    # ax.set_ylim(0,110)
    
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
    gplt.patch(coor=coor + disp, conn=conn, cindex=epseq_av, cmap="jet", axis=ax)
    # gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)
    # ax.set_xlim(-7,90)
    # ax.set_ylim(0,110)
    
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

    # plot
    fig, ax = plt.subplots()
    ax.plot(epseq, sigeq, c="r", label=r"LinearHardening")

    # optional save
    if args.save:
        fig.savefig('fixed-disp_sig-eps.pdf')
    else:
        plt.show()

    plt.close(fig)