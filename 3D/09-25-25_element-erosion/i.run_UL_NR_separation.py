import argparse
import sys

import GMatTensor.Cartesian3d
import GMatElastoPlasticFiniteStrainSimo.Cartesian3d as GMat
import GooseFEM
import numpy as np
import random
from srcTopDown.helper_functions.element_erosion import element_erosion_3D_PBC
# mesh
# ----

def newton_raphson_solve(
    disp,
    initial_guess,
    max_iter,
    periodicity,
    vector,
    conn,
    elem0,
    elem,
    mat,
    damage_prev,
    fe,
    fint,
    K,
    Fext,
    Fint,
    Solver,
    du,
    F,
    I2,
    coor,
    mode,
    RES_TOL=1e-06
):
    """
    Performs the Newton-Raphson iterative solve for one load increment.

    Returns:
        disp (np.array): Updated displacement vector.
        elem (GooseFEM.Element): Updated element object (due to erosion).
        mat (GMat): Updated material state.
        converged (bool): True if convergence was reached.
        total_increment (np.array): Total displacement increment applied.
    """
    converged = False
    
    # mat.increment() is performed *outside* the loop in your original script
    # disp += initial_guess is also performed *outside* the loop
    
    total_increment = initial_guess.copy()
    damage_prev = mat.D_damage.copy()
    
    # The actual N-R loop
    for iter in range(max_iter):
        # 1. Kinematics/Material Update
        ue = vector.AsElement(disp, conn)
        elem0.symGradN_vector(ue, mat.F)
        mat.F += I2
        mat.refresh()

        # 2. Internal Force
        fe = elem.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe, conn)
        
        # 3. Stiffness Matrix
        Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.clear()
        K.assemble(Ke, conn)
        K.finalize(stabilize=True)
        # 4. Residual
        fres = -fint # Assuming fext is zero here for the residual force calc
                      # You should define/pass fext if it's non-zero.
                      # Since fext isn't updated in the loop, we use only fint for fres.

        # 5. Convergence Check
        if iter > 0:
            # - internal/external force as DOFs (account for periodicity)
            # You seem to be calculating reaction forces for the check:
            vector.asDofs_i(fint, Fint)
            vector.copy_p(Fint, Fext) # This is a strange use; Fext is usually external force.
                                      # Assuming you mean to calculate reaction forces on prescribed DoFs (p-part).
            
            # Recalculate the residual and norm based on your original logic:
            vector.asDofs_i(fres, Fext) # Reuse Fext array for fres on internal DoFs
            nfres = np.sum(np.abs(Fext))
            nfint = np.sum(np.abs(Fint)) # Use Fint as a measure of total force magnitude
            
            if nfint:
                res = nfres / nfint # Relative residual to internal force
            else:
                res = nfres
            
            if res < RES_TOL:
                print (f"Converged at Iter {iter}, Residual = {res}")
                converged = True
                if mode is 'load_step':
                    if (np.amax(mat.D_damage) < 1):
                        break
                    else:
                        curr_failed = failed_elem.copy()
                        # delete elements with damage > 1 and append to vector
                        for k, obj in enumerate(mat.D_damage):
                            if any(IP >= 1 for IP in obj):
                                curr_failed.add(k)
                                # break

                        newly_failed = curr_failed - failed_elem
                        failed_elem = curr_failed.copy()

                        to_be_deleted.extend(list(newly_failed))

                        if to_be_deleted:
                            print(f"INFO: Element(s) failed: {to_be_deleted}.")
                            elem_to_delete = {to_be_deleted.pop(0)} 
                            print( f"INFO: Deleting element {elem_to_delete}.")
                            mat.delete_element(elem_to_delete[0])
                            mat.D_damage = damage_prev
                            decreasing_steps = 11
                            new_fext = np.zeros_like(fe)
                            initial_guess = np.zeros_like(disp)
                            for incr_factor in decreasing_steps:
                                new_fext[elem_to_delete[0]] = -incr_factor * fe[elem_to_delete[0]]
                                fext = fext.copy() + vector.AssembleNode(new_fext, conn)    
                                forces_smoothed = False
                                disp, elem, mat, forces_smoothed, total_increment = newton_raphson_solve(
                                        disp,
                                        initial_guess,
                                        max_iter,
                                        periodicity,
                                        vector,
                                        conn,
                                        elem0,
                                        elem,
                                        mat,
                                        damage_prev,
                                        fe,
                                        fint,
                                        K,
                                        Fext,
                                        Fint,
                                        Solver,
                                        du,
                                        F,
                                        I2,
                                        coor,
                                        mode='force_step',
                                        RES_TOL=1e-06
                                ) 
                                if not forces_smoothed:
                                    raise RuntimeError(f"Element erosion for element {elem_to_delete} unsuccessful.")     
        
        # 6. Solve and Update
        du.fill(0.0)

        # Initial displacement update (only on first iteration)
        if iter == 0 and mode == 'load_step':
             # Calculate the kinematic part of the current increment
             # The F is the total deformation gradient, so (F - I) is the total displacement gradient.
             du[periodicity.controlNodes, 0] = (F[0,:] - np.eye(3)[0, :]) 
             du[periodicity.controlNodes, 1] = (F[1,:] - np.eye(3)[1, :])  
             du[periodicity.controlNodes, 2] = (F[2,:] - np.eye(3)[2, :])
             
        # Solve the system
        Solver.solve(K, fres, du)
        
        # Add delta u
        disp += du
        total_increment += du
        
        # Update element coordinates (for large deformation)
        elem.update_x(vector.AsElement(coor + disp, conn))

    return disp, elem, mat, converged, total_increment


# define mesh
print("running a GooseFEM static PBC example...")
mesh = GooseFEM.Mesh.Hex8.Regular(10, 10, 10)

# mesh dimensions
nelem = mesh.nelem
nne = mesh.nne
ndim = mesh.ndim
tyinglist = mesh.nodesPeriodic

# mesh definition
coor = mesh.coor
conn = mesh.conn
dofs = mesh.dofs

# create control nodes
control = GooseFEM.Tyings.Control(coor, dofs)

# add control nodes
coor = control.coor

# list of prescribed DOFs (fixed node + control nodes)
iip = np.concatenate((
    dofs[np.array([mesh.nodesFrontBottomLeftCorner]), 0],
    dofs[np.array([mesh.nodesFrontBottomLeftCorner]), 1],
    dofs[np.array([mesh.nodesFrontBottomLeftCorner]), 2],    
    control.controlDofs[0],
    control.controlDofs[1],
    control.controlDofs[2]
))

# initialize my periodic boundary condition class
periodicity = GooseFEM.Tyings.Periodic(coor, control.dofs, control.controlDofs, tyinglist, iip)
dofs = periodicity.dofs

# simulation variables
# --------------------

# vector definition
vector = GooseFEM.VectorPartitionedTyings(dofs, periodicity.Cdu, periodicity.Cdp, periodicity.Cdi)

# element definition
elem0 = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, conn))
elem = GooseFEM.Element.Hex8.Quadrature(vector.AsElement(coor, conn))
nip = elem.nip

# nodal quantities
disp = np.zeros_like(coor)
du = np.zeros_like(coor)  # iterative displacement update
fint = np.zeros_like(coor)  # internal force
fext = np.zeros_like(coor)  # external force

# element vectors / matrix
ue = vector.AsElement(disp, conn)
coore = vector.AsElement(coor, conn)
fe = np.empty([nelem, nne, ndim])
Ke = np.empty([nelem, nne * ndim, nne * ndim])

# DOF values
Fext = np.zeros([periodicity.nni])
Fint = np.zeros([periodicity.nni])

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
tauy0 = randomizeMicrostr(nelem, nip, 0.7, 3.0, 0.3)
mat = GMat.LinearHardeningDamage2d(
    K=np.ones([nelem, nip])*170,
    G=np.ones([nelem, nip])*80,
    tauy0=tauy0,
    H=np.ones([nelem, nip])*1,
    D1=np.ones([nelem, nip])*0.1,
    D2=np.ones([nelem, nip])*0.2,
    D3=np.ones([nelem, nip])*-1.7
    )

# allocate system matrix
K = GooseFEM.MatrixPartitionedTyings(dofs, periodicity.Cdu, periodicity.Cdp)
Solver = GooseFEM.MatrixPartitionedTyingsSolver()

# array of unit tensor
I2 = GMatTensor.Cartesian3d.Array2d(mat.shape).I2

# solve
# -----
ninc = 1001
max_iter = 50
tangent = True

# initialize stress/strain arrays for eq. plastic strain / mises stress plot
epseq = np.zeros(ninc)
sigeq = np.zeros(ninc)

# ue = vector.AsElement(disp)
# du = np.zeros_like(disp)
initial_guess = np.zeros_like(disp)
total_increment = np.zeros_like(disp)
failed_elem = set()
to_be_deleted = []

# deformation gradient
F = np.array(
        [
            [1.0 + (0.15/ninc), 0.0, 0.0],
            [0.0, 1.0 / (1.0 + (0.15/ninc)), 0.0],
            [0.0, 0.0, 1.0]
        ]
    )

for ilam, lam in enumerate(np.linspace(0.0, 1.0, ninc)):

    converged = False

    mat.increment()

    disp += initial_guess
    total_increment = initial_guess.copy()
    damage_prev = mat.D_damage.copy()
    
    # Newton-Raphson solver scheme including element erosion
    disp, elem, mat, converged, total_increment = newton_raphson_solve(
    disp, initial_guess, max_iter, periodicity, vector, conn, elem0,
    mat, elem, fe, fint, K, Fext, Fint, Solver, du, F, I2, coor, mode='load_step',
    RES_TOL=1e-06)


    # accumulate strains and stresses
    epseq[ilam] = np.average(GMat.Epseq(np.average(GMat.Strain(mat.F), axis=1)))
    sigeq[ilam] = np.average(GMat.Sigeq(np.average(mat.Sig, axis=1)))
    
    if converged:
         # print(total_increment)
         initial_guess = 0.3 * total_increment
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