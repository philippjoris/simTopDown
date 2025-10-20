"""
file: geom.py
author: Philipp van der Loos

Creates a Gmsh 3D geometry: a quadratic cube with mutiple small quadratic volumes inside.
Grid mesh with hexaeder elements.
"""


import gmsh
import random
import math
import time
import json
import numpy as np

# --- USER INPUTS ---
# Outer cube size (L x L x L) in mm
L_CUBE = 10.0
#
n_elem_line = 10  # Number of elements along x
#
# Interface layer
t_interface = L_CUBE/n_elem_line
#
# Side length of a single quadratic particle (l_particle x l_particle x l_particle)
L_PARTICLE = 2.0
# Target volume fraction (VF) of the particle phase (e.20 for 20%)
VF_TARGET = 0.04
# *** Elements per side of the particle (determines grid size) ***
N_ELEM_PARTICLE_SIDE = L_PARTICLE * (n_elem_line/L_CUBE)

# number of elements of the interface
N_ELEM_INTERFACE = 1

# Calculate the required element size based on particle dimension and desired elements
L_ELEM = L_PARTICLE / N_ELEM_PARTICLE_SIDE    
N_ELEM_CUBE_SIDE = round(L_CUBE / L_ELEM)
V_cube = L_CUBE**3
V_particle = L_PARTICLE**3
V_target = V_cube * VF_TARGET

# Maximum number of attempts to place a non-overlapping particle (set high for sparse packing)
MAX_PLACEMENT_ATTEMPTS = 500
# Seed for reproducibility (set to None for truly random placement)
RANDOM_SEED = 10
# Tolerance
TOL = 1e-7

# --- HELPER FUNCTION ---
def get_sorted_open_edge_nodes_by_coords(edge_node_tags, fixed_axes):
    """
    Sorts and removes corners from a given set of node tags based on fixed axes.
    """
    if edge_node_tags.size <= 2: # Edge must have at least 3 nodes (2 corners + 1 open)
        return np.array([], dtype=np.intp)
    
    # Get coordinates for the edge nodes
    node_indices = np.searchsorted(nodeTags, edge_node_tags)
    edge_coords = coord[node_indices]
    
    # Determine the running (sorting) axis (0=X, 1=Y, 2=Z)
    running_axis = next(i for i in range(3) if i != fixed_axes[0] and i != fixed_axes[1])
    
    # Sort by the running axis coordinate
    sort_indices = np.argsort(edge_coords[:, running_axis])
    sorted_node_tags = edge_node_tags[sort_indices]
    
    # Return open edge nodes (excluding corners)
    return sorted_node_tags[1:-1]

# --- INITIALIZATION AND CALCULATION ---
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# Calculate the required number of non-overlapping particles
N_particles_needed = math.ceil(V_target / V_particle)

gmsh.initialize()
gmsh.model.add("perfect_grid_composite")
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Geometry.Tolerance", 1e-9)

# Use the OpenCASCADE kernel for geometry definition
# Start with the outer cube definition (tag 1)
cube_tag = gmsh.model.occ.addBox(0, 0, 0, L_CUBE, L_CUBE, L_CUBE)

# Helper list to store bounding boxes of placed particles for element classification
particle_bboxes = []
particles_placed = 0

# Maximum grid index for particle placement
max_grid_index = L_CUBE - (L_PARTICLE - 1)
min_grid_index = 0

while particles_placed < N_particles_needed:
    attempt = 0
    placed_this_iteration = False

    while attempt < MAX_PLACEMENT_ATTEMPTS:
        # 1. Randomly choose the starting grid indices (0 to 9 for a 10-element grid)
        x_grid_idx = random.randint(min_grid_index, max_grid_index)
        y_grid_idx = random.randint(min_grid_index, max_grid_index)
        z_grid_idx = random.randint(min_grid_index, max_grid_index)

        # 2. Snap coordinates to the grid
        x = x_grid_idx * L_ELEM
        y = y_grid_idx * L_ELEM
        z = z_grid_idx * L_ELEM

        new_bbox_start = np.array([x, y, z])
        new_bbox_end = new_bbox_start + L_PARTICLE

        is_overlapping = False
        
        # 3. Perform the 3D Periodic Overlap Check
        for old_bbox in particle_bboxes:
            old_bbox_start = np.array(old_bbox[:3])
            old_bbox_end = np.array(old_bbox[3:])

            # Check overlap against the 27 periodic images of the *old* particle
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        # Shifted coordinates of the old particle's image
                        shift = np.array([i * L_CUBE, j * L_CUBE, k * L_CUBE])
                        shifted_old_bbox_start = old_bbox_start + shift
                        shifted_old_bbox_end = old_bbox_end + shift

                        # Standard overlap check: Non-intersection means no overlap.
                        # If they are NOT non-intersecting, they overlap.
                        # Note: We use TOL to handle floating point comparisons.
                        
                        non_intersecting = (
                            new_bbox_end[0] <= shifted_old_bbox_start[0] + TOL or new_bbox_start[0] >= shifted_old_bbox_end[0] - TOL or
                            new_bbox_end[1] <= shifted_old_bbox_start[1] + TOL or new_bbox_start[1] >= shifted_old_bbox_end[1] - TOL or
                            new_bbox_end[2] <= shifted_old_bbox_start[2] + TOL or new_bbox_start[2] >= shifted_old_bbox_end[2] - TOL
                        )

                        if not non_intersecting:
                            # Overlap found with one of the periodic images!
                            is_overlapping = True
                            break
                    if is_overlapping: break
                if is_overlapping: break

            if is_overlapping:
                break # Stop checking against other old particles

        # 4. Placement Decision
        if not is_overlapping:
            # Store the bounding box (which may straddle the L_CUBE boundary)
            particle_bboxes.append((x, y, z, x + L_PARTICLE, y + L_PARTICLE, z + L_PARTICLE))
            particles_placed += 1
            placed_this_iteration = True
            break # Exit inner while loop to place the next particle

        attempt += 1

    if not placed_this_iteration:
        print(f"Warning: Could only place {particles_placed} out of {N_particles_needed} required particles due to packing difficulty.")
        break

print(f"Total particles placed: {particles_placed}")


gmsh.model.occ.synchronize()

# --- PERFECT GRID MESHING (TRANSFINITE) ---
all_curves = gmsh.model.getEntities(1)
all_surfaces = gmsh.model.getEntities(2)
all_volumes = gmsh.model.getEntities(3)
print("Applying Transfinite constraints for perfect hexahedral grid...")


# 2. Set number of divisions for all bounding curves (lines)
for dim, tag in all_curves:
    bbox = gmsh.model.getBoundingBox(dim, tag)
    length = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    if math.isclose(length, L_CUBE, abs_tol=1e-6):
        gmsh.model.mesh.setTransfiniteCurve(tag, n_elem_line + 1)
    
# 3. Set all surfaces to be transfinite (Structured)
for dim, tag in all_surfaces:
    gmsh.model.mesh.setTransfiniteSurface(tag)

# 4. Set the volume to be transfinite
for dim, tag in all_volumes:
    gmsh.model.mesh.setTransfiniteVolume(tag) 

# Ensure recombination is on to guarantee hexes
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)    # 2D: Frontal-Delaunay for quads
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 3D: Structured (transfinite)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # Ensure transfinite meshing
# Generate the mesh
gmsh.model.mesh.generate(3)

# --- NODE SETS ------------------
left_surface_tags = []
right_surface_tags = []
back_surface_tags = []
front_surface_tags = []
bottom_surface_tags = []
top_surface_tags = []

for dim, c_id in all_surfaces:
    if dim == 2:
        bbox = gmsh.model.getBoundingBox(dim, c_id)
        # X=0 (Left)
        if (math.isclose(bbox[0], 0.0, abs_tol=TOL) and math.isclose(bbox[3], 0.0, abs_tol=TOL)):
            left_surface_tags.append(c_id)
        # X=L (Right)
        elif (math.isclose(bbox[0], L_CUBE, abs_tol=TOL) and math.isclose(bbox[3], L_CUBE, abs_tol=TOL)):
            right_surface_tags.append(c_id)
        # Y=0 (Front)
        elif (math.isclose(bbox[1], 0.0, abs_tol=TOL) and math.isclose(bbox[4], 0.0, abs_tol=TOL)):
            bottom_surface_tags.append(c_id)
        # Y=L (Back)
        elif (math.isclose(bbox[1], L_CUBE, abs_tol=TOL) and math.isclose(bbox[4], L_CUBE, abs_tol=TOL)):
            top_surface_tags.append(c_id)
        # Z=0 (Bottom)
        elif (math.isclose(bbox[2], 0.0, abs_tol=TOL) and math.isclose(bbox[5], 0.0, abs_tol=TOL)):
            front_surface_tags.append(c_id)
        # Z=L (Top)
        elif (math.isclose(bbox[2], L_CUBE, abs_tol=TOL) and math.isclose(bbox[5], L_CUBE, abs_tol=TOL)):
            back_surface_tags.append(c_id)

face_tags_map = {
    'lft': list(set(left_surface_tags)), 
    'rgt': list(set(right_surface_tags)),
    'bck': list(set(back_surface_tags)), 
    'fro': list(set(front_surface_tags)),
    'bot': list(set(bottom_surface_tags)), 
    'top': list(set(top_surface_tags)),
}


nodeTags, coord, _ = gmsh.model.mesh.getNodes()
coord = coord.reshape(-1, 3) # (N_nodes, 3) array
node_sets = {}

# ----------------------------------------------------------------------
# --- STEP 1: IDENTIFY CORNER NODES ---
# ----------------------------------------------------------------------

corners = {
    'FBL': (0.0, 0.0, 0.0), 'FBR': (L_CUBE, 0.0, 0.0), 
    'BBL': (0.0, 0.0, L_CUBE), 'BBR': (L_CUBE, 0.0, L_CUBE),
    'FTL': (0.0, L_CUBE, 0.0), 'FTR': (L_CUBE, L_CUBE, 0.0),
    'BTL': (0.0, L_CUBE, L_CUBE), 'BTR': (L_CUBE, L_CUBE, L_CUBE),
}

for name, (x, y, z) in corners.items():
    dist_sq = (coord[:, 0] - x)**2 + (coord[:, 1] - y)**2 + (coord[:, 2] - z)**2
    corner_idx = np.where(dist_sq < TOL**2)[0]
    if len(corner_idx) == 1:
        node_sets[f'corner_{name}'] = nodeTags[corner_idx[0]]
    else:
        print(f"Warning: Could not uniquely find corner {name}. Found {len(corner_idx)} nodes.")
        
all_corner_nodes = set(node_sets.values())

# ----------------------------------------------------------------------
# --- STEP 2: IDENTIFY AND SORT OPEN EDGE NODES ---
# ----------------------------------------------------------------------

edge_entities_map = {
    # Name : (BBOX_Target, Fixed_Axes)
    'froBot': ((0.0, 0.0, 0.0, L_CUBE, 0.0, 0.0), (1, 2)), # Fixed Y=0, Z=0. Running X (axis 0).
    'froTop': ((0.0, L_CUBE, 0.0, L_CUBE, L_CUBE, 0.0), (1, 2)), # Fixed Y=L, Z=0. Running X (axis 0).
    'bckBot': ((0.0, 0.0, L_CUBE, L_CUBE, 0.0, L_CUBE), (1, 2)), # Fixed Y=0, Z=L. Running X (axis 0).
    'bckTop': ((0.0, L_CUBE, L_CUBE, L_CUBE, L_CUBE, L_CUBE), (1, 2)), # Fixed Y=L, Z=L. Running X (axis 0).

    'froLft': ((0.0, 0.0, 0.0, 0.0, L_CUBE, 0.0), (0, 2)), # Fixed X=0, Z=0. Running Y (axis 1).
    'froRgt': ((L_CUBE, 0.0, 0.0, L_CUBE, L_CUBE, 0.0), (0, 2)), # Fixed X=L, Z=0. Running Y (axis 1).
    'bckLft': ((0.0, 0.0, L_CUBE, 0.0, L_CUBE, L_CUBE), (0, 2)), # Fixed X=0, Z=L. Running Y (axis 1).
    'bckRgt': ((L_CUBE, 0.0, L_CUBE, L_CUBE, L_CUBE, L_CUBE), (0, 2)), # Fixed X=L, Z=L. Running Y (axis 1).
    
    'botLft': ((0.0, 0.0, 0.0, 0.0, 0.0, L_CUBE), (0, 1)), # Fixed X=0, Y=0. Running Z (axis 2).
    'botRgt': ((L_CUBE, 0.0, 0.0, L_CUBE, 0.0, L_CUBE), (0, 1)), # Fixed X=L, Y=0. Running Z (axis 2).
    'topLft': ((0.0, L_CUBE, 0.0, 0.0, L_CUBE, L_CUBE), (0, 1)), # Fixed X=0, Y=L. Running Z (axis 2).
    'topRgt': ((L_CUBE, L_CUBE, 0.0, L_CUBE, L_CUBE, L_CUBE), (0, 1)), # Fixed X=L, Y=L. Running Z (axis 2).
}

for name, (bbox_target, fixed_axes) in edge_entities_map.items():
    fixed_axis_1 = fixed_axes[0]
    fixed_axis_2 = fixed_axes[1]
    
    # 1. Determine the fixed values for the two axes (0 or L_CUBE)
    val_1 = 0.0 if math.isclose(bbox_target[fixed_axis_1], 0.0, abs_tol=TOL) else L_CUBE
    val_2 = 0.0 if math.isclose(bbox_target[fixed_axis_2], 0.0, abs_tol=TOL) else L_CUBE

    # 2. Find all nodes that lie on the intersection of the two fixed planes
    
    # Find nodes close to the first fixed plane
    indices_1 = np.where(np.isclose(coord[:, fixed_axis_1], val_1, atol=TOL))[0]
    
    # Find nodes close to the second fixed plane
    indices_2 = np.where(np.isclose(coord[:, fixed_axis_2], val_2, atol=TOL))[0]
    
    # The edge nodes are the intersection of these two sets of indices
    edge_node_indices = np.intersect1d(indices_1, indices_2)
    edge_node_tags = nodeTags[edge_node_indices]
    
    # 3. Sort and exclude corners using the modified helper function
    node_sets[name] = get_sorted_open_edge_nodes_by_coords(edge_node_tags, fixed_axes)

# ----------------------------------------------------------------------
# --- STEP 3: IDENTIFY AND SORT OPEN FACE NODES ---
# ----------------------------------------------------------------------

face_tags_map = {
    'lft': left_surface_tags, 'rgt': right_surface_tags,
    'bck': back_surface_tags, 'fro': front_surface_tags,
    'bot': bottom_surface_tags, 'top': top_surface_tags,
}

face_edges_map = {
    'fro': ['froBot', 'froTop', 'froLft', 'froRgt'], 'bck': ['bckBot', 'bckTop', 'bckLft', 'bckRgt'],
    'lft': ['froLft', 'bckLft', 'botLft', 'topLft'], 'rgt': ['froRgt', 'bckRgt', 'botRgt', 'topRgt'],
    'bot': ['froBot', 'bckBot', 'botLft', 'botRgt'], 'top': ['froTop', 'bckTop', 'topLft', 'topRgt'],
}

# Map face names to their fixed coordinate and value
face_coord_map = {
    'lft': (0, 0.0),      # X=0
    'rgt': (0, L_CUBE),   # X=L_CUBE
    'top': (1, L_CUBE),   # Y=L_CUBE
    'bot': (1, 0.0),      # Y=0.0
    'fro': (2, 0.0),      # Z=0
    'bck': (2, L_CUBE),   # Z=L_CUBE
}

# The face_tags_map and face_edges_map remain the same as before

for face_name in face_coord_map.keys():
    # 1. Get ALL nodes on the surface by checking coordinates
    axis, fixed_value = face_coord_map[face_name]
    
    # Find indices where the coordinate is close to the fixed value
    fixed_indices = np.where(np.isclose(coord[:, axis], fixed_value, atol=TOL))[0]
    
    # Get the node tags for all nodes on this surface
    face_node_tags_all = nodeTags[fixed_indices]
    
    # 2. Collect all bounding nodes (corners + open edges)
    # The face tag lists (tags) are no longer needed here, but the name is required for the edge map.
    bounding_nodes = set(all_corner_nodes)
    for edge_name in face_edges_map[face_name]:
        bounding_nodes.update(node_sets[edge_name])

    # 3. Open face nodes = All nodes - Bounding nodes
    open_face_nodes = np.array(list(set(face_node_tags_all) - bounding_nodes), dtype=np.intp)

    # 4. Sort (by non-fixed coordinates)
    fixed_axis = axis
    running_axes = [i for i in range(3) if i != fixed_axis]
    
    face_node_indices = np.searchsorted(nodeTags, open_face_nodes)
    open_face_coords = coord[face_node_indices]
    
    # Sort by the first running axis, then the second
    sort_indices = np.lexsort((open_face_coords[:, running_axes[1]], open_face_coords[:, running_axes[0]]))
    
    node_sets[face_name] = open_face_nodes[sort_indices]


# ----------------------------------------------------------------------
# --- FINAL RESULT: PYTHON DICTIONARY ---
# ----------------------------------------------------------------------

final_node_sets = {
    # Open Face Nodes
    'face_fro': node_sets['fro'], 'face_bck': node_sets['bck'], 
    'face_lft': node_sets['lft'], 'face_rgt': node_sets['rgt'], 
    'face_bot': node_sets['bot'], 'face_top': node_sets['top'],
    
    # Open Edge Nodes
    'edge_froBot': node_sets['froBot'], 'edge_froTop': node_sets['froTop'], 
    'edge_bckBot': node_sets['bckBot'], 'edge_bckTop': node_sets['bckTop'],
    'edge_froLft': node_sets['froLft'], 'edge_froRgt': node_sets['froRgt'], 
    'edge_bckLft': node_sets['bckLft'], 'edge_bckRgt': node_sets['bckRgt'],
    'edge_botLft': node_sets['botLft'], 'edge_botRgt': node_sets['botRgt'], 
    'edge_topLft': node_sets['topLft'], 'edge_topRgt': node_sets['topRgt'],
    
    # Corner Nodes (single integer indices)
    'corner_froBotLft': node_sets['corner_FBL'],
    'corner_froBotRgt': node_sets['corner_FBR'],
    'corner_bckBotRgt': node_sets['corner_BBR'],
    'corner_bckBotLft': node_sets['corner_BBL'],
    'corner_froTopLft': node_sets['corner_FTL'],
    'corner_froTopRgt': node_sets['corner_FTR'],
    'corner_bckTopRgt': node_sets['corner_BTR'],
    'corner_bckTopLft': node_sets['corner_BTL'],
}

json_node_sets = {}
for key, value in final_node_sets.items():
    if isinstance(value, np.ndarray):
        # Convert NumPy arrays of tags (faces/edges) to lists
        json_node_sets[key] = value.tolist()
    else:
        # Assumes single corner node tags are standard Python integers
        json_node_sets[key] = int(value) 

# --- ELEMENT CLASSIFICATION (PHASE DEFINITION) ---

# Get all 3D element tags and types
element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
if not element_tags:
    raise Exception("Mesh generation failed. No 3D elements found.")

element_type = element_types[0]
node_tags_list, node_coords_list, _ = gmsh.model.mesh.getNodesByElementType(element_type)

# Classify each element based on its geometric center
matrix_elem_tags = []
interface_elem_tags = []
particle_elem_tags = []

num_nodes_per_element = 8
num_elements = len(element_tags[0])

for e in range(num_elements):
    # Extract the nodes for the current element
    start_idx = e * num_nodes_per_element
    current_node_tags = node_tags_list[start_idx : start_idx + num_nodes_per_element]

    # Get the coordinates of these nodes
    all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
    # Make a dictionary: tag → (x, y, z)
    node_dict = {tag: all_coords[3*e:3*e+3] for e, tag in enumerate(all_tags)}

    # Now just look up your subset
    com_x = 0
    com_y = 0
    com_z = 0
    for tag in current_node_tags:
        x, y, z = node_dict[tag]
        com_x += x
        com_y += y
        com_z += z

    # Calculate Element Center of Mass (COM)
    com_x /= num_nodes_per_element
    com_y /= num_nodes_per_element
    com_z /= num_nodes_per_element
        
    # NEW: WRAP COM back into the [0, L_CUBE) domain for periodic checking
    # Note: Using fmod is robust for wrapping coordinates
    com_x_wrapped = math.fmod(com_x, L_CUBE)
    com_y_wrapped = math.fmod(com_y, L_CUBE)
    com_z_wrapped = math.fmod(com_z, L_CUBE)

    # Handle negative wrap (e.g., -1.0 becomes 9.0 for L_CUBE=10.0)
    if com_x_wrapped < 0: com_x_wrapped += L_CUBE
    if com_y_wrapped < 0: com_y_wrapped += L_CUBE
    if com_z_wrapped < 0: com_z_wrapped += L_CUBE

    is_particle_element = False
    is_interface_element = False

    # 1. Iterate over all placed particles
    for x_start, y_start, z_start, x_end, y_end, z_end in particle_bboxes:
        
        # 2. Check all 27 periodic images for THIS particle
        is_particle_for_this_bbox = False
        is_interface_for_this_bbox = False

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    
                    shift_x = i * L_CUBE
                    shift_y = j * L_CUBE
                    shift_z = k * L_CUBE
                    
                    x_shifted_start = x_start + shift_x
                    y_shifted_start = y_start + shift_y
                    z_shifted_start = z_start + shift_z
                    x_shifted_end = x_end + shift_x
                    y_shifted_end = y_end + shift_y
                    z_shifted_end = z_end + shift_z

                    # A. PERIODIC PARTICLE CORE CHECK (Highest Precedence)
                    if (x_shifted_start - 1e-9 <= com_x < x_shifted_end + 1e-9 and
                        y_shifted_start - 1e-9 <= com_y < y_shifted_end + 1e-9 and
                        z_shifted_start - 1e-9 <= com_z < z_shifted_end + 1e-9):
                        
                        is_particle_for_this_bbox = True
                        # If it's a particle core, no need to check other images for THIS particle
                        break 
                    
                    # B. PERIODIC INTERFACE CHECK (Only if A is False)
                    if (x_shifted_start - t_interface <= com_x <= x_shifted_end + t_interface and
                        y_shifted_start - t_interface <= com_y <= y_shifted_end + t_interface and
                        z_shifted_start - t_interface <= com_z <= z_shifted_end + t_interface):
                        
                        # Ensure it's not the particle core itself (already checked above, but good practice)
                        # The check in A already ensures the COM isn't in the core.
                        is_interface_for_this_bbox = True
                        # Do NOT break, as a subsequent image check (A) might reveal it IS a core.
                
                if is_particle_for_this_bbox: break
            if is_particle_for_this_bbox: break
        
        # Update global flags based on checks for this particle (since particles overlap)
        if is_particle_for_this_bbox:
            is_particle_element = True
            
        if is_interface_for_this_bbox:
            is_interface_element = True # Element is interface if any particle's interface overlaps it

    # 3. Final Classification (Ensuring Particle Core > Interface > Matrix)
    if is_particle_element:
        particle_elem_tags.append(element_tags[0][e])
    elif is_interface_element:
        # This element is only interface if it wasn't already classified as core by any overlapping particle.
        interface_elem_tags.append(element_tags[0][e])
    else:
        matrix_elem_tags.append(element_tags[0][e])

# --- CREATE PHYSICAL GROUPS (BASED ON MESH ELEMENTS) ---
matrix_elem_tags = np.array(matrix_elem_tags, dtype=np.uint64) 
interface_elem_tags = np.array(interface_elem_tags, dtype=np.uint64) 
particle_elem_tags = np.array(particle_elem_tags, dtype=np.uint64)

# ----------------------------------------------------------------------
# 2. Define the Complete Data Structure
# ----------------------------------------------------------------------

data_to_dump = {
    "element_sets": {
        "matrix": matrix_elem_tags.tolist(),
        "interface": interface_elem_tags.tolist(),
        "particle": particle_elem_tags.tolist()
    },
    "node_sets": json_node_sets
}

try:
    with open("i.geo_element_sets.json", "w") as f:
        json.dump(data_to_dump, f, indent=4) # 4-space indentation for readability
    print("Successfully wrote element sets and periodic node sets to i.geo_element_sets.json")
except Exception as e:
    print(f"Error writing JSON file: {e}")

gmsh.write("i.geo_composite_cube.msh2")

# Launch the gmsh GUI to visualize the geometry and physical groups
gmsh.fltk.run()

gmsh.finalize()