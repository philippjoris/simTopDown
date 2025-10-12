"""
file: geom.py
author: Philipp van der Loos

A file to create a gmsh geometry. The geometry is a cohesive line with particles on it 
based on the parameter input.

____________________________________________________
|                                                   |
|                                                   |
|- - - -O- - -O- - -O- - -O- - -O- - -O- - -O- - - -|
|                                                   |
|                                                   |
|___________________________________________________|


"""

import gmsh
import math

# Helper function to get Y-coordinates of curve's start/end points
def get_curve_y_coords(curve_tag):
    point_dim_tags = gmsh.model.getBoundary([(1, curve_tag)], False, True)
    y_coords = []
    for p_dim, p_tag in point_dim_tags:
        coords = gmsh.model.getValue(p_dim, p_tag, []) # Corrected: Added [] for parametricCoord
        y_coords.append(coords[1]) # Index 1 for Y-coordinate
    return min(y_coords), max(y_coords)

# Helper function to get bounding box coordinates (xmin, ymin, zmin, xmax, ymax, zmax)
def get_bbox_coords(dim, tag):
    return gmsh.model.getBoundingBox(dim, tag)

# Helper function to check if an element is in a list (for clarity in boundary identification)
def is_in_list(item, list_of_items):
    return item in list_of_items

# Helper to check if a point (x,y) is inside a circle (cx, cy, r)
def is_point_inside_circle(px, py, cx, cy, r):
    return (px - cx)**2 + (py - cy)**2 < r**2

# 1. Initialize Gmsh
gmsh.initialize()
gmsh.model.add("multi_material_cohesive")

# Set the OpenCASCADE geometry kernel
gmsh.model.occ.addPoint(0, 0, 0, 0) # Dummy point to ensure OCC is loaded early

# --- Parameters ---
Lx = 10.0
Ly = 3.0
r_kink = 0.6
r_particle = 0.5
nr_particles = 2
particle_spacing = 2*r_particle + 0.5 # Define particle spacing for even distribution
res = 0.5  # Global desired mesh size
res_coh = 0.1

# --- Mesh size globally ---
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

# --- Geometry Creation (all as distinct entities) ---
# Points for upper block (from Y=0 to Y=Ly)
p1 = gmsh.model.occ.addPoint(0, -0.07, 0, res)
p2 = gmsh.model.occ.addPoint(Lx, -0.07, 0, res)
p3 = gmsh.model.occ.addPoint(Lx, Ly, 0, res)
p4 = gmsh.model.occ.addPoint(0, Ly, 0, res)

# Lines for upper block
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)

# Surface for upper block
ll1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
s1_original_tag = gmsh.model.occ.addPlaneSurface([ll1]) # Renamed for clarity

# Points for lower block (from Y=-Ly to Y=0)
p_lower_left = gmsh.model.occ.addPoint(0, -Ly, 0, res)
p_lower_right = gmsh.model.occ.addPoint(Lx, -Ly, 0, res)

# Lines for lower block
l_lower_left = gmsh.model.occ.addLine(p1, p_lower_left)
l_lower_bottom = gmsh.model.occ.addLine(p_lower_left, p_lower_right)
l_lower_right = gmsh.model.occ.addLine(p_lower_right, p2)

# Surface for lower block
ll2 = gmsh.model.occ.addCurveLoop([l_lower_left, l_lower_bottom, l_lower_right, -l1])
s2_original_tag = gmsh.model.occ.addPlaneSurface([ll2]) # Renamed for clarity

gmsh.model.occ.synchronize()

# Particle Disks (these will NOT be "holes" but separate material domains)
kink_disk_dim_tags = [] # Store (dim, tag) for all particle surfaces
other_particle_disk_dim_tags = []
# Kink Disk (at origin)
kink_disk_cut_tag = gmsh.model.occ.addDisk(0, 0, 0, r_kink, r_kink)
kink_disk_dim_tag = (2, kink_disk_cut_tag)
kink_disk_dim_tags.append(kink_disk_dim_tag)

# --- Perform Cutting Fragmentation ---
# Gather all entities that might overlap/intersect
objects_for_kink_cut = [(2, s1_original_tag), (2, s2_original_tag)]
tools_for_kink_cut = [(2, kink_disk_cut_tag)]

modified_main_surfaces_dim_tags, _ = gmsh.model.occ.cut(
    objects_for_kink_cut,
    tools_for_kink_cut,
    removeObject=True, 
    removeTool=True    
)

s1_post_kink_cut_tag = modified_main_surfaces_dim_tags[0][1]
s2_post_kink_cut_tag = modified_main_surfaces_dim_tags[1][1]
gmsh.model.occ.synchronize() 

# Dictionaries to store the new fragmented entity tags for physical groups
domain_surfaces = []
# Other boundary curves (Left, Right, Top, Bottom) need careful re-identification too.

# Cache original disk centers for identification
particle_disk_tags = []
original_disk_info = []
# Kink disk
# original_disk_info.append({'center_x': 0, 'center_y': 0, 'radius': r_kink})
# Particle disks
for i in range(nr_particles):
    x_pos = particle_spacing * (i + 1)
    particle_disk_tag = gmsh.model.occ.addDisk(x_pos, -0.0, 0, r_particle, r_particle)
    original_disk_info.append({'center_x': x_pos, 'center_y': +0.0, 'radius': r_particle})
    particle_disk_tags.append(particle_disk_tag)

gmsh.model.occ.synchronize()

all_domain_dim_tags = [(2, s1_post_kink_cut_tag), (2, s2_post_kink_cut_tag)]
all_particle_tools_dim_tags = [(2, tag) for tag in particle_disk_tags]

gmsh.model.occ.cut(
    objectDimTags=all_domain_dim_tags, # Main domains to cut from
    toolDimTags=all_particle_tools_dim_tags, # Particles to cut out
    removeObject=True, # Remove the original s1_original_tag and s2_original_tag
    removeTool=False    # Remove the original particle disks (new ones are created as cut_entities)
)

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

# Iterate through all 2D entities (surfaces) that exist after fragmentation
all_final_surfaces = gmsh.model.getEntities(2)

particle_surfaces = []

for dim, tag in all_final_surfaces:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    is_particle_surface = False
    # Check if this fragment's center is within any of the *original* particle disk locations
    for disk_info in original_disk_info:
        if is_point_inside_circle(center_x, center_y, disk_info['center_x'], disk_info['center_y'], disk_info['radius'] * 1.01): # Small buffer for robustness
            particle_surfaces.append(tag)
            is_particle_surface = True            
            break
        if not is_particle_surface:
            domain_surfaces.append(tag)

# --- Define Physical Surfaces ---
# Ensure tags are unique within each physical group
gmsh.model.addPhysicalGroup(2, list(set(particle_surfaces)), name="ParticleDomain")
gmsh.model.addPhysicalGroup(2, list(set(domain_surfaces)), name="LowerDomain")


# --- Identify Cohesive Zone Curves ---
# Cohesive curves are 1D entities that are boundaries of both a particle surface
# and an upper/lower domain surface.
all_final_curves = gmsh.model.getEntities(1)


# 2. Left Upper Line (X=0, Y>0, external boundary of upper domain)
left_upper_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        # Check if X coordinates are at 0 (left edge) AND it's above Y=0
        if (math.isclose(bbox[0], 0.0, abs_tol=1e-7) and
            math.isclose(bbox[3], 0.0, abs_tol=1e-7) and
            bbox[1] > 1e-9): # ymin > 0 (to distinguish from y=0 interface)
            
            # Use getAdjacencies here!
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]

            left_upper_curves_tags.append(c_id)
gmsh.model.addPhysicalGroup(1, list(set(left_upper_curves_tags)), name="LeftUpperLine")


# 3. Left Lower Line (X=0, Y<0, external boundary of lower domain)
left_lower_curves_tags = []
right_line_curves_tags = []
bottom_line_curves_tags = []
top_line_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        if (math.isclose(bbox[0], 0.0, abs_tol=1e-7) and
            math.isclose(bbox[3], 0.0, abs_tol=1e-7) and
            bbox[4] < -1e-9): # ymax < 0 (to distinguish from y=0 interface)            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]
            left_lower_curves_tags.append(c_id)
        elif (math.isclose(bbox[0], Lx, abs_tol=1e-6) and
            math.isclose(bbox[3], Lx, abs_tol=1e-6)): # xmin and xmax are Lx
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]            
            right_line_curves_tags.append(c_id)
        elif (math.isclose(bbox[1], -Ly, abs_tol=1e-6) and
            math.isclose(bbox[4], -Ly, abs_tol=1e-6)): # ymin and ymax are -Ly
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]
            bottom_line_curves_tags.append(c_id)
        elif (math.isclose(bbox[1], Ly, abs_tol=1e-6) and
            math.isclose(bbox[4], Ly, abs_tol=1e-6)):            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]
            top_line_curves_tags.append(c_id)

gmsh.model.addPhysicalGroup(1, list(set(left_lower_curves_tags)), name="LeftLowerLine")
gmsh.model.addPhysicalGroup(1, list(set(right_line_curves_tags)), name="RightLine")
gmsh.model.addPhysicalGroup(1, list(set(bottom_line_curves_tags)), name="BottomLine")
gmsh.model.addPhysicalGroup(1, list(set(top_line_curves_tags)), name="TopLine")

# --- Mesh Refinement Fields ---

# Distance Field (Field 2) for cohesive zones
field2_tag = gmsh.model.mesh.field.add("Distance")

# Threshold Field (Field 3) for cohesive zones
# field3_tag = gmsh.model.mesh.field.add("Threshold")
# gmsh.model.mesh.field.setNumber(field3_tag, "InField", field2_tag)
# gmsh.model.mesh.field.setNumber(field3_tag, "LcMin", fine_cl_zones)
# gmsh.model.mesh.field.setNumber(field3_tag, "LcMax", coarse_cl_zones)
# gmsh.model.mesh.field.setNumber(field3_tag, "DistMin", 0)
# gmsh.model.mesh.field.setNumber(field3_tag, "DistMax", dist_transition)

# Box Field (Field 1) - Still refining around Y=0 main interface
field1_tag = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field1_tag, "VIn", res_coh) # Finer mesh size within the box
gmsh.model.mesh.field.setNumber(field1_tag, "VOut", res) # Coarser mesh size outside the box
field1_xmin, field1_xmax = 0, Lx
field1_ymin, field1_ymax = -2.3, 2.3 # Box around Y=0 line
gmsh.model.mesh.field.setNumber(field1_tag, "XMin", field1_xmin)
gmsh.model.mesh.field.setNumber(field1_tag, "XMax", field1_xmax)
gmsh.model.mesh.field.setNumber(field1_tag, "YMin", field1_ymin)
gmsh.model.mesh.field.setNumber(field1_tag, "YMax", field1_ymax)


# Combine fields (Min Field 4)
field4_tag = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(field4_tag, "FieldsList", [field1_tag])

gmsh.model.mesh.field.setAsBackgroundMesh(field4_tag)

# --- Meshing Algorithms ---
gmsh.option.setNumber("Mesh.RecombineAll", 1) # Try to create quadrilateral elements
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # Select an algorithm for recombination (e.g., Anisotropic)
gmsh.option.setNumber("Mesh.Algorithm", 8) # Frontal-Delaunay for 2D (often good for complex geometries)

print(f"Number of 'LeftUpper' curves: {len(left_upper_curves_tags)}")
print(f"Number of 'LeftLower' curves: {len(left_lower_curves_tags)}")
print(f"Number of 'RightLine' curves: {len(right_line_curves_tags)}")
print(f"Number of 'BottomLine' curves: {len(bottom_line_curves_tags)}")
print(f"Number of 'TopLine' curves: {len(top_line_curves_tags)}")

# Generate 1D elements
gmsh.model.mesh.generate(1)
# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Optional: Save the mesh
gmsh.write("i.geo_w_particle.msh2")

# Optional: Launch Gmsh GUI to visualize the mesh
gmsh.fltk.run()

# 4. Finalize Gmsh
gmsh.finalize()