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
Lx = 30.0
Ly = 6.0
r_kink = 0.5
r_particle = 0.3
nr_particles = 1
Lx_particle_min = 2.5
Lx_particle_max = 3.0
res = 0.5  # Global desired mesh size
res_coh = 0.15
fine_cl_zones = 0.15 # Renamed variable for clarity
coarse_cl_zones = res
dist_transition = 0.5

# --- Mesh size globally ---
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

# --- Geometry Creation (all as distinct entities) ---
# Points for upper block (from Y=0 to Y=Ly)
p1 = gmsh.model.occ.addPoint(0, 0, 0, res)
p2 = gmsh.model.occ.addPoint(Lx, 0, 0, res)
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

gmsh.model.occ.synchronize() # Synchronize before adding particles to ensure main surfaces exist

# --- Particle Generation (Collecting all particles to cut later) ---
particle_disk_tags = [] # Stores only the integer tags of the disks
original_disk_info = [] # Stores {'center_x', 'center_y', 'radius'} for identifying fragmented particles

# --- Kink Particle ---
kink_disk_dim_tags = []
kink_disk_cut_tag = gmsh.model.occ.addDisk(0, 0, 0, 3*r_kink, 0.5*r_kink)
kink_disk_dim_tag = (2, kink_disk_cut_tag)
kink_disk_dim_tags.append(kink_disk_dim_tag)

# --- Perform Cutting Fragmentation ---
objects_for_kink_cut = [(2, s1_original_tag), (2, s2_original_tag)]
tools_for_kink_cut = [(2, kink_disk_cut_tag)]

modmain_surf_dim_tags, _ = gmsh.model.occ.cut(
    objects_for_kink_cut,
    tools_for_kink_cut,
    removeObject=True, 
    removeTool=True    
)

s1_tag = modmain_surf_dim_tags[0][1]
s2_tag = modmain_surf_dim_tags[1][1]

gmsh.model.occ.synchronize() 

original_disk_info = []

# Particle Disks along x-axis
for i in range(nr_particles):
    x_pos = Lx_particle_min + i*(Lx_particle_max - Lx_particle_min)/nr_particles
    particle_disk_tag = gmsh.model.occ.addDisk(x_pos, 0.0, 0, r_particle, r_particle)
    original_disk_info.append({'center_x': x_pos, 'center_y': 0.0, 'radius': r_particle})
    particle_disk_tags.append(particle_disk_tag)

gmsh.model.occ.synchronize()

all_domain_dim_tags = [(2, s1_original_tag), (2, s2_original_tag)]
all_particle_tools_dim_tags = [(2, tag) for tag in particle_disk_tags]

gmsh.model.occ.fragment(
    all_domain_dim_tags,
    all_particle_tools_dim_tags
)

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize() 

# --- Classify Final Surfaces for Physical Groups ---
fragmented_upper_domain_surfaces = []
fragmented_lower_domain_surfaces = []
fragmented_upper_particle_surfaces = []
fragmented_lower_particle_surfaces = []

all_final_surfaces = gmsh.model.getEntities(2) 

for dim, tag in all_final_surfaces:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    is_particle_surface = False
    # Check if this fragment's center is within any of the *original* particle disk locations
    for disk_info in original_disk_info:
        if is_point_inside_circle(center_x, center_y, disk_info['center_x'], disk_info['center_y'], disk_info['radius'] * 1.01): # Small buffer for robustness
            if center_y > 1e-9:
                fragmented_upper_particle_surfaces.append(tag)
                is_particle_surface = True
            elif center_y < 1e-9:
                fragmented_lower_particle_surfaces.append(tag)
                is_particle_surface = True                
            break

    if not is_particle_surface:
        if center_y > 1e-9: # Y > 0 for upper domain
            fragmented_upper_domain_surfaces.append(tag)
        elif center_y < -1e-9: # Y < 0 for lower domain
            fragmented_lower_domain_surfaces.append(tag)

# --- Define Physical Surfaces ---
gmsh.model.addPhysicalGroup(2, list(set(fragmented_upper_domain_surfaces)), name="UpperDomain")
gmsh.model.addPhysicalGroup(2, list(set(fragmented_lower_domain_surfaces)), name="LowerDomain")
gmsh.model.addPhysicalGroup(2, list(set(fragmented_upper_particle_surfaces)), name="UpperParticleDomain")
gmsh.model.addPhysicalGroup(2, list(set(fragmented_lower_particle_surfaces)), name="LowerParticleDomain")

# --- Identify and Classify Final Curves (Lines) ---
cohesive_line_interface_curves = []
cohesive_particle_interface_curves = []
all_final_curves = gmsh.model.getEntities(1)

for dim, curve_tag in all_final_curves:
    adj_entities = gmsh.model.getAdjacencies(1, curve_tag)
    surface_tags_bounding_curve = adj_entities[0]

    has_upper = False
    has_lower = False
    has_particle = False

    for s_tag in surface_tags_bounding_curve:
        if s_tag in fragmented_upper_domain_surfaces:
            has_upper = True
        if s_tag in fragmented_lower_domain_surfaces:
            has_lower = True
        if s_tag in fragmented_upper_particle_surfaces or s_tag in fragmented_lower_particle_surfaces:
            has_particle = True

    y_min_curve, y_max_curve = get_curve_y_coords(curve_tag)
    
    # 1. Cohesive Interface (Matrix-Matrix boundary)
    if has_upper and has_lower and not has_particle and math.isclose(y_min_curve, 0.0, abs_tol=1e-7):
        cohesive_line_interface_curves.append(curve_tag)

    # 2. Cohesive Interface (Matrix-Matrix boundary)
    if not has_upper and not has_lower and math.isclose(y_min_curve, 0.0, abs_tol=1e-7):
        cohesive_line_interface_curves.append(curve_tag)        

    # 3. Cohesive Particle (Matrix-Particle boundary)
    if (has_upper and has_particle) or (has_lower and has_particle):
        cohesive_particle_interface_curves.append(curve_tag)

# --- Define Physical Groups for Curves ---
gmsh.model.addPhysicalGroup(1, list(set(cohesive_line_interface_curves)), name="CohesiveInterface")
gmsh.model.addPhysicalGroup(1, list(set(cohesive_particle_interface_curves)), name="CohesiveParticle")

# --- Identify other boundary lines ---
left_upper_curves_tags = []
left_lower_curves_tags = []
right_line_curves_tags = []
bottom_line_curves_tags = []
top_line_curves_tags = []

for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        
        # Check if the curve is not part of any "cohesive" interface
        is_cohesive_line = c_id in cohesive_line_interface_curves or c_id in cohesive_particle_interface_curves
        if is_cohesive_line:
            continue

        adj_entities = gmsh.model.getAdjacencies(1, c_id)
        bounded_by_s_tags = adj_entities[0]
        
        # Left Upper Line (X=0, Y>0)
        if math.isclose(bbox[0], 0.0, abs_tol=1e-7) and math.isclose(bbox[3], 0.0, abs_tol=1e-7) and bbox[1] > 1e-9 and all(s_tag in fragmented_upper_domain_surfaces for s_tag in bounded_by_s_tags):
            left_upper_curves_tags.append(c_id)
            
        # Left Lower Line (X=0, Y<0)
        elif math.isclose(bbox[0], 0.0, abs_tol=1e-7) and math.isclose(bbox[3], 0.0, abs_tol=1e-7) and bbox[4] < -1e-9 and all(s_tag in fragmented_lower_domain_surfaces for s_tag in bounded_by_s_tags):
            left_lower_curves_tags.append(c_id)
            
        # Right Line (X=Lx)
        elif math.isclose(bbox[0], Lx, abs_tol=1e-6) and math.isclose(bbox[3], Lx, abs_tol=1e-6) and all(s_tag in fragmented_upper_domain_surfaces or s_tag in fragmented_lower_domain_surfaces for s_tag in bounded_by_s_tags):
            right_line_curves_tags.append(c_id)
            
        # Bottom Line (Y=-Ly)
        elif math.isclose(bbox[1], -Ly, abs_tol=1e-6) and math.isclose(bbox[4], -Ly, abs_tol=1e-6) and all(s_tag in fragmented_lower_domain_surfaces for s_tag in bounded_by_s_tags):
            bottom_line_curves_tags.append(c_id)
            
        # Top Line (Y=Ly)
        elif math.isclose(bbox[1], Ly, abs_tol=1e-6) and math.isclose(bbox[4], Ly, abs_tol=1e-6) and all(s_tag in fragmented_upper_domain_surfaces for s_tag in bounded_by_s_tags):
            top_line_curves_tags.append(c_id)

gmsh.model.addPhysicalGroup(1, list(set(left_upper_curves_tags)), name="LeftUpperLine")
gmsh.model.addPhysicalGroup(1, list(set(left_lower_curves_tags)), name="LeftLowerLine")
gmsh.model.addPhysicalGroup(1, list(set(right_line_curves_tags)), name="RightLine")
gmsh.model.addPhysicalGroup(1, list(set(bottom_line_curves_tags)), name="BottomLine")
gmsh.model.addPhysicalGroup(1, list(set(top_line_curves_tags)), name="TopLine")


# --- Mesh Refinement Fields ---
# Distance Field for cohesive zones
field2_tag = gmsh.model.mesh.field.add("Distance")
all_cohesive_curves = list(set(cohesive_line_interface_curves + cohesive_particle_interface_curves))
gmsh.model.mesh.field.setNumbers(field2_tag, "CurvesList", all_cohesive_curves)
gmsh.model.mesh.field.setNumber(field2_tag, "Sampling", 100)

# Threshold Field to apply refinement
field3_tag = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field3_tag, "InField", field2_tag)
gmsh.model.mesh.field.setNumber(field3_tag, "LcMin", res_coh)
gmsh.model.mesh.field.setNumber(field3_tag, "LcMax", res)
gmsh.model.mesh.field.setNumber(field3_tag, "DistMin", 0)
gmsh.model.mesh.field.setNumber(field3_tag, "DistMax", dist_transition)

gmsh.model.mesh.field.setAsBackgroundMesh(field3_tag)

# --- Meshing Algorithms ---
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
gmsh.option.setNumber("Mesh.Algorithm", 8)

# Print summary
print(f"Number of 'CohesiveInterface' curves: {len(cohesive_line_interface_curves)}")
print(f"Number of 'CohesiveParticle' curves: {len(cohesive_particle_interface_curves)}")
print(f"Number of 'LeftUpper' curves: {len(left_upper_curves_tags)}")
print(f"Number of 'LeftLower' curves: {len(left_lower_curves_tags)}")
print(f"Number of 'RightLine' curves: {len(right_line_curves_tags)}")
print(f"Number of 'BottomLine' curves: {len(bottom_line_curves_tags)}")
print(f"Number of 'TopLine' curves: {len(top_line_curves_tags)}")

# Generate and save
gmsh.model.mesh.generate(1)
gmsh.model.mesh.generate(2)
gmsh.write("i.geo.msh2")
gmsh.fltk.run()
gmsh.finalize()