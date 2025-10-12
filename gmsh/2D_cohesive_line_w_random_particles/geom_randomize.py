"""
file: geom.py
author: Philipp van der Loos

A file to create a gmsh geometry. The geometry is a cohesive line with randomly distributed
particles in the matrix. Particles may be on the cohesive line or not. A Probability density
function is added so that particles are more likely to spawn closer to the interface line.
Based on the particle/matrix ratio and the radius of the particles the script determines
the amount of particles.

____________________________________________________
|       O                  O                        |
|                    O              O       O       |
|- - - - -O- -O- - - - -O- - - -O- - -O- - - - - - -|
|  O             O                       O          |
|       O                                           |
|_____________________________________O_____________|

"""

import gmsh
import math
import random
import numpy as np

# Helper function to get Y-coordinates of curve's start/end points
def get_curve_y_coords(curve_tag):
    point_dim_tags = gmsh.model.getBoundary([(1, curve_tag)], False, True)
    y_coords = []
    for p_dim, p_tag in point_dim_tags:
        coords = gmsh.model.getValue(p_dim, p_tag, [])
        y_coords.append(coords[1])
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

# New helper to check for overlap between two circles (with a small buffer)
def do_circles_overlap(c1x, c1y, r1, c2x, c2y, r2, buffer=1e-6): # Added a small buffer
    distance_sq = (c1x - c2x)**2 + (c1y - c2y)**2
    min_dist_sq = (r1 + r2 + buffer)**2
    return distance_sq < min_dist_sq

# 1. Initialize Gmsh
gmsh.initialize()
gmsh.model.add("multi_material_cohesive_pdf_particles_revised")

# Set the OpenCASCADE geometry kernel
gmsh.model.occ.addPoint(0, 0, 0, 0) # Dummy point to ensure OCC is loaded early

# --- Parameters ---
Lx = 30.0
Ly = 6.0
r_kink = 0.3
r_particle = 0.4
buffer_distance = 0.1

desired_area_ratio = 0.005 # e.g., 15% of the total domain area should be particles (Adjust as needed)

particle_y_dist_sigma = Ly / 100.0 # Significantly increased for better particle distribution

res = 0.5 # Global desired mesh size
res_coh = 0.15 # Fine mesh size in cohesive zone (Y=0 interface)
fine_cl_zones = 0.15 # Mesh size for the particle interfaces (set to same as res_coh for consistency)
coarse_cl_zones = res # Mesh size outside fine zones
dist_transition = 0.5 # Distance over which mesh size transitions

# Parameters for random particle generation
max_attempts_per_particle = 200 # How many times to try placing a single particle before giving up

# Particle centers range for x and y
# Ensure enough space for particle radius + mesh resolution
particle_min_x_center = 0 + r_particle + res
particle_max_x_center = Lx - r_particle - res # Use Lx for full width
particle_min_y_center = -Ly + r_particle + res
particle_max_y_center = Ly - r_particle - res

# --- Calculate Number of Particles based on Area Ratio ---
total_domain_area = Lx * (2 * Ly) # Total area of the simulation domain
area_of_kink = math.pi * r_kink**2
area_of_random_particle = math.pi * r_particle**2

target_total_particle_area = desired_area_ratio * total_domain_area

# Calculate remaining area to be filled by random particles (after accounting for the kink)
remaining_area_for_random_particles = target_total_particle_area - area_of_kink

num_random_particles_to_place = 0
if remaining_area_for_random_particles > 0:
    num_random_particles_to_place = math.floor(remaining_area_for_random_particles / area_of_random_particle)
    num_random_particles_to_place = max(0, num_random_particles_to_place)

print(f"Desired area ratio: {desired_area_ratio}")
print(f"Total domain area: {total_domain_area:.2f}")
print(f"Target total particle area: {target_total_particle_area:.2f}")
print(f"Area of fixed kink particle: {area_of_kink:.2f}")
print(f"Area of each random particle: {area_of_random_particle:.2f}")
print(f"Calculated number of random particles to try and place: {num_random_particles_to_place}")

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
particle_disk_tags_to_cut_with = [] # Stores only the integer tags of the disks
original_disk_info = [] # Stores {'center_x', 'center_y', 'radius'} for identifying fragmented particles

# --- Kink Particle ---
kink_disk_dim_tags = []
kink_disk_cut_tag = gmsh.model.occ.addDisk(0, 0, 0, r_kink, r_kink)
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

# --- Random Particles ---
placed_random_particles_count = 0
total_attempts = 0

while placed_random_particles_count < num_random_particles_to_place and total_attempts < num_random_particles_to_place * max_attempts_per_particle:
    x_pos = random.uniform(particle_min_x_center, particle_max_x_center)

    y_pos = None
    attempt_y_gen = 0
    while y_pos is None:
        y_candidate = random.gauss(0, particle_y_dist_sigma)
        if particle_min_y_center <= y_candidate <= particle_max_y_center:
            y_pos = y_candidate
        attempt_y_gen += 1
        if attempt_y_gen > 1000:
            print(f"Warning: Could not find suitable y_pos after {attempt_y_gen} attempts for a particle. Adjust particle_y_dist_sigma or domain size.")
            break

    if y_pos is None:
        total_attempts += 1
        continue

    is_valid_position = True
    for existing_disk in original_disk_info:
        if do_circles_overlap(x_pos, y_pos, r_particle + buffer_distance,
                              existing_disk['center_x'], existing_disk['center_y'], existing_disk['radius']):
            is_valid_position = False
            break

    if is_valid_position:
        new_particle_tag = gmsh.model.occ.addDisk(x_pos, y_pos, 0, r_particle, r_particle)
        particle_disk_tags_to_cut_with.append(new_particle_tag)
        original_disk_info.append({'center_x': x_pos, 'center_y': y_pos, 'radius': r_particle})
        placed_random_particles_count += 1

    total_attempts += 1

print(f"\n--- Particle Placement Summary ---")
actual_total_particles_count = len(original_disk_info)
actual_total_particle_area = sum(math.pi * info['radius']**2 for info in original_disk_info)
actual_area_ratio = actual_total_particle_area / total_domain_area

print(f"Total particles attempted (1 kink + {num_random_particles_to_place} random): {1 + num_random_particles_to_place}")
print(f"Total particles successfully placed: {actual_total_particles_count}")
print(f"Achieved area ratio: {actual_area_ratio:.4f} (Target: {desired_area_ratio:.4f})")

if actual_total_particles_count < (1 + num_random_particles_to_place):
    print(f"Warning: Could not place all target particles. This often happens at higher densities due to packing limitations or hitting max attempts.")

gmsh.model.occ.synchronize() # Synchronize all particle disks before cutting

# --- Perform Boolean Cut Operation ---
# Subtract all particle disks from the main domain surfaces.
# The result will be new matrix surfaces with holes, and the particle disks as separate surfaces.

# Convert particle_disk_tags_to_cut_with to (dim, tag) format for gmsh.model.occ.cut
all_particle_tools_dim_tags = [(2, tag) for tag in particle_disk_tags_to_cut_with]

# Perform the cut operation
# The output `new_entities` will contain the new fragmented main domain surfaces
# The `cut_entities` will contain the particle surfaces themselves.
new_entities, cut_entities = gmsh.model.occ.cut(
    objectDimTags=[(2, s1_tag), (2, s2_tag)], # Main domains to cut from
    toolDimTags=all_particle_tools_dim_tags, # Particles to cut out
    removeObject=True, # Remove the original s1_original_tag and s2_original_tag
    removeTool=False    # Remove the original particle disks (new ones are created as cut_entities)
)

gmsh.model.occ.removeAllDuplicates() # Essential after boolean ops to merge coincident entities
gmsh.model.occ.synchronize() # Crucial for getting the final, consistent geometry into Gmsh's model

# --- Classify Final Surfaces for Physical Groups ---
fragmented_upper_domain_surfaces = []
fragmented_lower_domain_surfaces = []
fragmented_particle_surfaces = []

all_final_surfaces = gmsh.model.getEntities(2) # Get all 2D entities after cutting and synchronization

for dim, tag in all_final_surfaces:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    is_particle_surface = False
    # Check if this fragment's center is within any of the *original* particle disk locations
    for disk_info in original_disk_info:
        if is_point_inside_circle(center_x, center_y, disk_info['center_x'], disk_info['center_y'], disk_info['radius'] * 1.01): # Small buffer for robustness
            fragmented_particle_surfaces.append(tag)
            is_particle_surface = True
            break

    if not is_particle_surface:
        # Classify as upper or lower domain based on approximate Y-position
        # Use more robust check for small y-values if particles are near Y=0
        if center_y > 1e-9: # Y > 0 for upper domain
            fragmented_upper_domain_surfaces.append(tag)
        elif center_y < -1e-9: # Y < 0 for lower domain
            fragmented_lower_domain_surfaces.append(tag)

# --- Define Physical Surfaces ---
gmsh.model.addPhysicalGroup(2, list(set(fragmented_upper_domain_surfaces)), name="UpperDomain")
gmsh.model.addPhysicalGroup(2, list(set(fragmented_lower_domain_surfaces)), name="LowerDomain")
gmsh.model.addPhysicalGroup(2, list(set(fragmented_particle_surfaces)), name="ParticleDomain")

# --- Identify Cohesive Zone Curves (Particle-Matrix Interfaces) ---
cohesive_zone_curves = []
all_final_curves = gmsh.model.getEntities(1) # Get all 1D entities after all operations

for dim, curve_tag in all_final_curves:
    adj_entities = gmsh.model.getAdjacencies(1, curve_tag)
    surface_tags_bounding_curve = adj_entities[0]

    has_particle_neighbor = False
    has_main_domain_neighbor = False

    for s_tag in surface_tags_bounding_curve:
        if s_tag in fragmented_particle_surfaces:
            has_particle_neighbor = True
        elif s_tag in fragmented_upper_domain_surfaces or s_tag in fragmented_lower_domain_surfaces:
            has_main_domain_neighbor = True

        if has_particle_neighbor and has_main_domain_neighbor:
            cohesive_zone_curves.append(curve_tag)
            break

gmsh.model.addPhysicalGroup(1, list(set(cohesive_zone_curves)), name="CohesiveParticle")


# --- Identify other boundary lines ---

# 1. Cohesive Line (shared boundary between original s1 and s2, at Y=0, *not* occupied by particles)
cohesive_line_original_interface_curves = []
for dim, curve_tag in all_final_curves:
    if dim == 1:
        y_min_curve, y_max_curve = get_curve_y_coords(curve_tag)

        # Check if the curve is essentially flat at Y=0
        if math.isclose(y_min_curve, 0.0, abs_tol=1e-7) and math.isclose(y_max_curve, 0.0, abs_tol=1e-7):
            adj_entities = gmsh.model.getAdjacencies(1, curve_tag)
            surface_tags_bounding_curve = adj_entities[0]

            has_upper = False
            has_lower = False
            is_cohesive_with_particle = False # This flag ensures it's NOT a particle interface

            for s_tag in surface_tags_bounding_curve:
                if s_tag in fragmented_upper_domain_surfaces:
                    has_upper = True
                if s_tag in fragmented_lower_domain_surfaces:
                    has_lower = True
                if s_tag in fragmented_particle_surfaces: # If it borders a particle, it's not the main interface
                    is_cohesive_with_particle = True
                    break

            if has_upper and has_lower and not is_cohesive_with_particle:
                cohesive_line_original_interface_curves.append(curve_tag)

gmsh.model.addPhysicalGroup(1, list(set(cohesive_line_original_interface_curves)), name="CohesiveInterface")


# 2. Left Upper Line (X=0, Y>0, external boundary of upper domain)
left_upper_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        if (math.isclose(bbox[0], 0.0, abs_tol=1e-7) and
            math.isclose(bbox[3], 0.0, abs_tol=1e-7) and
            bbox[1] > 1e-9): # ymin > 0 (to distinguish from y=0 interface)
            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]

            is_solely_upper_boundary = len(bounded_by_s_tags) > 0 and \
                                       all(s_tag in fragmented_upper_domain_surfaces for s_tag in bounded_by_s_tags)
            
            if is_solely_upper_boundary and not is_in_list(c_id, cohesive_zone_curves) and not is_in_list(c_id, cohesive_line_original_interface_curves):
                left_upper_curves_tags.append(c_id)
gmsh.model.addPhysicalGroup(1, list(set(left_upper_curves_tags)), name="LeftUpperLine")


# 3. Left Lower Line (X=0, Y<0, external boundary of lower domain)
left_lower_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        if (math.isclose(bbox[0], 0.0, abs_tol=1e-7) and
            math.isclose(bbox[3], 0.0, abs_tol=1e-7) and
            bbox[4] < -1e-9): # ymax < 0 (to distinguish from y=0 interface)
            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]
            
            is_solely_lower_boundary = len(bounded_by_s_tags) > 0 and \
                                       all(s_tag in fragmented_lower_domain_surfaces for s_tag in bounded_by_s_tags)

            if is_solely_lower_boundary and not is_in_list(c_id, cohesive_zone_curves) and not is_in_list(c_id, cohesive_line_original_interface_curves):
                left_lower_curves_tags.append(c_id)
gmsh.model.addPhysicalGroup(1, list(set(left_lower_curves_tags)), name="LeftLowerLine")


# 4. Right Line (X=Lx, external boundary of both domains)
right_line_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        if (math.isclose(bbox[0], Lx, abs_tol=1e-6) and
            math.isclose(bbox[3], Lx, abs_tol=1e-6)): # xmin and xmax are Lx
            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]
            
            is_main_domain_boundary = all(s_tag in fragmented_upper_domain_surfaces or s_tag in fragmented_lower_domain_surfaces for s_tag in bounded_by_s_tags)

            if is_main_domain_boundary and not is_in_list(c_id, cohesive_zone_curves) and not is_in_list(c_id, cohesive_line_original_interface_curves):
                right_line_curves_tags.append(c_id)

gmsh.model.addPhysicalGroup(1, list(set(right_line_curves_tags)), name="RightLine")


# 5. Bottom Line (Y=-Ly, external boundary of lower domain)
bottom_line_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        if (math.isclose(bbox[1], -Ly, abs_tol=1e-6) and
            math.isclose(bbox[4], -Ly, abs_tol=1e-6)): # ymin and ymax are -Ly
            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]

            is_lower_domain_boundary = all(s_tag in fragmented_lower_domain_surfaces for s_tag in bounded_by_s_tags)
            
            if is_lower_domain_boundary and not is_in_list(c_id, cohesive_zone_curves) and not is_in_list(c_id, cohesive_line_original_interface_curves):
                bottom_line_curves_tags.append(c_id)
gmsh.model.addPhysicalGroup(1, list(set(bottom_line_curves_tags)), name="BottomLine")


# 6. Top Line (Y=Ly, external boundary of upper domain)
top_line_curves_tags = []
for dim, c_id in all_final_curves:
    if dim == 1:
        bbox = get_bbox_coords(dim, c_id)
        if (math.isclose(bbox[1], Ly, abs_tol=1e-6) and
            math.isclose(bbox[4], Ly, abs_tol=1e-6)):
            
            adj_entities = gmsh.model.getAdjacencies(1, c_id)
            bounded_by_s_tags = adj_entities[0]

            is_upper_domain_boundary = all(s_tag in fragmented_upper_domain_surfaces for s_tag in bounded_by_s_tags)
            
            if is_upper_domain_boundary and not is_in_list(c_id, cohesive_zone_curves) and not is_in_list(c_id, cohesive_line_original_interface_curves):
                top_line_curves_tags.append(c_id)
gmsh.model.addPhysicalGroup(1, list(set(top_line_curves_tags)), name="TopLine")


# --- Mesh Refinement Fields ---

# Field 2: Distance Field for CohesiveParticle zones (particle-matrix interfaces)
field2_tag = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field2_tag, "CurvesList", cohesive_zone_curves) # Targets particle-matrix interfaces
gmsh.model.mesh.field.setNumber(field2_tag, "Sampling", 100)

# Field 3: Threshold Field for CohesiveParticle zones (uses field 2) - UNCOMMENTED AND ACTIVATED
field3_tag = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field3_tag, "InField", field2_tag)
gmsh.model.mesh.field.setNumber(field3_tag, "LcMin", fine_cl_zones) # Fine mesh at particle interfaces
gmsh.model.mesh.field.setNumber(field3_tag, "LcMax", coarse_cl_zones) # Coarser mesh away from interfaces
gmsh.model.mesh.field.setNumber(field3_tag, "DistMin", 0)
gmsh.model.mesh.field.setNumber(field3_tag, "DistMax", dist_transition)

# Field 1: Box Field for CohesiveInterface (Y=0 line)
field1_tag = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field1_tag, "VIn", res_coh) # Finer mesh size within the box
gmsh.model.mesh.field.setNumber(field1_tag, "VOut", res) # Coarser mesh size outside the box
field1_xmin, field1_xmax = 0, Lx
field1_ymin, field1_ymax = -0.5, 0.5 # Box around Y=0 line
gmsh.model.mesh.field.setNumber(field1_tag, "XMin", field1_xmin)
gmsh.model.mesh.field.setNumber(field1_tag, "XMax", field1_xmax)
gmsh.model.mesh.field.setNumber(field1_tag, "YMin", field1_ymin)
gmsh.model.mesh.field.setNumber(field1_tag, "YMax", field1_ymax)

# Field 4: Min Field to combine all relevant refinement fields
field4_tag = gmsh.model.mesh.field.add("Min")
# IMPORTANT: Include both the Box field (for Y=0 interface) and the Threshold field (for particle interfaces)
gmsh.model.mesh.field.setNumbers(field4_tag, "FieldsList", [field1_tag, field3_tag])

gmsh.model.mesh.field.setAsBackgroundMesh(field4_tag)

# --- Meshing Algorithms ---
gmsh.option.setNumber("Mesh.RecombineAll", 1) # Try to create quadrilateral elements
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # Select an algorithm for recombination (e.g., Anisotropic)
gmsh.option.setNumber("Mesh.Algorithm", 8) # Frontal-Delaunay for 2D (often good for complex geometries)

# --- Final Output ---
print(f"\n--- Physical Group Summary ---")
print(f"Number of 'CohesiveParticle' curves: {len(cohesive_zone_curves)}")
print(f"Number of 'CohesiveInterface' curves: {len(cohesive_line_original_interface_curves)}")
print(f"Number of 'LeftUpperLine' curves: {len(left_upper_curves_tags)}")
print(f"Number of 'LeftLowerLine' curves: {len(left_lower_curves_tags)}")
print(f"Number of 'RightLine' curves: {len(right_line_curves_tags)}")
print(f"Number of 'BottomLine' curves: {len(bottom_line_curves_tags)}")
print(f"Number of 'TopLine' curves: {len(top_line_curves_tags)}")

# Generate Mesh
gmsh.model.mesh.generate(1)
gmsh.model.mesh.generate(2)

gmsh.write("random_density_pdf_particles_test.msh2")

# Optional: Launch Gmsh GUI to visualize the mesh
gmsh.fltk.run()

# 4. Finalize Gmsh
gmsh.finalize()