"""
file: geom.py
author: Philipp van der Loos

Creates a Gmsh 3D geometry: a cube split by a horizontal cohesive interface at z = Lz/2,
with a cylindrical kink cut at x = 0. The mesh uses hexahedral elements via transfinite meshing
for the bulk volumes, with local handling for the kink region.
"""

import gmsh
import math

# Initialize Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("cohesive_interface_mesh")

# Define dimensions
Lx = 10.0
Ly = 5
Lz = 4.0
z_split = Lz / 2.0
r_kink = Ly / 20.0  # Radius of cylindrical kink

# Mesh parameters
global_mesh_max = 1.0
cohesive_interface_mesh_size = 1.0
nx = 20  # Number of elements along x
ny = 10  # Number of elements along y
nz = 12   # Number of elements along z (split across both volumes)

# --- Create Geometry using OpenCASCADE ---
# Create cylinder for kink
# cyl_kink_tag = gmsh.model.occ.addCylinder(0, 0, z_split, 0, Ly, 0, r_kink)

# Create lower and upper cuboids
box_lower_tag = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, z_split)
box_upper_tag = gmsh.model.occ.addBox(0, 0, z_split, Lx, Ly, z_split)

# Merge coincident surfaces
gmsh.model.occ.removeAllDuplicates()

# Perform boolean cut to create kink
# objects_for_kink_cut = [(3, box_lower_tag), (3, box_upper_tag)]
# tools_for_kink_cut = [(3, cyl_kink_tag)]
# modified_main_solid_dim_tags, _ = gmsh.model.occ.cut(
#     objects_for_kink_cut,
#     tools_for_kink_cut,
#     removeObject=True,
#     removeTool=True
# )

# lower_volume_tag = modified_main_solid_dim_tags[0][1]
# upper_volume_tag = modified_main_solid_dim_tags[1][1]
lower_volume_tag = box_lower_tag
upper_volume_tag = box_upper_tag

# Synchronize geometry
gmsh.model.occ.synchronize()

# --- Identify Physical Groups ---
all_surfaces = gmsh.model.getEntities(2)
lower_vol_boundary_surfaces = gmsh.model.getBoundary([(3, lower_volume_tag)], oriented=False)
upper_vol_boundary_surfaces = gmsh.model.getBoundary([(3, upper_volume_tag)], oriented=False)

# Identify cohesive interface surfaces
cohesive_interface_tags = []
lower_tags_set = set([s[1] for s in lower_vol_boundary_surfaces])
upper_tags_set = set([s[1] for s in upper_vol_boundary_surfaces])
for tag in lower_tags_set:
    if tag in upper_tags_set:
        cohesive_interface_tags.append(tag)

# Initialize lists for boundary surfaces
upper_right_planar_surfaces = []
lower_right_planar_surfaces = []
top_surfaces = []
bottom_surfaces = []
upper_left_planar_surfaces = []
lower_left_planar_surfaces = []

# Identify boundary surfaces
for dim, tag in all_surfaces:
    bbox = gmsh.model.getBoundingBox(dim, tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    # Right side (x = Lx)
    if math.isclose(xmin, Lx, abs_tol=1e-6) and math.isclose(xmax, Lx, abs_tol=1e-6):
        if tag not in cohesive_interface_tags:
            adj_volumes = gmsh.model.getAdjacencies(2, tag)[0]
            if upper_volume_tag in adj_volumes:
                upper_right_planar_surfaces.append(tag)
            elif lower_volume_tag in adj_volumes:
                lower_right_planar_surfaces.append(tag)
    # Top surface (z = Lz)
    if math.isclose(zmin, Lz, abs_tol=1e-6) and math.isclose(zmax, Lz, abs_tol=1e-6):
        top_surfaces.append(tag)
    # Bottom surface (z = 0)
    if math.isclose(zmin, 0, abs_tol=1e-6) and math.isclose(zmax, 0, abs_tol=1e-6):
        bottom_surfaces.append(tag)
    # Left surfaces (x = 0)
    if math.isclose(xmin, 0, abs_tol=1e-6) and math.isclose(xmax, 0, abs_tol=1e-6):
        if tag not in cohesive_interface_tags:
            adj_volumes = gmsh.model.getAdjacencies(2, tag)[0]
            if upper_volume_tag in adj_volumes:
                upper_left_planar_surfaces.append(tag)
            elif lower_volume_tag in adj_volumes:
                lower_left_planar_surfaces.append(tag)

# Define physical groups
gmsh.model.addPhysicalGroup(2, cohesive_interface_tags, name="CohesiveInterface")
gmsh.model.addPhysicalGroup(2, upper_right_planar_surfaces, name="UpperRightSurface")
gmsh.model.addPhysicalGroup(2, lower_right_planar_surfaces, name="LowerRightSurface")
gmsh.model.addPhysicalGroup(2, upper_left_planar_surfaces, name="UpperLeftSurface")
gmsh.model.addPhysicalGroup(2, lower_left_planar_surfaces, name="LowerLeftSurface")
gmsh.model.addPhysicalGroup(2, top_surfaces, name="TopSurface")
gmsh.model.addPhysicalGroup(2, bottom_surfaces, name="BottomSurface")
gmsh.model.addPhysicalGroup(3, [lower_volume_tag], name="LowerDomain")
gmsh.model.addPhysicalGroup(3, [upper_volume_tag], name="UpperDomain")

# --- Transfinite Meshing for Hexahedral Elements ---
# Get all curves and surfaces
all_curves = gmsh.model.getEntities(1)
all_volumes = gmsh.model.getEntities(3)

# Apply transfinite meshing to curves
for dim, tag in all_curves:
    bbox = gmsh.model.getBoundingBox(dim, tag)
    length = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    if math.isclose(length, Lx, abs_tol=1e-6):
        gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
    elif math.isclose(length, Ly, abs_tol=1e-6):
        gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1)
    elif math.isclose(length, z_split, abs_tol=1e-6):
        gmsh.model.mesh.setTransfiniteCurve(tag, nz // 2 + 1)
    # Curves of the cylindrical kink are left unstructured

# Apply transfinite meshing to surfaces (except kink surfaces)
for dim, tag in all_surfaces:
    # Skip cylindrical kink surfaces (identified by non-planar geometry)
    bbox = gmsh.model.getBoundingBox(dim, tag)
    if not (math.isclose(bbox[0], 0, abs_tol=1e-6) and math.isclose(bbox[3], 0, abs_tol=1e-6) and
            abs(bbox[5] - bbox[2]) < r_kink * 2):
        gmsh.model.mesh.setTransfiniteSurface(tag)

# Apply transfinite meshing to volumes
for dim, tag in all_volumes:
    gmsh.model.mesh.setTransfiniteVolume(tag)

# Local mesh refinement near cohesive interface and kink
field1_tag = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(field1_tag, "VIn", cohesive_interface_mesh_size)
gmsh.model.mesh.field.setNumber(field1_tag, "VOut", global_mesh_max)
field1_xmin, field1_xmax = 0, Lx  # Refine near x = 0 for kink
field1_ymin, field1_ymax = 0, Ly
field1_zmin, field1_zmax = z_split - r_kink, z_split + r_kink
gmsh.model.mesh.field.setNumber(field1_tag, "XMin", field1_xmin)
gmsh.model.mesh.field.setNumber(field1_tag, "XMax", field1_xmax)
gmsh.model.mesh.field.setNumber(field1_tag, "YMin", field1_ymin)
gmsh.model.mesh.field.setNumber(field1_tag, "YMax", field1_ymax)
gmsh.model.mesh.field.setNumber(field1_tag, "ZMin", field1_zmin)
gmsh.model.mesh.field.setNumber(field1_tag, "ZMax", field1_zmax)

field4_tag = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(field4_tag, "FieldsList", [field1_tag])
gmsh.model.mesh.field.setAsBackgroundMesh(field4_tag)

# Set meshing algorithms for hexahedral mesh
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 8)    # 2D: Frontal-Delaunay for quads
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 3D: Structured (transfinite)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # Ensure transfinite meshing

# Debugging output
print(f"Number of 'CohesiveInterface' surfaces: {len(cohesive_interface_tags)}")
print(f"Number of 'RightSide' surfaces: {len(upper_right_planar_surfaces)}")
print(f"Number of 'RightSide' surfaces: {len(lower_right_planar_surfaces)}")
print(f"Number of 'UpperLeftSurface' surfaces: {len(upper_left_planar_surfaces)}")
print(f"Number of 'LowerLeftSurface' surfaces: {len(lower_left_planar_surfaces)}")
print(f"Number of 'TopSurface' surfaces: {len(top_surfaces)}")
print(f"Number of 'BottomSurface' surfaces: {len(bottom_surfaces)}")

# Generate and save mesh
gmsh.model.mesh.generate(3)
# gmsh.option.setString("Mesh.Format", "msh2")
gmsh.write("cohesive_interface_mesh_simple.msh2")

# Optional: Run GUI
gmsh.fltk.run()

# Finalize
gmsh.finalize()
print("Gmsh mesh 'cohesive_interface_mesh.msh2' generated successfully.")