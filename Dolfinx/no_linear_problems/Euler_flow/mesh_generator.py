import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.mesh import create_cell_partitioner, GhostMode


def generate(comm, L , H , x0 , y0 , theta ):
    gmsh.initialize()


    
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    res_min = 1 / 100
    fluid_marker = 1

    angle = np.deg2rad(theta)
    
    
    volume_id = {"fluid": 1}

    boundary_id = {"inlet": 2,
                   "outlet": 3,
                   "wall": 4,
                   "obstacle": 5}
    inflow, outflow, walls, obstacle = [], [], [], []

    if mesh_comm.rank == model_rank:
        gmsh.merge('Mesh/0012 v2.step')
            
        # Sincronizar el modelo
        gmsh.model.geo.synchronize()
        gmsh.model.occ.dilate([tag for tag in gmsh.model.getEntities(2)],0,0,0,0.1,0.1,0.1)
        gmsh.model.occ.rotate([tag for tag in gmsh.model.getEntities(2)],0,0,0,0.0,0.0,1.0,angle)
        rectangle = gmsh.model.occ.addRectangle(x0, y0, 0, L, H)
        #disk = gmsh.model.occ.add_disk(0,0,0,r,r)
        #dom = gmsh.model.occ.fuse([(gdim, 3)], [(gdim, 2)])
        fluid = gmsh.model.occ.cut([(gdim, 2)], [(gdim, 1)])
        
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(dim=gdim)
        print(volumes)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            #print(center_of_mass)
            if np.allclose(center_of_mass, [(L + x0), 0, 0]):
                #print(boundary_id['outlet'])
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass[1],  H/2 ) or np.allclose(center_of_mass[1],  -H/2):
                #print(boundary_id['wall'])
                walls.append(boundary[1])
            elif np.all(center_of_mass < (0.0,0.0,0.0)):
                #print(boundary_id['inlet'])
                inflow.append(boundary[1])
            else:
                #print(boundary_id['obstacle'])
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, boundary_id['wall'])
        gmsh.model.setPhysicalName(1, boundary_id['wall'], "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, boundary_id['inlet'])
        gmsh.model.setPhysicalName(1, boundary_id['inlet'], "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, boundary_id['outlet'])
        gmsh.model.setPhysicalName(1, boundary_id['outlet'], "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, boundary_id['obstacle'])
        gmsh.model.setPhysicalName(1, boundary_id['obstacle'], "Obstacle")
    
        #mesh size
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.1 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.005)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax",  2* H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        #mesh
        # gmsh.option.setNumber("Mesh.Algorithm", 8)
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("meshtest.msh")

    partitioner = create_cell_partitioner(GhostMode.shared_facet)
    msh, _, ft = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ft, boundary_id
gmsh.initialize()





if __name__ == "__main__":
    msh, ft, boundary_id = generate(MPI.COMM_WORLD, h=0.05)

    from dolfinx import io
    with io.XDMFFile(msh.comm, "benchmark_mesh.xdmf", "w") as file:
        file.write_mesh(msh)
       # print(msh.geometry.x)
        file.write_meshtags(ft,msh.geometry)