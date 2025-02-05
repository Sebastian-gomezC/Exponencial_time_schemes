from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from basix.ufl import element 
from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from ufl import (CellDiameter, FacetNormal, TestFunction, TrialFunction, avg,
                 conditional, div, dot, dS, ds, dx, grad, gt, inner, outer)
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
if np.issubdtype(PETSc.ScalarType, np.complexfloating):  # type: ignore
    print("Demo should only be executed with DOLFINx real mode")
    exit(0)

def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(msh, default_real_type(1.0)) * dx)), op=MPI.SUM)
    return (1 / vol) * msh.comm.allreduce(fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)





import gmsh
from mpi4py import MPI
import numpy as np
gmsh.initialize()

L = 13
H = 4
x0 = -3
y0 = -2
c_x = c_y = 0.2

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
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
angle = np.deg2rad(0)

if mesh_comm.rank == model_rank:
    gmsh.merge('Mesh/0012 v2.step')
        
    # Sincronizar el modelo
    gmsh.model.geo.synchronize()
    gmsh.model.occ.dilate([tag for tag in gmsh.model.getEntities(2)],0,0,0,0.1,0.1,0.1)
    gmsh.model.occ.rotate([tag for tag in gmsh.model.getEntities(2)],0,0,0,0.0,0.0,1.0,angle)
    rectangle = gmsh.model.occ.addRectangle(x0, y0, 0, L, H)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, 1)])
    
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        #print(center_of_mass)
        if np.allclose(center_of_mass, [(L + x0), 0, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [x0, 0, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2 + x0, H/2, 0]) or np.allclose(center_of_mass, [L / 2 + x0, -H/2, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")



    #mesh size
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.1 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    #mesh
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write("meshtest.msh")

msh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"

gmsh.finalize()
t = 0
T = 2                       # Final time
dt = 1 / 900                 # Time step size
num_steps = int(T / dt)
mu_ =  1.8e-5
mu = fem.Constant(msh, PETSc.ScalarType(mu_))  # Dynamic viscosity
rho_ = 1.2
rho = fem.Constant(msh, PETSc.ScalarType(rho_))     # Density
Re = 1e6
vel = Re*mu_/rho_
k =  1 # Polynomial degree
# Function spaces for the velocity and for the pressure
v_ele = element("DG", msh.topology.cell_name(), k+1 , shape=(msh.geometry.dim,))
V = fem.functionspace(msh, v_ele) 
Q = fem.functionspace(msh, ("Discontinuous Lagrange", k))

# Funcion space for visualising the velocity field
gdim = msh.geometry.dim
W = fem.functionspace(msh, ("Discontinuous Lagrange", k + 1, (gdim,)))

# Define trial and test functions

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

delta_t = fem.Constant(msh, dt)
alpha = fem.Constant(msh, default_real_type(6.0 * k**2))

h = CellDiameter(msh)
n = FacetNormal(msh)


def jump(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


a_00 = (1.0 / Re) * (inner(grad(u), grad(v)) * dx
                     - inner(avg(grad(u)), jump(v, n)) * dS
                     - inner(jump(u, n), avg(grad(v))) * dS
                     + (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS
                     - inner(grad(u), outer(v, n)) * ds
                     - inner(outer(u, n), grad(v)) * ds
                     + (alpha / h) * inner(outer(u, n), outer(v, n)) * ds)
a_01 = - inner(p, div(v)) * dx
a_10 = - inner(div(u), q) * dx

a = fem.form([[a_00, a_01],
              [a_10, None]])

f = fem.Function(W)
u_D = fem.Function(V)

class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        if self.t < 1 :
            values[0] =   vel*self.t**2 * ( 1 - x[1]**2 / ((H/2)**2))
        else : 
            values[0] = vel  * ( 1 - x[1]**2 / ((H/2)**2))
        return values

inlet_velocity = InletVelocity(t)
u_D.interpolate(inlet_velocity)
L_0 = inner(f, v) * dx + (1 / Re) * (- inner(outer(u_D, n), grad(v)) * ds
                                     + (alpha / h) * inner(outer(u_D, n), outer(v, n)) * ds)
L_1 = inner(fem.Constant(msh, default_real_type(0.0)), q) * dx
L = fem.form([L_0, L_1])

# Boundary conditions
# Inlet
fdim = msh.topology.dim - 1
bcu_inflow = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls
u_nonslip = np.array((0,) * msh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
# Obstacle
bcu_obstacle = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
bcs = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
bcp_outlet = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
#bcs.append( bcp_outlet)

# Assemble Stokes problem
A = assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = assemble_vector_block(L, a, bcs=bcs)

# Create and configure solver
ksp = PETSc.KSP().create(msh.comm)  # type: ignore
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()  # type: ignore
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

# Solve Stokes for initial condition
x = A.createVecRight()
try:
    ksp.solve(b, x)
except PETSc.Error as e:  # type: ignore
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e


# Split the solution
u_h = fem.Function(V)
p_h = fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()
# Subtract the average of the pressure since it is only determined up to
# a constant
p_h.x.array[:] -= domain_average(msh, p_h)

u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file
t = 0.0
try:
    u_file = VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
    p_file = VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])
    u_file.write(t)
    p_file.write(t)
except AttributeError:
    print("File output requires ADIOS2.")

# Create function to store solution and previous time step
u_n = fem.Function(V)
u_n.x.array[:] = u_h.x.array
lmbda = conditional(gt(dot(u_n, n), 0), 1, 0)
u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")
a_00 += inner(u / delta_t, v) * dx - \
    inner(u, div(outer(v, u_n))) * dx + \
    inner((dot(u_n, n))("+") * u_uw, v("+")) * dS + \
    inner((dot(u_n, n))("-") * u_uw, v("-")) * dS + \
    inner(dot(u_n, n) * lmbda * u, v) * ds
a = fem.form([[a_00, a_01],
              [a_10, None]])

L_0 += inner(u_n / delta_t, v) * dx - inner(dot(u_n, n) * (1 - lmbda) * u_D, v) * ds
L = fem.form([L_0, L_1])

# Time stepping loop
for n in range(num_steps):
    
    t += delta_t.value
    print(f'time = {t}')
    inlet_velocity.t = t
    u_D.interpolate(inlet_velocity)
    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, a, bcs=bcs)  # type: ignore
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)  # type: ignore

    # Compute solution
    ksp.solve(b, x)

    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= domain_average(msh, p_h)

    u_vis.interpolate(u_h)

    # Write to file
    try:
        u_file.write(t)
        p_file.write(t)
    except NameError:
        pass

    # Update u_n
    u_n.x.array[:] = u_h.x.array

try:
    u_file.close()
    p_file.close()
except NameError:
    pass