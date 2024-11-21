from petsc4py import PETSc
from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem, io, plot ,geometry
import dolfinx.mesh as Mesh
from dolfinx.fem.petsc import (
    assemble_vector, 
    assemble_matrix,
    assemble_vector,
    create_vector,
    apply_lifting,
    set_bc
)
from ufl import (
    CellDiameter,
    FacetNormal,
    TestFunction,
    TrialFunction,
    avg,
    jump,
    conditional,
    div,
    dot,
    dS,
    ds,
    dx,
    extract_blocks,
    grad,
    gt,
    inner,
    outer,
    lhs,
    rhs,
)

import numpy as np  # NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Matplotlib para gráficos
from matplotlib.legend_handler import HandlerLine2D
from label_lines import * #graficas 
import math
import scipy
import time
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import pandas as pd
import sys
import os
from scipy import integrate
from pathlib import Path


np.set_printoptions(formatter={'float': '{: 0.2F}'.format})


start_code = time.time()
def fi (A,m,tau=1):
    d = scipy.sparse.identity(m)
    exp_A = scipy.linalg.expm(tau*A)
    A_1 = spsl.spsolve(tau*A,d)
    
    return (exp_A-d)*A_1 , exp_A
# algoritmos de arnoldi

def arnoldi_iteration(A, b,tau,n_old):
    m  = A.size[0]
    
    tol=1E-3
    n_int = n_old -1
    n=n_int
    error=1
    
    H  = np.zeros((n + 1, n))
    
    Q_ = np.zeros((m, n + 1))
    
    beta = b.norm()
    q = b.duplicate()
    q = b.copy()
    q.scale(1 / beta)
    q.assemble()
    j=0
    Q_[:, 0]  =  q.getArray()

    v = PETSc.Vec().createSeq(m)
    column_vec = PETSc.Vec().createSeq(m)
    m_indices = np.arange(m, dtype=np.int32)  # Índices de las filas (todas las filas)
    
    while error >= tol:
        
        if n>2 :
            H=np.pad(H, [(0, 1), (0, 1)], mode='constant', constant_values=0)
            Q_=np.pad(Q_, [(0, 0), (0, 1)], mode='constant', constant_values=0)
        e_1 = np.zeros(n)
        e_1[0] = 1
        e_m = np.zeros(n)
        e_m[n - 1] = 1
        n_indices = np.arange(n, dtype=np.int32)

        A.mult(q, v)

        for i in range(j + 1):

            column_vec.setValues(m_indices,Q_[:, i])
            column_vec.assemble()

            h_ij = column_vec.dot(v)
            H[i, j] = h_ij
            v.axpy(-h_ij, column_vec)

        v_n = v.norm()
        H[i + 1, j] = v_n
        q = v.copy()
        q.scale(1 / v_n)
        Q_[:, j + 1] =  q.getArray()

        j += 1
        if n>n_int or (n==n_int and j==n_int):

            fi_m, exp_H = fi(H[0:n, 0:n], n, tau)
            error = beta * abs(H[n,n-1]* tau * e_m.dot(np.array(fi_m.dot(e_1))[0]))
            n += 1
            
    print(f"H dim= {Q_[:,0:n].shape[1]} error = {error}")
    return Q_[:,0:n-1], H[0:n-1, 0:n-1],fi_m,e_1,n,exp_H,beta,Q_[:,0:n].shape[1]

def arnoldi_iteration_m(A, b, tau, n):
    print('ar_start')
    m  = A.size[0]

    H  = np.zeros((n + 1, n))
    
    Q_ = np.zeros((m, n + 1))

    e_1 = np.zeros(n)
    e_1[0] = 1
    e_m = np.zeros(n)
    e_m[n - 1] = 1

    beta = b.norm()
    q = b.duplicate()
    q = b.copy()
    q.scale(1 / beta)
    q.assemble()

    m_indices = np.arange(m, dtype=np.int32)  # Índices de las filas (todas las filas)
    n_indices = np.arange(n, dtype=np.int32)

    Q_[:, 0]  =  q.getArray()
    
    v = PETSc.Vec().createSeq(m)
    column_vec = PETSc.Vec().createSeq(m)
    print('ar_loop_start')
    for j in range(n):
        A.mult(q, v)
        
        for i in range(j + 1):

            column_vec.setValues(m_indices,Q_[:, i])
            column_vec.assemble()
            h_ij = column_vec.dot(v)
            H[i, j] = h_ij
            v.axpy(-h_ij, column_vec)
        
        v_n = v.norm()
        H[i + 1, j] = v_n
        q = v.copy()
        q.scale(1 / v_n)
        Q_[:, j + 1] =  q.getArray()
    # Implementar fi y exp_H de acuerdo a tus requisitos
    fi_m, exp_H = fi(H[0:n, 0:n], n, tau)
    error = beta * abs(H[n,n-1]* tau * e_m.dot(np.array(fi_m.dot(e_1))[0]))
    
    print(f"H dim= {Q_[:,0:n].shape[1]} error = {error}")

    return Q_[:,0:n], H[0:n, 0:n], fi_m, e_1, n, exp_H,beta
    

def exp_form(mesh,flux_method,C_file):
    orden =1
    
    V_ele = element("DG", mesh.topology.cell_name(), orden)
    V     = fem.functionspace(mesh, V_ele)
    
    
    u_a   = u_analy()
    u_a.t = 0.0
    u_0   = fem.Function(V)
    u_0.interpolate(u_a.eval)

#_________________________________ Dirichlet Boundary Conditions
    dofs_D = fem.locate_dofs_geometrical(V, boundary_D)
    u_bc   = fem.Function(V)
    u_bc.interpolate(u_a.eval)
    bcs    = fem.dirichletbc(u_bc, dofs_D)

#__________________________________ Varational Formulation    

    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n = FacetNormal(mesh)
    alpha = 1
    h = CellDiameter(mesh)

    
    if 'LaxF' == flux_method:
        C = w[0]
        flux = dot(avg(w*u),n('+')) + 0.5*C*jump(u)
    elif 'upwind' == flux_method:
        un = (dot(w, n) + abs(dot(w,n)))/2.0
        flux = jump(un*u)
    
    dif_nflux = -k*inner(jump(u,n),avg(grad(v)))*dS \
                - k*inner(avg(grad(u)),jump(v,n))*dS \
                + (alpha)*inner(jump(u,n),jump(v,n))*dS #flujo difusivo
    
    adv_nflux =  dot(jump(v), flux)*dS  #+ dot(v, u_analytical)*ds
    
        
    
    
    F_fem = - dot(grad(v),w*u)*dx + adv_nflux \
            - dot(k*grad(u), grad(v))*dx + dif_nflux
    
    K_fem = lhs(F_fem)
    Q_fem = rhs(F_fem) 
    print(Q_fem)
    C_fem=u*v*dx
    
    
    c = fem.form(C_fem)
    C = assemble_matrix(c)
    C.assemble()
    if orden == 1:
        dof_elem=mesh.ufl_cell().num_edges()
    elif orden == 2:
        dof_elem=mesh.ufl_cell().num_edges()*2
    elif orden == 3:
        dof_elem=mesh.ufl_cell().num_edges()*3 + 1
    
    C.setBlockSizes(dof_elem,dof_elem)
    
    # visualized block matrix 
    #def petsc2array(v):
    #    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    #    return s
    #plt.spy(petsc2array(C))
    
    C_1_array = C.invertBlockDiagonal() # matriz capacitiva inversa 
    
    # Crear la matriz de PETSc de tamaño total
    
    num_blocks = C_1_array.shape[0]
    matrix_size = C_1_array.shape[1] * num_blocks
    
    C_1 = PETSc.Mat().create()
    C_1.setSizes([matrix_size, matrix_size])
    C_1.setType(PETSc.Mat.Type.AIJ)
    C_1.setUp()
    
    # Asignar cada bloque en la diagonal
    for i in range(num_blocks):
        start = i * dof_elem
        end = start + dof_elem
        C_1.setValues(range(start, end),range(start, end), C_1_array[i])
    
    # Finalizar el ensamblaje de la matriz
    C_1.assemble()
    
    if stage_m:
        print('C invertion finish')
        print('K assemble init')
    #ensamble matriz de rigidez
    k_ = fem.form(K_fem)
    K = assemble_matrix(k_)
    
    
    
    K.assemble()
    K.scale(-1.0)
    if stage_m:
        print('K assemble finish')
    # Multiplicación de C_1 y K
    A = C_1.matMult(K)
    return(V,u,v,A,C_1,Q_fem,u_0,[bcs],u_bc)

def exp_solver(A,u_i,n,table,auto,m_u,u=0):
    arnoldi_ti=time.time()
    if stage_m:
        print('Arnoldi iteration init')
    if auto:
        V_m,H_m,fi_m,e_1,m,exp_H,Beta,m_res  = arnoldi_iteration(A,u_i,real_dt,m_u)
    else:
        V_m,H_m,fi_m,e_1,m,exp_H,Beta  = arnoldi_iteration_m(A,u_i,real_dt,m_u)
    arnoldi_t=time.time()-arnoldi_ti
    if stage_m:
        print(f'KSP projection time = {arnoldi_t}')
    solver_ti=time.time()
    if n==0:
        u = real_dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)+u_0.x.array
    else:
        u += real_dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)
    solver_t=time.time() -solver_ti
    table.append([arnoldi_t,solver_t,V_m.shape[1],t])
    return u , m

def Q_t(Q_fem,C_1,bcs):
    q_v = fem.form(Q_fem)
    Q_v = assemble_vector(q_v)
    set_bc(Q_v,[bcs])
    Q_v.assemble()
    
    
    #Q_ = -Q_v.get_local()
    Q_v.scale(-1.0)
    Q_a = PETSc.Vec().create()
    Q_a.setSizes(Q_v.getSize())
    Q_a.setType(PETSc.Vec.Type.SEQ)
    Q_a.assemble()
    C_1.mult(Q_v,Q_a) 
    return Q_a
    
def BDF_form(mesh,pconst,flux_method):
    
    # definición del espacio de funciones
    
    V_ele = element("DG", mesh.topology.cell_name(), 1)
    V     = fem.functionspace(mesh, V_ele)
    #w_ele = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim))
    #V_w =fem.functionspace(mesh, w_ele)
    
    #w=project(u1, V_w)
    
    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n = FacetNormal(mesh)
    
    u_a   = u_analy()
    u_a.t = 0.0

    
    u_bc = fem.Function(V)
    u_bc.interpolate(u_a.eval)

    u_0   = fem.Function(V)
    u_0.interpolate(u_a.eval)
    
    u_n   = fem.Function(V)  # Interpola u_D en el espacio de funciones V
    u_n.interpolate(u_a.eval)
    
    u_nn  = fem.Function(V) 
    u_nn.interpolate(u_a.eval)
    
    u_nnn = fem.Function(V)
    u_nnn.interpolate(u_a.eval)
    
    pconst = BDF_coef[Integrador]
    du     = pconst[0]*u
    du_n   = pconst[1]*u_n
    du_nn  = pconst[2]*u_nn
    du_nnn = pconst[3]*u_nnn
    
    du_t = du+du_n +du_nn +du_nnn
    
    dofs_D = fem.locate_dofs_geometrical(V, boundary_D)

# ________________Boundary condition 
    bcs = fem.dirichletbc(u_bc, dofs_D)
    
    alpha=1
    h = CellDiameter(mesh)
    #ds = Measure('ds', domain=mesh, subdomain_data=contorno)
    
    if 'LaxF' == flux_method:
        C = w[0]
        flux = dot(avg(w*u),n('+')) + 0.5*C*jump(u)
    elif 'upwind' == flux_method:
        un = (dot(w, n) + abs(dot(w,n)))/2.0
        flux = jump(un*u,n)
    
    dif_nflux = -k*inner(jump(u,n),avg(grad(v)))*dS - k*inner(avg(grad(u)),jump(v,n))*dS \
                + (alpha)*inner(jump(u,n),jump(v,n))*dS #flujo difusivo
    
    adv_nflux =  dot(jump(v,n), flux)*dS  #+ dot(v, u_analytical)*ds
    
    #F = du_t*v*dx - real_dt*dot(grad(v),w*u)*dx + real_dt*adv_nflux - real_dt*dot(k*grad(u), grad(v))*dx + real_dt*dif_nflux
    
    F = du_t*v*dx - real_dt*dot(grad(v), w*u)*dx - real_dt*dot(k*grad(u), grad(v))*dx
    #F += real_dt*dot(jump(v), flux)*dS + real_dt*dot(v, u_analytical*u)*ds
    F +=  real_dt*dif_nflux + real_dt*adv_nflux 

    # Separa la parte izquierda y derecha del problema variacional  
    L = fem.form(lhs(F))
    R = fem.form(rhs(F))  
      
    return(V,L,R,u_n,u_nn,u_nnn,[bcs],u_bc)



np.set_printoptions(formatter={'float': '{: 0.2F}'.format})
start_code = time.time()


ny  = int(sys.argv[1])
nx = ny*20

x_start = 0
x_end = 20
mesh = Mesh.create_rectangle(MPI.COMM_WORLD, [np.array([x_start,0 ]), np.array([x_end, 1])],[nx, ny], 
                             Mesh.CellType.triangle,diagonal=Mesh.DiagonalType(1))

Integrador  = sys.argv[2]#"exp"  # time integration method
flux_method = sys.argv[3]#"upwind"
mode_KSP = sys.argv[4] #"auto"
stage_m =True

# Parámetros de simulación
dt = 1
T = 10         # Tiempo final
num_steps = int(round(T/dt,0))# Número de pasos de tiempo
real_dt= T/num_steps
k = fem.Constant(mesh,PETSc.ScalarType((0.0001)))
w = fem.Constant(mesh,PETSc.ScalarType((1,0)))
print(1)


#f=Expression('x[0]*(6*t - pow(x[0], 2) - 4*pow(x[0], 2)*t*t)*exp(-t*pow(x[0], 2))',t=0,degree=2)
class u_analy:
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        # Added some spatial variation here. Expression is sin(t)*x
        return np.full(x.shape[1], 1/(np.sqrt(1+0.0004*self.t))*np.exp(-np.pow((x[0]-3-self.t),2)/(1+0.0004*self.t)))
        
class u_analy_bcs:
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        # Added some spatial variation here. Expression is sin(t)*x
        return np.full(x.shape[1], -1/(np.sqrt(1+0.0004*self.t))*np.exp(-np.pow((x[0]-3-self.t),2)/(1+0.0004*self.t)))

def boundary_D(x):
    Left  = np.isclose(x[0], x_start )
    Right = np.isclose(x[0], x_end   )
    Up    = np.isclose(x[1], 1       )
    Down  = np.isclose(x[1], 0       )
    return np.logical_or(np.logical_or(Left, Right), np.logical_or(Down, Up))




f=fem.Constant(mesh,PETSc.ScalarType(0))

#q=Expression('-1',t=0,degree=2) #neumman BC

BDF_coef = {"BDF1":[1,-1,0,0],"BDF2":[3./2,-2,1./2,0.0],"BDF3":[11/6,-3,3/2,-1/3],"BDF_OP":[0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3]}

# Creación de la malla y definición del espacio de funciones



#contorno = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
#left().mark(contorno, 1)
#right().mark(contorno, 2)
print('Number of nodes:', mesh.geometry.x.shape[0] )
print('Number of elements:', mesh.topology.index_map(2).size_local )
print('Number of DOFs:',mesh.geometry.x.shape[0] )

try:
    os.mkdir(f"results_dt_{real_dt}") 
except FileExistsError:
    pass
t= 0
if Integrador == "exp":
    C_file= sys.argv[5] == "y"
    assemble_ti=time.time()
    V,u_t,v_t,A,C_1,Q_a,u_0,bcs,u_bc = exp_form(mesh,flux_method,C_file)
    u_i = PETSc.Vec().create()
    u_i.setSizes(A.sizes[0][0])
    u_i.setType(PETSc.Vec.Type.SEQ)
    u_i.assemble()
    if stage_m:
        print(f"Tiempo de ensamblaje {time.time()- assemble_ti}")
elif (Integrador == "BDF1")or(Integrador == "BDF2") or (Integrador == "BDF3") or (Integrador == "BDF_OP"):
    V,L,R,u_n,u_nn,u_nnn,bcs,u_bc = BDF_form(mesh,BDF_coef[Integrador],flux_method)
    


    A = assemble_matrix(L, bcs=bcs)
    A.assemble()
    b = create_vector(R)
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    
else: 
    print("esquema de integración temporal erroneo")
    exit()

uh = fem.Function(V)
uh.name = "u"
t=0
u_file = io.VTKFile(mesh.comm, f"resultados_{Integrador}/u.pvd","w")
u_file.write_mesh(mesh)

L2=[]
table =[]
sol=[]

#points for plotting 


tol = 0.001  # Avoid hitting the outside of the domain
x_dot = np.linspace(x_start + tol, x_end - tol, 101)
points = np.zeros((3, 101))
points[0] = x_dot
points[1] = np.full(101, 0.5)
bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
#________ Solución analitica
u_a = u_analy()
u_a.t = 0
u_analitica = fem.Function(V)
def near(var,value,tol):
    return abs(var-value) < tol  
L2_error=fem.form((uh - u_analitica )**2 * dx)
for n_s in range(num_steps):
    #f.t=t
    #q.t=t
    if Integrador == "exp":
        if mode_KSP == 'auto': 
            Q=0#Q_t(Q_fem,C_1,bc)

            if n_s ==0:
                #A.multAdd(u_0.x.petsc_vec,Q,u_i)
                A.mult(u_0.x.petsc_vec , u_i)
                u,m_u =exp_solver(A,u_i,n_s,table,True,20)
            else:
                #A.multAdd(uh.x.petsc_vec,Q,u_i)
                A.mult(uh.x.petsc_vec , u_i)
                u,m_u =exp_solver(A,u_i,n_s,table,True,m_u,u)
        else: 
            m_u = int(mode_KSP)
            Q=0#Q_t(Q_fem,C_1,bc)
            if n_s ==0:
                #A.multAdd(u_0.x.petsc_vec,Q,u_i)
                A.mult(u_0.x.petsc_vec , u_i)
                u,H_m =exp_solver(A,u_i,n_s,table,False,m_u)
            else:
                #A.multAdd(uh.x.petsc_vec,Q,u_i)
                A.mult(uh.x.petsc_vec , u_i)
                u =exp_solver(A,u_i,n_s,table,False,m_u,u)
        uh.x.array[:] =u.getA()[0]
    
    elif (Integrador == "BDF1")or(Integrador == "BDF2") or (Integrador == "BDF3") or (Integrador == "BDF_OP"):
        with b.localForm() as loc_b:
                loc_b.set(0)
        assemble_vector(b , R)
    
        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [L], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)
    
        # Solve linear problem
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
    
        # Update solution at previous time step (u_n)
        u_nnn.x.array[:] = u_nn.x.array
        u_nn.x.array[:] = u_n.x.array
        u_n.x.array[:] = uh.x.array
    
    u_file.write_function(uh,t)
    t += real_dt
    u_a.t += dt
    u_analitica.interpolate(u_a.eval)
    u_bc.interpolate(u_a.eval)
    
    u_an = []
    u_p = []
    u_p = uh.eval(points_on_proc, cells)
    u_an = u_analitica.eval(points_on_proc,cells)
    if near(t,real_dt,real_dt/2) or near(t,5,real_dt/2) or near(t,10,real_dt/2):
        if near(t,real_dt,real_dt/2):
            plt.plot(points_on_proc[:, 0],u_p,'ko',label=f't = {t:.2f} [s]')
        elif near(t,5,real_dt/2):
            plt.plot(points_on_proc[:, 0],u_p,'k^',label=f't = {t:.2f} [s]')
        elif near(t,10,real_dt/2):
            plt.plot(points_on_proc[:, 0],u_p,'k*',label=f't = {t:.2f} [s]')
        plt.plot(points_on_proc[:, 0],u_an,'b--')
       
    L2_norm=np.sqrt(mesh.comm.allreduce(fem.assemble_scalar(L2_error), op=MPI.SUM))
    L2.append([L2_norm,t])
    print(f'step:{n_s+1} of {num_steps} time = {t}, L2 error = {L2_norm}')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim(0,1)
    plt.legend(loc='upper left')
plt.show()

print(f"total time = {time.time() - start_code}")