

from fenics import *  

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
parameters['linear_algebra_backend'] = 'Eigen'
parameters["ghost_mode"] = "shared_facet"

start_code = time.time()
def fi (A,m,tau=1):
    d=scipy.sparse.identity(m)
    exp_A =scipy.linalg.expm(tau*A)
    return (exp_A-d)*spsl.spsolve(tau*A,d) , exp_A
# algoritmos de arnoldi
def arnoldi_iteration_m(A, b,tau,n):
    m = A.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    e_1=np.zeros(n)
    e_1[0]=1
    e_m=np.zeros(n)
    e_m[n-1]=1
    beta= np.linalg.norm(b)
    q = b / beta
    Q[:, 0] = q
    for k in range(n):
        v = A.dot(q)
        for j in range(k + 1):
            h[j, k] = np.dot(Q[:, j].conj(), v)  # <-- Q needs conjugation!
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        q = v / h[k + 1, k]
        Q[:, k + 1] = q

    fi_m,exp_H =fi(h[0:n, 0:n],n,tau)
    error = beta*abs(h[n,n-1]*tau*e_m.dot(np.array(fi_m.dot(e_1))[0]))
    print(f"H dim= {Q[:,0:n].shape[1]} error = {error}")
          
    return Q[:,0:n], h[0:n, 0:n],fi_m,e_1,n,exp_H

def arnoldi_iteration(A, b,tau):
    m = A.shape[0]
    beta= np.linalg.norm(b)
    tol=1E-3
    n_int=4
    n=n_int
    error=1
    #print("beta on er",beta)
    eta=1/sqrt(2)
    H = np.zeros((n + 1, n))
    V = np.zeros((m, n + 1))
    V[:, 0] = b / beta
    j=0
    while error >= tol:
        if n>2 :
            H=np.pad(H, [(0, 1), (0, 1)], mode='constant', constant_values=0)
            V=np.pad(V, [(0, 0), (0, 1)], mode='constant', constant_values=0)
        e_m=np.zeros(n)
        e_m[n-1]=1
        e_1=np.zeros(n)
        e_1[0]=1
        v = A.dot(V[:, j])
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i].conj(), v)  # <-- V needs conjugation!
            v = v - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(v)
        V[:, j + 1] = v / H[j + 1, j]
        j += 1
        if n>n_int or (n==n_int and j==n_int):
            fi_m,exp_H =fi(H[0:n, 0:n],n,tau)
            error = beta*abs(H[n,n-1]*tau*e_m.dot(np.array(fi_m.dot(e_1))[0]))
            n += 1
        if stage_m:
            print(f"H dim= {V.shape[1]} error = {error}")
    print(f"H dim= {V.shape[1]} error = {error}")
    return V[:,0:n-1], H[0:n-1, 0:n-1],fi_m,e_1,n,exp_H 
class left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0],0) and on_boundary
class right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],1)
def on_boundary(x, on_boundary):
    return on_boundary
def exp_form(mesh,flux_method,C_file):
    orden =1
    V=FunctionSpace(mesh,"DG",orden)
    V_w =VectorFunctionSpace(mesh, "DG", 1)
    
    w=project(u1, V_w) 
    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)
    
    u_n = Function(V)  # Interpola u_D en el espacio de funciones V    
    u_n = project(u_0, V)  # Interpola u_D en el espacio de funciones V

    #bcs =[]
    #bcs.append(DirichletBC(V, Constant(0.0),contorno,2))
    #bcs.append(DirichletBC(V, Constant(0.0),contorno,1))
    bcs =DirichletBC(V, u_analytical_bcs, on_boundary,method='geometric') 
    alpha=1
    h= mesh.cells().size
    ds = Measure('ds', domain=mesh, subdomain_data=contorno)
    
    if 'LaxF' == flux_method:
        C = w[0]
        flux = dot(avg(w*u),n('+')) + 0.5*C*jump(u)
    elif 'upwind' == flux_method:
        un = (dot(w, n) + abs(dot(w,n)))/2.0
        flux = jump(un*u)
    
    dif_nflux = -k*dot(jump(u,n),avg(grad(v)))*dS - k*dot(avg(grad(u)),jump(v,n))*dS \
    + (alpha/h)*dot(jump(u,n),jump(v,n))*dS #flujo difusivo
    
    adv_nflux =  dot(jump(v), flux)*dS + dot(v, u_analytical)*ds

    F_fem =- dot(grad(v),w*u)*dx + adv_nflux - dot(k*grad(u), grad(v))*dx + dif_nflux

    K_fem , Q_fem = lhs(F_fem), rhs(F_fem) 
    
    C_fem=u*v*dx
    if stage_m:
        print('C invertion init')
    #ensamble matriz capacitiva 

    C_petsc = PETScMatrix()
    C=assemble(C_fem, tensor=C_petsc)
    bcs.apply(C_petsc)
    if orden == 1:
        dof_elem=mesh.ufl_cell().num_edges()
    elif orden == 2:
        dof_elem=mesh.ufl_cell().num_edges()*2
    elif orden == 3:
        dof_elem=mesh.ufl_cell().num_edges()*3 + 1
    
    C.mat().setBlockSizes(dof_elem,dof_elem)
    C_1_array = C.mat().invertBlockDiagonal()
    #C_1_csr =scipy.sparse.block_diag([C_1_array[i] for i in range(C_1_array.shape[0])], format='csr')
    #print(f"csr site C^-1 {C_1_csr.shape}")
    num_blocks = len(C_1_array)

# Rellena la matriz por bloques con las submatrices cuadradas
    C_1 = BlockMatrix(num_blocks,num_blocks)
    for i in range(num_blocks):
        
        submatrix_petsc = PETScMatrix()
        submatrix_petsc.mat().createDense(array=submatrices[i][j])

        

    
    

    if stage_m:
        print('C invertion finish')
        print('K assemble init')
    #ensamble matriz de rigidez
    
    K_petsc = PETScMatrix()
    assemble(K_fem, tensor=K_petsc)
    bcs.apply(K_petsc)
    K_petsc.mat().scale(-1.0)
    if stage_m:
        print('K assemble finish')
    if stage_m:
        print(f"K matrix type = {type(K_petsc.mat())} tamaño = {K_petsc.mat().sizes}")
        print(f"C^-1 matrix type = {type(C_1.mat())}  tamaño = {C_1.mat().sizes}")
    A=PETScMatrix()
    #C_1.mat().matMult(K_petsc.mat()) 
    if stage_m:
        print('C^-1 x K multipli assemble')
        # print("K matrix" , K_.toarray())
        # print("C matrix" , C_.toarray())  
    return(V,u,v,A,C_1,Q_fem,u_n,bcs)

def Q_t(Q_fem,C_1,u_t,v_t,f,bcs):
    Q_v = PETScMatrix()
    C=assemble(Q_fem, tensor=Q_v)
    bcs.apply(Q_v)
    #Q_ = -Q_v.get_local()
    Q__v.mat().scale(-1.0)
    Q_a = PETScMatrix()
    C_1.mat().mult(Q__v.mat(),Q_a.mat()) 
    return Q_a
def exp_solver(A,u_i,n,table,auto,m_u,u=0):
    arnoldi_ti=time.time()
    Beta=np.linalg.norm(u_i)
    if stage_m:
        print('Arnoldi iteration init')
    if auto:
        V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration(A,u_i,real_dt)
    else:
        V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration_m(A,u_i,real_dt,m_u)
    arnoldi_t=time.time()-arnoldi_ti
    if stage_m:
        print(f'KSP projection time = {arnoldi_t}')
    solver_ti=time.time()
    if n==0:
        u = real_dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)+u_n.vector().get_local()
    else:
        u += real_dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)
    solver_t=time.time() -solver_ti
    table.append([arnoldi_t,solver_t,V_m.shape[1],t])
    return(u)
    return(u_i)
def BDF_form(mesh,pconst,flux_method):
    V=FunctionSpace(mesh,"DG",1)
    V_w =VectorFunctionSpace(mesh, "DG", 1)
    w=project(u1, V_w)
    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)
    
    u_n = Function(V)  # Interpola u_D en el espacio de funciones V
    u_nn = Function(V) 
    u_nnn = Function(V)
    
    u_n = project(u_0, V)  # Interpola u_D en el espacio de funciones V
    u_nn = project(u_0, V) 
    u_nnn = project(u_0, V) 
    

    du=pconst[0]*u
    du_n=pconst[1]*u_n
    du_nn=pconst[2]*u_nn
    du_nnn=pconst[3]*u_nnn
    du_t= du+du_n +du_nn +du_nnn

    #bcs =[]
    #bcs.append(DirichletBC(V, Constant(0.0),contorno,2))
    #bcs.append(DirichletBC(V, Constant(0.0),contorno,1))
    bcs =DirichletBC(V, u_analytical, on_boundary,method='geometric') 
    alpha=1
    h= mesh.cells().size
    ds = Measure('ds', domain=mesh, subdomain_data=contorno)
    
    if 'LaxF' == flux_method:
        C = w[0]
        flux = dot(avg(w*u),n('+')) + 0.5*C*jump(u)
    elif 'upwind' == flux_method:
        un = (dot(w, n) + abs(dot(w,n)))/2.0
        flux = jump(un*u)
    
    dif_nflux = -k*dot(jump(u,n),avg(grad(v)))*dS - k*dot(avg(grad(u)),jump(v,n))*dS \
    + (alpha/h)*dot(jump(u,n),jump(v,n))*dS #flujo difusivo
    
    adv_nflux =  dot(jump(v), flux)*dS + dot(v, u_analytical)*ds

    F = du_t*v*dx - real_dt*dot(grad(v),w*u)*dx + real_dt*adv_nflux - real_dt*dot(k*grad(u), grad(v))*dx + real_dt*dif_nflux
    
    #F = du_t*v*dx - Constant(real_dt)*dot(grad(v), w*u)*dx 
    #F += Constant(real_dt)*dot(jump(v), flux)*dS + Constant(real_dt)*dot(v, u_analytical*u)*ds
    
    
    L, R = lhs(F), rhs(F)  # Separa la parte izquierda y derecha de la ecuación
    L_se=assemble(R)
    return(V,L,R,u_n,u_nn,u_nnn,bcs)






if __name__ == "__main__":

    np.set_printoptions(formatter={'float': '{: 0.2F}'.format})
    start_code = time.time()
    dt = 0.1

    ny  = 10 
    nx = ny*10
    solver  = "exp"  # time integration method
    flux_method = "upwind"
    mode_KSP = "40"
    stage_m =False

    # Parámetros de simulación

    T = 10         # Tiempo final
    num_steps = int(round(T/dt,0))# Número de pasos de tiempo
    real_dt= T/num_steps
    k=Constant(0.0001)
    w=Constant((1,0))

    vtkfile_u_exp = XDMFFile(f"results/u_exp.xdmf")
    vtkfile_u_exp.parameters["flush_output"] = True
    vtkfile_u_exp.parameters["rewrite_function_mesh"] = False
    vtkfile_u = XDMFFile(f"results/u_analytical.xdmf")
    vtkfile_u.parameters["flush_output"] = True
    vtkfile_u.parameters["rewrite_function_mesh"] = False
    vtkfile_ubdf = XDMFFile(f"results/u_{solver}.xdmf")
    vtkfile_ubdf.parameters["flush_output"] = True
    vtkfile_ubdf.parameters["rewrite_function_mesh"] = False


    #f=Expression('x[0]*(6*t - pow(x[0], 2) - 4*pow(x[0], 2)*t*t)*exp(-t*pow(x[0], 2))',t=0,degree=2)
    f=Expression('0',t=0,degree=2) 
    #q=Expression('-1',t=0,degree=2) #neumman BC
    u_analytical = Expression('1/(sqrt(1+0.0004*t))*exp(-pow((x[0]-3-t),2)/(1+0.0004*t))',t=0,degree=2)
    u_analytical_bcs = Expression('-1/(sqrt(1+0.0004*t))*exp(-pow((x[0]-3-t),2)/(1+0.0004*t))',t=0,degree=2)
    u_0 = Expression('exp(-pow((x[0]-3),2))',degree=2)
    BDF_coef = {"BDF1":[1,-1,0,0],"BDF2":[3./2,-2,1./2,0.0],"BDF3":[11/6,-3,3/2,-1/3],"BDF_OP":[0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3]}

    # Creación de la malla y definición del espacio de funciones


    start = 0
    x_end = 10
    mesh = RectangleMesh(Point(start, 0),Point(x_end,1),nx,ny,"crossed")
    contorno = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    left().mark(contorno, 1)
    right().mark(contorno, 2)
    print('Number of nodes:', mesh.num_vertices() )
    print('Number of elements:', mesh.num_cells() )
    print('Number of DOFs:',mesh.num_vertices() )

    try:
        os.mkdir(f"results_dt_{real_dt}") 
    except FileExistsError:
        pass
    t= 0
    if solver == "exp":
        C_file= False#sys.argv[5] == "y"
        assemble_ti=time.time()
        V,u_t,v_t,A,C_1,Q_a,u_n,bcs = exp_form(mesh,flux_method,C_file)

        if stage_m:
            print(f"Tiempo de ensamblaje {time.time()- assemble_ti}")
    elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
        V,L,R,u_n,u_nn,u_nnn,bcs = BDF_form(mesh,BDF_coef[solver],flux_method)
    else: 
        print("esquema de integración temporal erroneo")
        exit()


    u_=Function(V)
    L2=[]
    table =[]
    sol=[]
    for n in range(num_steps):
        #f.t=t
        #q.t=t
        t += real_dt
        #u_a.t=t
        u_analytical.t=t
        u_analytical_bcs.t=t 
        print(f'step:{n+1} of {num_steps} time = {t}')
        if solver == "exp":
            if mode_KSP == 'auto': 
                Q=Q_t(Q_a,C_1,u_t,v_t,f,bcs)

                if n ==0:
                    u_i=np.dot(A,u_n.vector().get_local())+Q
                    u =exp_solver(A,u_i,n,table,True,0)
                else:
                    u_i=np.dot(A,np.array(u)[0])+Q
                    u =exp_solver(A,u_i,n,table,True,0,u)
            else: 
                m_u = int(mode_KSP)
                Q=Q_t(Q_a,C_1,u_t,v_t,f,bcs)
                if n ==0:
                    u_i=np.dot(A,u_n.vector().get_local())+Q
                    u =exp_solver(A,u_i,n,table,False,m_u)
                else:
                    u_i=np.dot(A,np.array(u)[0])+Q
                    u =exp_solver(A,u_i,n,table,False,m_u,u)
            u_.vector()[:]=np.array(u)[0]
        elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):

            solve(L == R, u_,bcs)

            u_nnn.assign(u_nn)
            u_nn.assign(u_n)
            u_n.assign(u_)
        u_.rename("u_", "u_");vtkfile_u_exp.write(u_, t)  
        x_cor=np.linspace(start,x_end,100)
        u_p=[]
        u_an=[]
        for i in x_cor:
            u_p.append(u_(i,0.5))
            u_an.append(u_analytical(i,0.5))
        if near(t,real_dt,real_dt/2) or near(t,5,real_dt/2) or near(t,10,real_dt/2):
            if near(t,real_dt,real_dt/2):
                plt.plot(x_cor,u_p,'ko',label=f't = {t:.2f} [s]')
            elif near(t,5,real_dt/2):
                plt.plot(x_cor,u_p,'k^',label=f't = {t:.2f} [s]')
            elif near(t,10,real_dt/2):
                plt.plot(x_cor,u_p,'k*',label=f't = {t:.2f} [s]')
            plt.plot(x_cor,u_an,'b--')

        u_ana = interpolate(u_analytical,V)
        L2_norm=assemble((u_ - u_analytical)**2 * dx)**0.5
        L2.append([L2_norm,t])
        print(f'step:{n+1} of {num_steps} time = {t}, L2 error = {L2_norm}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.ylim(0,1)
        plt.legend(loc='upper left')
    plt.show()