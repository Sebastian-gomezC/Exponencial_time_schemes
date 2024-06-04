from __future__ import print_function
from fenics import *  # Importa FEniCS, una biblioteca popular para elementos finitos
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
from mshr import *
import sys
import os
from scipy import integrate
from pathlib import Path
np.set_printoptions(formatter={'float': '{: 0.2F}'.format})
parameters['linear_algebra_backend'] = 'Eigen'
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
    tol=1E-16
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
def exp_form(mesh,C_file):
    V=FunctionSpace(mesh,"CG",1)

    u_n=project(u_0,V)

    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)

    bcs =0 #[DirichletBC(V, u_a,contorno,1)] #[DirichletBC(V,Constant(0),contorno,1)]
    # bc2 =DirichletBC(V,Constant(25),contorno,2)
    #bcs =DirichletBC(V, u_a, on_boundary)    # bcs=[bc2]
    # ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    K_fem =0.0001*dot(grad(u), grad(v))*dx +Dx(u, 0)*v*dx   # Formulación débil
    C_fem=u*v*dx

    #ensamble matriz capacitiva 
    C=assemble(C_fem)
    
    #ensamble matriz de rigidez
    K=assemble(K_fem)
    #  #penalización

    if bcs != 0:
        if isinstance(bcs, list):
            [bc.apply(K) for bc in bcs]
            [bc.apply(C) for bc in bcs]#penalización
        else:
            bcs.apply(K)
            bcs.apply(C)
    #metodo sparse
    K_=scipy.sparse.csr_matrix(K.array()) 
    
    if C_file:
        print("reading file")
        C_1=scipy.sparse.load_npz("C_I.npz")
        print("sucesfull read file ")
    else:
        N_degree=C.array().shape[0]
        C_=scipy.sparse.csc_matrix(C.array())
        I_n=scipy.sparse.csc_matrix(scipy.sparse.identity(N_degree))
        C_1=scipy.sparse.linalg.spsolve(C_,I_n)
        scipy.sparse.save_npz("C_I.npz", C_1)
    #metodo coarse np
   
    A=-C_1.dot(K_)
    u_i=u_n.vector().get_local()
    return(V,A,u_i)
def exp_solver(A,u_i,table,auto,m_u=0):
    arnoldi_ti=time.time()
    Beta=np.linalg.norm(u_i)
    if auto:
        V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration(A,u_i,real_dt)
    else:
        V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration_m(A,u_i,real_dt,m_u)

    arnoldi_t=time.time()-arnoldi_ti
    solver_ti=time.time()
    u_i=Beta*np.dot(np.dot(V_m,exp_H),e_1.T)
    solver_t=time.time() -solver_ti
    table.append([arnoldi_t,solver_t,V_m.shape[1]])
    return(u_i)
def BDF_form(mesh,pconst):
    V=FunctionSpace(mesh,"P",1)

    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test

    

    du=pconst[0]*u
    du_n=pconst[1]*u_n
    du_nn=pconst[2]*u_nn
    du_nnn=pconst[3]*u_nnn
    du_t= du+du_n +du_nn +du_nnn


    bcs =0#[DirichletBC(V, u_a, on_boundary)]
    
    ds = Measure('ds', domain=mesh, subdomain_data=contorno)
    F = du_t*v*dx + real_dt*Dx(u, 0)*v*dx + 0.0001*real_dt*dot(grad(u), grad(v))*dx - real_dt*v*f*dx #- real_dt*v*q*ds(1)

    L, R = lhs(F), rhs(F)  # Separa la parte izquierda y derecha de la ecuación
    A1 = assemble(L)
    
    if bcs != 0:
        if isinstance(bcs, list):
            [bc.apply(A1) for bc in bcs]#penalización
        else:
            bcs.apply(A1)
    return(V,L,R,bcs,u_n,u_nn,u_nnn)
if __name__ == "__main__":
    start_code = time.time()
    dt = float(sys.argv[1])
    nx = ny  = int(sys.argv[2])  #nx ny
    solver  = sys.argv[3]  # time integration method
                
    # Parámetros de simulación
    T = math.pi/2            # Tiempo final
    num_steps = int(round(T/dt,0))# Número de pasos de tiempo
    real_dt= T/num_steps
    #Esquemas BDF
    BDF_coef = {"BDF1":[1,-1,0,0],"BDF2":[3./2,-2,1./2,0.0],"BDF3":[11/6,-3,3/2,-1/3],"BDF_OP":[0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3]}

    # Creación de la malla y definición del espacio de funciones

    start = 0
    x_end = 20
    mesh = IntervalMesh(nx, start, x_end)  # Crea una malla cuadrada unitaria
    
    u_analytical=Expression('1/(sqrt(1+0.0004*t))*exp(-pow((x[0]-t),2)/(1+0.0004*t))',t=0,degree=2)
    u_0 = Expression('exp(-pow((x[0]),2))',degree=2)
    
    h = CellDiameter(mesh)
    DG = FunctionSpace(mesh, "DG", 0)
    CFL = project(2*sqrt(inner(w,w))*real_dt/h, DG)
    maxCFL =round(CFL.vector().max(),2) 
    vtkfile_u_exp = XDMFFile(f"results/u_exp.xdmf")
    vtkfile_u_exp.parameters["flush_output"] = True
    vtkfile_u_exp.parameters["rewrite_function_mesh"] = False
    vtkfile_u = XDMFFile(f"results/u_analytical.xdmf")
    vtkfile_u.parameters["flush_output"] = True
    vtkfile_u.parameters["rewrite_function_mesh"] = False
    vtkfile_ubdf = XDMFFile(f"results/u_{solver}.xdmf")
    vtkfile_ubdf.parameters["flush_output"] = True
    vtkfile_ubdf.parameters["rewrite_function_mesh"] = False
    try:
        os.mkdir(f"results_dt_{maxCFL}") 
    except FileExistsError:
        pass
    t= 0
    if solver == "exp":
        assemble_ti=time.time()
        V,A,u_i = exp_form(mesh)
        assemble_t =time.time()- assemble_ti
    elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
        V,L,R,u_n,u_nn,u_nnn = BDF_form(mesh,BDF_coef[solver])
    else: 
        print("esquema de integración temporal erroneo")
        exit()
    
    u_=Function(V)
    
    L2=[]
    table =[]
    for n in range(num_steps):
        t += real_dt
        u_analytical.t=t
        u_ana = interpolate(u_analytical,V)
        print(f'step:{n+1} of {num_steps} time = {t}')
        if solver == "exp":
            if sys.argv[4] == 'auto':
                u_i =exp_solver(A,u_i,table,True)
                u_.vector()[:]=u_i
            else:
                m_u = int(sys.argv[4])
                u_i =exp_solver(A,u_i,table,False,m_u)
                u_.vector()[:]=u_i
            u_.rename("u_exp", "u_exp");vtkfile_u_exp.write(u_, t)
        elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
            solve(L == R, u_)
            u_nnn.assign(u_nn)
            u_nn.assign(u_n)
            u_n.assign(u_)
            # u_.rename("u_BDF", "u_BDF");vtkfile_ubdf.write(u_, t)
        
        u_ana.rename("u_a", "u_a");vtkfile_u.write(u_ana, t)
        L2_norm = errornorm(u_ana,u_)
        L2.append([L2_norm,t])
        print('error norm',L2_norm,'cfl ',maxCFL )

    end_code = time.time()
    exec_t=end_code-start_code
    print("execution time: ", exec_t )
    if   (solver == "exp") and (not (sys.argv[4] == 'auto')):
        L2=np.array(L2)
        solution=[sys.argv[4],np.linalg.norm(L2[:,0]),exec_t]
        solution =pd.DataFrame([solution],columns=['krylov_dim','l2_norm_sum','cpu_time'])
        Hdim_path=Path(f"Hdim_error_{solver}.csv")
        if not Hdim_path.is_file():
            solution.to_csv(f"Hdim_error_{solver}.csv",index=False)
        else : 
            histor=pd.read_csv(f'Hdim_error_{solver}.csv')
            update=pd.concat([histor,solution],sort=False)
            update.to_csv(f"Hdim_error_{solver}.csv",index=False)
    else:
        error=pd.DataFrame(L2,columns=[solver,'tiempo'])
        error.to_csv(f'results_dt_{maxCFL}/error_scheme_{solver}.csv')
        time_path=Path(f"time_error_{solver}.csv")
        solution=[np.linalg.norm(error[solver].to_numpy()),maxCFL, exec_t ]
        solution =pd.DataFrame([solution],columns=['L2_norm_sum','max_CFL_number','cpu_time'])
        if not time_path.is_file():
            solution.to_csv(f"time_error_{solver}.csv",index=False)
        else : 
            histor=pd.read_csv(f'time_error_{solver}.csv')
            update=pd.concat([histor,solution],sort=False)
            update.to_csv(f"time_error_{solver}.csv",index=False)
        
        
        if solver == "exp":
            exp_path=Path('exp_times.csv')
            time_sta =pd.DataFrame(table,columns=['krylov_t','solver_t','m'])
            exp_dat=pd.DataFrame([[time_sta['krylov_t'].mean(),time_sta['solver_t'].mean(),exec_t,time_sta['m'].mean()]],columns=['mean_kylov_projection_time','mean_solver_time','cpu_time','mean_krylov_dimension_$m$'])
            print(time_sta['krylov_t'].mean(),time_sta['solver_t'].mean(),exec_t,time_sta['m'].mean())
            if  not exp_path.is_file():
                exp_dat.to_csv(f"exp_times.csv",index=False)
            else : 
                exp_h=pd.read_csv(f'exp_times.csv')
                exp_dat=pd.concat([exp_h,exp_dat],sort=False)
                exp_dat.to_csv(f"exp_times.csv",index=False)






