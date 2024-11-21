
from dolfinx import *  # Importa FEniCS, una biblioteca popular para elementos finitos
import importlib.util

import numpy as np  # NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Matplotlib para gráficos
from matplotlib.legend_handler import HandlerLine2D
from label_lines import * #graficas 
from petsc4py import PETSc
from slepc4py import SLEPc
import scipy
import time
import scipy.sparse.linalg as spsl
import pandas as pd
from mshr import *
import sys
import os
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
    table.append([arnoldi_t,solver_t,V_m.shape[1],t])
    return(u_i)
def BDF_form(mesh,pconst):
    V=FunctionSpace(mesh,"P",1)

    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    u_n = interpolate(u_0, V)  # Interpola u_D en el espacio de funciones V
    u_nn = interpolate(u_0, V) 
    u_nnn = interpolate(u_0, V) 
    

    du=pconst[0]*u
    du_n=pconst[1]*u_n
    du_nn=pconst[2]*u_nn
    du_nnn=pconst[3]*u_nnn
    du_t= du+du_n +du_nn +du_nnn


    bcs =0#[DirichletBC(V, u_a, on_boundary)]
    

    F = du_t*v*dx + real_dt*Dx(u, 0)*v*dx + 0.0001*real_dt*dot(grad(u), grad(v))*dx

    L, R = lhs(F), rhs(F)  # Separa la parte izquierda y derecha de la ecuación
    A1 = assemble(L)
    
    if bcs != 0:
        if isinstance(bcs, list):
            [bc.apply(A1) for bc in bcs]#penalización
        else:
            bcs.apply(A1)
    return(V,L,R,u_n,u_nn,u_nnn)
if __name__ == "__main__":
    start_code = time.time()
    dt = float(sys.argv[1])
    nx = ny  = int(sys.argv[2])  #nx ny
    solver  = sys.argv[3]  # time integration method
                
    # Parámetros de simulación
    T = 10         # Tiempo final
    num_steps = int(round(T/dt,0))# Número de pasos de tiempo
    real_dt= T/num_steps
    #f=Expression('x[0]*(6*t - pow(x[0], 2) - 4*pow(x[0], 2)*t*t)*exp(-t*pow(x[0], 2))',t=0,degree=2)
    #f=Expression('4',t=0,degree=2) 
    #q=Expression('-1',t=0,degree=2) #neumman BC
    u_analytical = Expression('1/(sqrt(1+0.0004*t))*exp(-pow((x[0]-3-t),2)/(1+0.0004*t))',t=0,degree=2)

    u_0 = Expression('exp(-pow((x[0]-3),2))',degree=2)
    BDF_coef = {"BDF1":[1,-1,0,0],"BDF2":[3./2,-2,1./2,0.0],"BDF3":[11/6,-3,3/2,-1/3],"BDF_OP":[0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3]}

    # Creación de la malla y definición del espacio de funciones

    start = 0
    x_end = 20
    mesh = IntervalMesh(nx, start, x_end)  # Crea una malla cuadrada unitaria
    

    start = 0
    x_end = 20
    mesh = IntervalMesh(nx, start, x_end)
    contorno = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    left().mark(contorno, 1)
    right().mark(contorno, 2)
   

    try:
        os.mkdir(f"results_dt_{dt}") 
    except FileExistsError:
        pass
    t= 0
    if solver == "exp":
        C_file= sys.argv[5] == "y"
        assemble_ti=time.time()
        V,A,u_i = exp_form(mesh,C_file)
        assemble_t =time.time()- assemble_ti
    elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
        V,L,R,u_n,u_nn,u_nnn = BDF_form(mesh,BDF_coef[solver])
    else: 
        print("esquema de integración temporal erroneo")
        exit()
    
    u_=Function(V)
    
    L2=[]
    table =[]
    sol=[]
    plt.figure(figsize=(12,6))
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
            # u_.rename("u_exp", "u_exp");vtkfile_u_exp.write(u_, t)
        elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
            solve(L == R, u_)
            u_nnn.assign(u_nn)
            u_nn.assign(u_n)
            u_n.assign(u_)
            # u_.rename("u_BDF", "u_BDF");vtkfile_ubdf.write(u_, t)
        
        x_cor=np.linspace(start,x_end,100)

        # u_.rename("u_"+solver, "u_"+solver);vtkfile_uh.write(u_, t)
        # u_a_=project(u_a,V)
        # u_a_.rename("u", "u");vtkfile_u.write(u_a_, t) 

    #     u_p=[]
    #     u_an=[]
    #     for i in x_cor:
    #         u_p.append(u_(i))
    #         u_an.append(u_analytical(i))
    #     if near(t,real_dt,real_dt/2) or near(t,5,real_dt/2) or near(t,10,real_dt/2):
    #         if near(t,real_dt,real_dt/2):
    #             plt.plot(x_cor,u_p,'ko',label=f't = {t:.2f} [s]')
    #         elif near(t,5,real_dt/2):
    #             plt.plot(x_cor,u_p,'k^',label=f't = {t:.2f} [s]')
    #         elif near(t,10,real_dt/2):
    #             plt.plot(x_cor,u_p,'k*',label=f't = {t:.2f} [s]')
    #         plt.plot(x_cor,u_an,'b--')
    #     L2_norm=assemble((u_ - u_analytical)**2 * dx)**0.5
    #     L2.append([L2_norm,t])
    # plt.xlabel('x')
    # plt.ylabel('u')
    # plt.legend(loc='upper left')
    # plt.savefig(f'results_dt_{real_dt}/solution_{solver}.png',dpi=300, bbox_inches='tight')
    # sol=pd.DataFrame({f'{solver}': u_p, 'Analytical':u_an,'x':x_cor}) 
    
    # Hdim_path=Path(f'results_dt_{real_dt}/solution.csv')
    # if not Hdim_path.is_file():
    #     sol.to_csv(f'results_dt_{real_dt}/solution.csv',index=False)
    # else : 
    #     histor=pd.read_csv(f'results_dt_{real_dt}/solution.csv')
    #     update=pd.concat([histor,sol[f'{solver}']],sort=False)
    #     update.to_csv(f"results_dt_{real_dt}/solution.csv",index=False)

    end_code = time.time()
    exec_t=end_code-start_code
    print("execution time: ", exec_t )
    if   (solver == "exp") and (not (sys.argv[4] == 'auto')):
        L2=np.array(L2)
        solution=[sys.argv[4],L2[:,0].max(),exec_t]
        solution =pd.DataFrame([solution],columns=['krylov_dim','l2_norm_sum','cpu_time'])
        Hdim_path=Path(f"Hdim_error_{solver}.csv")
        if not Hdim_path.is_file():
            solution.to_csv(f"Hdim_error_{solver}.csv",index=False)
        else : 
            histor=pd.read_csv(f'Hdim_error_{solver}.csv')
            update=pd.concat([histor,solution],sort=False)
            update.to_csv(f"Hdim_error_{solver}.csv",index=False)
        
        error=pd.DataFrame(L2,columns=[solver+'_'+sys.argv[4],'tiempo'])
        error.to_csv(f'results_dt_{real_dt}/error_scheme_{solver}_H_{sys.argv[4]}.csv')
        time_path=Path(f"time_error_{solver}.csv")
        solution=[error[solver+'_'+sys.argv[4]].to_numpy().max() ,real_dt, exec_t ]
        solution =pd.DataFrame([solution],columns=['L2_norm_sum','dt','cpu_time'])
        if not time_path.is_file():
            solution.to_csv(f"time_error_{solver}.csv",index=False)
        else : 
            histor=pd.read_csv(f'time_error_{solver}.csv')
            update=pd.concat([histor,solution],sort=False)
            update.to_csv(f"time_error_{solver}.csv",index=False)
    else:
        error=pd.DataFrame(L2,columns=[solver,'tiempo'])

        time_path=Path(f"time_error_{solver}.csv")
        solution=[error[solver].to_numpy().max(),real_dt, exec_t ]
        solution =pd.DataFrame([solution],columns=['L2_norm_sum','dt','cpu_time'])
        
        
        if solver == "exp":
            se =0
            for col in ['krylov_time','solver_time','dim H_m','time']:
                error[col] =np.array(table)[:,se]
                se +=1
        error.to_csv(f'results_dt_{real_dt}/error_scheme_{solver}.csv')               
        if not time_path.is_file():
            solution.to_csv(f"time_error_{solver}.csv",index=False)
        else : 
            histor=pd.read_csv(f'time_error_{solver}.csv')
            update=pd.concat([histor,solution],sort=False)
            update.to_csv(f"time_error_{solver}.csv",index=False)
        

        






