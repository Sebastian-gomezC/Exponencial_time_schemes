from __future__ import print_function
from fenics import *  # Importa FEniCS, una biblioteca popular para elementos finitos
import numpy as np  # NumPy para operaciones numéricas
import matplotlib.pyplot as plt  # Matplotlib para gráficos
from matplotlib.legend_handler import HandlerLine2D
from label_lines import *
import math
import scipy
import time
import scipy.sparse as sps
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
    tol=1E-4
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
def exp_form(mesh):
    V=FunctionSpace(mesh,"CG",1)

    u_n=project(u_0,V)

    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)

    #bc1 =DirichletBC(V,Constant(20),boundaries,1)
    # bc2 =DirichletBC(V,Constant(25),boundaries,2)
    bcs =DirichletBC(V, bc_2, on_boundary)    # bcs=[bc2]
    # ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    K_fem = dot(grad(u), grad(v))*dx # Formulación débil
    C_fem=u*v*dx
    ds = Measure('ds', domain=mesh, subdomain_data=contorno)
    Q_fem =v*f*dx #+v*q*ds(1)
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
    K_=scipy.sparse.csr_matrix(-K.array()) 
    N_degree=C.array().shape[0]
    C_=scipy.sparse.csc_matrix(C.array())
    I_n=scipy.sparse.csc_matrix(scipy.sparse.identity(N_degree))
    C_1=scipy.sparse.linalg.spsolve(C_,I_n)

    #metodo coarse np
    A=C_1.dot(K_)
    A=A.toarray()
    # print("K matrix" , K_.toarray())
    # print("C matrix" , C_.toarray())  
    return(V,u,v,A,C_1,Q_fem,u_n,bcs)

def Q_t(Q_fem,C_1,u_t,v_t,f,bcs):

    Q_v=assemble(Q_fem)
    if bcs != 0:
        if isinstance(bcs, list):
            [bc.apply(Q_v) for bc in bcs] #penalización
        else:
            bcs.apply(Q_v)
    Q_ = Q_v.get_local()
    Q_a =C_1.dot(Q_)
    return Q_a
def exp_solver(A,u_i,n,table,auto,m_u,u=0):
    arnoldi_ti=time.time()
    Beta=np.linalg.norm(u_i)
    if auto:
        V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration(A,u_i,real_dt)
    else:
        V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration_m(A,u_i,real_dt,m_u)
    arnoldi_t=time.time()-arnoldi_ti
    solver_ti=time.time()
    if n==0:
        u = real_dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)+u_n.vector().get_local()
    else:
        u += real_dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)
    solver_t=time.time() -solver_ti
    table.append([arnoldi_t,solver_t,V_m.shape[1]])
    return(u)

def BDF_form(mesh,pconst):
    V=FunctionSpace(mesh,"P",1)

    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)

    
    # Definición del valor inicial
    u_n = Function(V)
    u_n = interpolate(u_0, V)  # Interpola u_D en el espacio de funciones V
    u_nn = Function(V)
    u_nn = project(u_0, V) 
    u_nnn = Function(V)
    u_nnn = project(u_0, V) 


    du=pconst[0]*u
    du_n=pconst[1]*u_n
    du_nn=pconst[2]*u_nn
    du_nnn=pconst[3]*u_nnn
    du_t= du+du_n +du_nn +du_nnn


    bcs =DirichletBC(V, bc_2, on_boundary)
    
    ds = Measure('ds', domain=mesh, subdomain_data=contorno)
    F = du_t*v*dx + real_dt*dot(grad(u), grad(v))*dx  - real_dt*v*f*dx #- real_dt*v*q*ds(1)

    L, R = lhs(F), rhs(F)  # Separa la parte izquierda y derecha de la ecuación
    A1 = assemble(L)
    
    if bcs != 0:
        if isinstance(bcs, list):
            [bc.apply(A1) for bc in bcs]#penalización
        else:
            bcs.apply(A1)
    return(V,A1,R,bcs,u_n,u_nn,u_nnn)

    
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2F}'.format})
    start_code = time.time()
    dt = float(sys.argv[1])
    nx = ny = int(sys.argv[2])  #nx ny
    solver  = sys.argv[3]  # time integration method
                
    # Parámetros de simulación
    T = 1            # Tiempo final
    num_steps = int(round(T/dt,0))# Número de pasos de tiempo
    real_dt= T/num_steps

    #f=Expression('x[0]*(6*t - pow(x[0], 2) - 4*pow(x[0], 2)*t*t)*exp(-t*pow(x[0], 2))',t=0,degree=2)
    f=Expression('-1',t=0,degree=2) 
    #f=Expression('4',t=0,degree=2) 
    #q=Expression('-1',t=0,degree=2) #neumman BC


    u_a = Expression('pow(x[0],2) +t',t=0,degree=2)
    u_0 = Expression('pow(x[0],2)',degree=2)
    #bc_1= Constant(0.0) #left
    bc_2= Expression('pow(x[0],2) +t',t=0,degree=2)
    BDF_coef = {"BDF1":[1,-1,0,0],"BDF2":[3./2,-2,1./2,0.0],"BDF3":[11/6,-3,3/2,-1/3],"BDF_OP":[0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3]}

    # Creación de la malla y definición del espacio de funciones

    start = 0
    x_end = 1
    mesh = IntervalMesh(nx, start, x_end)
    contorno = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    left().mark(contorno, 1)
    right().mark(contorno, 2)


    # vtkfile_u = XDMFFile(f"results_dt_{dt}/u_analytical.xdmf")
    # vtkfile_u.parameters["flush_output"] = True
    # vtkfile_u.parameters["rewrite_function_mesh"] = False
    # vtkfile_uh = XDMFFile(f"results_dt_{dt}/u_{solver}.xdmf")
    # vtkfile_uh.parameters["flush_output"] = True
    # vtkfile_uh.parameters["rewrite_function_mesh"] = False

    try:
        os.mkdir(f"results_dt_{real_dt}") 
    except FileExistsError:
        pass
    t= 0
    if solver == "exp":
        assemble_ti=time.time()
        V,u_t,v_t,A,C_1,Q_a,u_n,bcs = exp_form(mesh)
        assemble_t =time.time()- assemble_ti
    elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
        V,L,R,bcs,u_n,u_nn,u_nnn = BDF_form(mesh,BDF_coef[solver])
    else: 
        print("esquema de integración temporal erroneo")
        exit()
    u_=Function(V)

    L2=[]
    table =[]
    for n in range(num_steps):
        f.t=t
        #q.t=t
        t += real_dt
        bc_2.t=t
        print(f'step:{n+1} of {num_steps} time = {t}')
        if solver == "exp":
            if sys.argv[4] == 'auto': 
                Q=Q_t(Q_a,C_1,u_t,v_t,f,bcs)
                if n ==0:
                    u_i=np.dot(A,u_n.vector().get_local())+Q
                    u =exp_solver(A,u_i,n,table,True,0)
                else:
                    u_i=np.dot(A,np.array(u)[0])+Q
                    u =exp_solver(A,u_i,n,table,True,0,u)
            else: 
                m_u = int(sys.argv[4])
                Q=Q_t(Q_a,C_1,u_t,v_t,f,bcs)
                if n ==0:
                    u_i=np.dot(A,u_n.vector().get_local())+Q
                    u =exp_solver(A,u_i,n,table,False,m_u)
                else:
                    u_i=np.dot(A,np.array(u)[0])+Q
                    u =exp_solver(A,u_i,n,table,False,m_u,u)
            u_.vector()[:]=np.array(u)[0]
        elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
            Q = assemble(R)
            if bcs != 0:
                if isinstance(bcs, list):
                    [bc.apply(Q) for bc in bcs]#penalización
                else:
                    bcs.apply(Q)
                solve(L, u_.vector(),Q)
            u_nnn.assign(u_nn)
            u_nn.assign(u_n)
            u_n.assign(u_)
        
    #     u_.rename("u_"+solver, "u_"+solver);vtkfile_uh.write(u_, t)
        #u_an=project(u_a,V)
    #     u_an.rename("u", "u");vtkfile_u.write(u_an, t) 
        x_cor=np.linspace(0.001,0.999,100)
        u_a.t=t
        
            
        u_p=[]
        u_an=[]
        for i in x_cor:
            u_p.append(u_(i))
            u_an.append(u_a(i))
        if near(t,0.1,real_dt/2) or near(t,0.5,real_dt/2) or near(t,1,real_dt/2) :
            plt.plot(x_cor,u_p)
            plt.plot(x_cor,u_an,'--')
        L2_norm=np.sum((np.array(u_an)-np.array(u_p))**2)**0.5 #assemble((u_ - u_an)**2 * dx)**0.5
        L2.append([L2_norm,t])
    plt.grid(True)
    plt.xlim = [0,1]
    plt.ylim = [0,6]
    plt.show()    
        
    # end_code = time.time()
    # exec_t=end_code-start_code
    # print("execution time: ", exec_t )
    # if   (solver == "exp") and (not (sys.argv[4] == 'auto')):
    #     L2=np.array(L2)
    #     solution=[sys.argv[4],np.linalg.norm(L2[:,0]),exec_t]
    #     solution =pd.DataFrame([solution],columns=['krylov_dim','l2_norm_sum','cpu_time'])
    #     Hdim_path=Path(f"Hdim_error_{solver}.csv")
    #     if not Hdim_path.is_file():
    #         solution.to_csv(f"Hdim_error_{solver}.csv",index=False)
    #     else : 
    #         histor=pd.read_csv(f'Hdim_error_{solver}.csv')
    #         update=pd.concat([histor,solution],sort=False)
    #         update.to_csv(f"Hdim_error_{solver}.csv",index=False)

    # else:  
    #     error=pd.DataFrame(L2,columns=[solver,'tiempo'])
    #     error.to_csv(f'results_dt_{real_dt}/error_scheme_{solver}.csv')
        
    #     time_path=Path(f"time_error_{solver}.csv")

    #     solution=[np.linalg.norm(error[solver].to_numpy()),real_dt, exec_t ]
    #     solution =pd.DataFrame([solution],columns=['L2_norm_sum','dt','cpu_time'])
        
    #     if not time_path.is_file():
    #         solution.to_csv(f"time_error_{solver}.csv",index=False)
    #     else : 
    #         histor=pd.read_csv(f'time_error_{solver}.csv')
    #         update=pd.concat([histor,solution],sort=False)
    #         update.to_csv(f"time_error_{solver}.csv",index=False)
        
        
    #     if solver == "exp":
    #         exp_path=Path('exp_times.csv')
    #         time_sta =pd.DataFrame(table,columns=['krylov_t','solver_t','m'])
    #         exp_dat=pd.DataFrame([[time_sta['krylov_t'].mean(),time_sta['solver_t'].mean(),exec_t,time_sta['m'].mean()]],columns=['mean_kylov_projection_time','mean_solver_time','cpu_time','mean_krylov_dimension_$m$'])
    #         print(time_sta['krylov_t'].mean(),time_sta['solver_t'].mean(),exec_t,time_sta['m'].mean())
    #         if  not exp_path.is_file():
    #             exp_dat.to_csv(f"exp_times.csv",index=False)
    #         else : 
    #             exp_h=pd.read_csv(f'exp_times.csv')
    #             exp_dat=pd.concat([exp_h,exp_dat],sort=False)
    #             exp_dat.to_csv(f"exp_times.csv",index=False)
