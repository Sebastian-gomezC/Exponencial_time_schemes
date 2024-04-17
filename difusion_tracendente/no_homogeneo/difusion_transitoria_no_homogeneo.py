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
np.set_printoptions(formatter={'float': '{: 0.2F}'.format})
parameters['linear_algebra_backend'] = 'Eigen'
start_code = time.time()
def fi (A,m,tau=1):
    d=scipy.sparse.identity(m)
    exp_A =scipy.linalg.expm(tau*A)
    return (exp_A-d)*spsl.spsolve(tau*A,d) , exp_A
# algoritmo de arnoldi 
def arnoldi_iteration(A, b,tau):
    m = A.shape[0]
    beta= np.linalg.norm(b)
    tol=1E-4
    n=4
    error=1
    #print("beta on er",beta)
    eta=1/sqrt(2)
    H = np.zeros((n + 1, n))
    V = np.zeros((m, n + 1))
    V[:, 0] = b / beta
    j=0
    while error >= tol:
        if n>4 :
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
        if n>4 or (n==4 and j==4):
            fi_m,exp_H =fi(H[0:n, 0:n],n,tau)
            error = beta*abs(H[n,n-1]*tau*e_m.dot(np.array(fi_m.dot(e_1))[0]))
            n += 1
    print(f"H dim= {n-1} error = {error}")
    return V[:,0:n-1], H[0:n-1, 0:n-1],fi_m,e_1,n,exp_H 

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], start) and on_boundary


class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x_end) and on_boundary

def exp_form(mesh):
    boundaries= MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    LeftBoundary().mark(boundaries, 1)  # mark left boundary with a 0
    RightBoundary().mark(boundaries, 2)  # mark right boundary with a 1
    V=FunctionSpace(mesh,"CG",1)
    # Definición de la condición inicial
    u_0 =Expression(('25'),degree=2)

    u_n=project(u_0,V)

    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)

    #bc1 =DirichletBC(V,Constant(20),boundaries,1)
    bc2 =DirichletBC(V,Constant(25),boundaries,2)
    bcs=[bc2]
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    K_fem = k*dot(grad(u), grad(v))*dx # Formulación débil
    Q_fem =v*a*ds(1)

    C_fem=u*v*dx
    #ensamble vector de cargas

    Q_v=assemble(Q_fem)
    [bc.apply(Q_v) for bc in bcs] #penalización
    #ensamble matriz capacitiva 
    C=assemble(C_fem)
    [bc.apply(C) for bc in bcs]#penalización
    #ensamble matriz de rigidez

    K=assemble(K_fem)
    [bc.apply(K) for bc in bcs] #penalización


    #metodo sparse
    K_=scipy.sparse.csr_matrix(-K.array()) 

    N_degree=C.array().shape[0]
    C_=scipy.sparse.csc_matrix(C.array())
    I_n=scipy.sparse.csc_matrix(scipy.sparse.identity(N_degree))
    Q_v=Q_v.get_local()


    C_1=scipy.sparse.linalg.spsolve(C_,I_n)

    #metodo coarse np
    A=C_1.dot(K_)

    A=A.toarray()
    Q=C_1.dot(Q_v)
    u_i=np.dot(A,u_n.vector().get_local())+Q
    return(V,A,Q,u_i,u_n)
def exp_solver(A,u_i,u,n):
    Beta=np.linalg.norm(u_i)
    V_m,H_m,fi_m,e_1,m,exp_H  = arnoldi_iteration(A,u_i,real_dt)
    if n==0:
        u = dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)+u_n.vector().get_local()
    else:
        u += dt*Beta*np.dot(np.dot(V_m,fi_m),e_1.T)
    return(u)
def BDF_form(mesh,pconst):
    boundaries= MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    LeftBoundary().mark(boundaries, 1)  # mark left boundary with a 0
    RightBoundary().mark(boundaries, 2)  # mark right boundary with a 1
    V=FunctionSpace(mesh,"CG",1)
    # Definición de la condición inicial
    u_o =Expression(('25'),degree=2)



    # Definición del problema variacional
    u = TrialFunction(V)  # Función de prueba
    v = TestFunction(V)   # Función de test
    n=FacetNormal(mesh)





    # Definición del valor inicial
    u_n = interpolate(u_o, V)  # Interpola u_D en el espacio de funciones V
    u_nn = interpolate(u_o, V) 
    u_nnn = interpolate(u_o, V) 


    du=pconst[0]*u
    du_n=pconst[1]*u_n
    du_nn=pconst[2]*u_nn
    du_nnn=pconst[3]*u_nnn
    du_t= du+du_n +du_nn +du_nnn
    t= 0

    #bc1 =DirichletBC(V,Constant(20),boundaries,1)
    bc2 =DirichletBC(V,Constant(25),boundaries,2)
    bcs=[bc2]
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    F = du_t*v*dx + real_dt*k*dot(grad(u), grad(v))*dx - real_dt*v*a*ds(1)

    L, R = lhs(F), rhs(F)  # Separa la parte izquierda y derecha de la ecuación
    return(V,L,R,bcs,u_n,u_nn,u_nnn)
def Analytical(x,t):
    def b_n(n):
        return 8/((1-2*n)**2*np.pi**2)
    print('time:', t)
    s_m=1000
    u=[]
    j=0
    for x_n in x:
        u_j=0
        for n in range(1,s_m):
            u_j += b_n(n)*math.exp(-1*((n-0.5)*np.pi)**2*t)*np.cos((n-0.5)*np.pi*x_n)
        u_j += x_n+24
        u.append(u_j)
    return u

if __name__ == "__main__":
    start_code = time.time()
    dt = float(sys.argv[1])
    nx  = int(sys.argv[2])  #nx ny
    solver  = sys.argv[3]  # time integration method
                
    # Parámetros de simulación
    T = 1            # Tiempo final
    num_steps = int(round(T/dt,0))# Número de pasos de tiempo
    real_dt= T/num_steps
    k= 1
    a=-1
    #Esquemas BDF
    BDF_coef = {"BDF1":[1,-1,0,0],"BDF2":[3./2,-2,1./2,0.0],"BDF3":[11/6,-3,3/2,-1/3],"BDF_OP":[0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3]}

    # Creación de la malla y definición del espacio de funciones

    start = 0
    x_end = 1
    mesh = IntervalMesh(nx, start, x_end)

    
    try:
        os.mkdir(f"results_dt_{real_dt}") 
    except FileExistsError:
        pass
    t= 0
    if solver == "exp":
        V,A,Q,u_i,u_n = exp_form(mesh)
        u = u_n.vector().get_local()
    elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
        V,L,R,bcs,u_n,u_nn,u_nnn = BDF_form(mesh,BDF_coef[solver])
    else: 
        print("esquema de integración temporal erroneo")
        exit()
    u_=Function(V)
    
    L2=[]
    
    for n in range(num_steps):
        t += real_dt
        
        print(f'step:{n+1} of {num_steps} time = {t}')
        if solver == "exp":
            u =exp_solver(A,u_i,u,n)
            u_i=np.dot(A,np.array(u)[0])+Q
            x_cor=np.linspace(0.001,0.999,100)
            y=[]
            u_.vector()[:]=np.array(u)[0]
            for i in x_cor:
                y.append(u_(i))

        elif (solver == "BDF1")or(solver == "BDF2") or (solver == "BDF3") or (solver == "BDF_OP"):
            solve(L == R, u_,bcs)
            u_nnn.assign(u_nn)
            u_nn.assign(u_n)
            u_n.assign(u_)

            x_cor=np.linspace(0.001,0.999,100)
            y=[]
            for i in x_cor:
                y.append(u_(i))
        
        y_ana=Analytical(x_cor,t)
        L2_norm = np.sum((np.array(y)-np.array(y_ana))**2)
        L2.append([L2_norm,t])
        plt.plot(x_cor,y,"o",color='black')
        plt.plot(x_cor,y_ana,"-",color='red')
    end_code = time.time()
    plt.show()
    print("execution time: ", end_code-start_code )
    error=pd.DataFrame(L2,columns=[solver,'tiempo'])
    error.to_csv(f'results_dt_{real_dt}/error_scheme_{solver}.csv')



