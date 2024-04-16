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

from mshr import *
start_code = time.time()
flag ="R"
parameters['linear_algebra_backend'] = 'Eigen'
np.set_printoptions(formatter={'float': '{: 0.2F}'.format})
def tran2SparseMatrix(A):
    row, col, val = as_backend_type(A).data()
    return sps.csr_matrix((val, col, row))
# mempool = cupy.get_default_memory_pool()

# with cupy.cuda.Device(0):
#     mempool.set_limit(size=4*1024**3)
    
#np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
# Parámetros de simulación
T = math.pi/2            # Tiempo final
num_steps = 20# Número de pasos de tiempo
dt = T / num_steps # Tamaño del paso de tiempo

# Creación de la malla y definición del espacio de funciones
nx = ny =50# Número de divisiones en x e y
#R=Rectangle(Point(-0.5,-0.5),Point(0.5,0.5))
#mesh=generate_mesh(R,64)
mesh = RectangleMesh(Point(-0.5,-0.5),Point(0.5,0.5),nx,ny)  # Crea una malla cuadrada unitaria
V = FunctionSpace(mesh, 'CG', 1) # Define el espacio de funciones con elementos lineales
V_vec = VectorFunctionSpace(mesh, 'CG',2)


w =Expression(('-4*x[1]','4*x[0]'),degree=2)
w = interpolate(w,V_vec)
# Definición de la condición de frontera
u_o = Expression('exp(-(pow(x[0]+0.25,2)+pow(x[1],2))/(0.004))',
                 degree=2)  # Expresión para u_D

u_analytical=Expression('(1.0/(1.0+0.1*t))*exp(-(pow(x[0]*cos(4*t)+x[1]*sin(4*t)+0.25,2)+pow(-x[0]*sin(4*t)+x[1]*cos(4*t),2))/(0.004*(1+0.1*t)))',t=0,degree=1)
class MyExpression0(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self._t=t
    def eval(self, value, x,):
        x1_bar = x[0]*cos(4*self._t)+x[1]*sin(4*self._t)
        x2_bar = -x[0]*sin(4*self._t)+x[1]*cos(4*self._t)
        value[0] = (1.0/(1.0+0.1*self._t))*exp(-(pow(x1_bar+0.25,2)+pow(x2_bar,2))/(0.004*(1+0.1*self._t)))


# Definición del valor inicial
u_n = interpolate(u_o, V)  # Interpola u_D en el espacio de funciones V

# Definición del problema variacional
u = TrialFunction(V)  # Función de prueba
v = TestFunction(V)   # Función de test




K_fem =0.0001*dot(grad(u), grad(v))*dx +div(w*u)*v*dx  # Formulación débil
C_fem=u*v*dx

def fi (A,m,tau=1):
    d=scipy.sparse.identity(m)
    
    return (scipy.linalg.expm(tau*A)-d)*scipy.sparse.linalg.spsolve(tau*A,d)
# algoritmo de arnoldi 
def arnoldi_iteration_error(A, b,tau):
    m = A.shape[0]
    beta= np.linalg.norm(b)
    tol=1E-12
    n=1
    error=1
    it_max=100
    #print("beta on er",beta)
    while error > tol:
        n += 1
        e_m=np.zeros(n)
        e_m[n-1]=1
        e_1=np.zeros(n)
        e_1[0]=1
        h = np.zeros((n + 1, n))
        Q = np.zeros((m, n + 1))
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
        fi_m=fi(h[0:n, 0:n],n,tau)
        error = beta*abs(h[n,n-1]*tau*e_m.dot(np.array(fi_m.dot(e_1))[0]))
    print(f"H dim= {n} error = {error}")
    return Q[:,0:n], h[0:n, 0:n],fi_m,e_1,n

# proceso 
vtkfile_u_exp = XDMFFile(f"results_dt{dt}/u_exp.xdmf")
vtkfile_u_exp.parameters["flush_output"] = True
vtkfile_u_exp.parameters["rewrite_function_mesh"] = False
vtkfile_u = XDMFFile(f"results_dt{dt}/u_analitical.xdmf")
vtkfile_u.parameters["flush_output"] = True
vtkfile_u.parameters["rewrite_function_mesh"] = False
vtkfile_ubdf = XDMFFile(f"results_dt{dt}/u_BDF.xdmf")
vtkfile_ubdf.parameters["flush_output"] = True
vtkfile_ubdf.parameters["rewrite_function_mesh"] = False
t= 0
u_=Function(V)


K= assemble(lhs(K_fem))# Separa la parte izquierda y derecha de la ecuación
#bc.apply(K)
K_=scipy.sparse.csr_matrix(K.array()) 
C=assemble(C_fem)
N_degree=C.array().shape[0]
#C_c = tran2SparseMatrix(C)
#C_c = cupyx.scipy.sparse.csr_matrix(C_c)
#bc.apply(C)


#I_np=np.identity(N_degree)
#I_cupy=cupyx.scipy.sparse.identity(N_degree)
# A1 = assemble(L)
# [bc.apply(A1) for bc in bcs]


# b1 = assemble(R)
# [bc.apply(b1) for bc in bcs]
# b1= b1[:]
# b1 = cupy.array(b1)
# X_w.vector()[:] = cupy.asnumpy(linalge.spsolve(A1, b1))
    
C=scipy.sparse.csc_matrix(C.array().astype(np.float32))
I_n=scipy.sparse.csc_matrix(scipy.sparse.identity(N_degree))

# 


start=time.time()
C_1=scipy.sparse.linalg.spsolve(C,I_n)
#C_1=np.linalg.solve(C.array(),I_np)
#C_1 = cupyx.scipy.sparse.linalg.spsolve(C_c, I_cupy)

A=-C_1.dot(K_)
m=9
v_hat=np.zeros(m)
v_hat[0]=1


u_n = interpolate(u_analytical, V)
u_i=u_n.vector().get_local()


#BDF
u_n = interpolate(u_o, V)  # Interpola u_D en el espacio de funciones V
u_nn = interpolate(u_o, V) 
u_nnn = interpolate(u_o, V) 
# Definición del problema variacional
u = TrialFunction(V)  # Función de prueba
v = TestFunction(V)   # Función de test

# Formulación del problema variacional
#pconst=[3./2,-2,1./2,0.0] #bdf2
#pconst = [0.48*11/6+0.52*3/2,0.48*-3+0.52*-2,0.48*3/2+0.52*1/2,0.48*-1/3] #bdf2 op
pconst= [11/6,-3,3/2,-1/3] #bdf 3
#pconst=[1,-1,0,0] #bdf1

du=pconst[0]*u
du_n=pconst[1]*u_n
du_nn=pconst[2]*u_nn
du_nnn=pconst[3]*u_nnn
du_t= du+du_n +du_nn +du_nnn
u_BDF=Function(V)
F= du_t*v*dx + dt*0.0001*dot(grad(u), grad(v))*dx +dt*(div(w*u))*v*dx  # Formulación débil
a, L = lhs(F), rhs(F)  # Separa la parte izquierda y derecha de la ecuación
for n in range(num_steps):
    t += dt
    print(f'step:{n+1} of {num_steps}')
    Beta=np.linalg.norm(u_i)
    V_m,H_m,fi_m,e_1,m= arnoldi_iteration_error(A,u_i,dt)
    u_i=Beta*np.dot(np.dot(V_m,scipy.linalg.expm(dt*H_m)),e_1.T)
    #u_i=np.dot(scipy.linalg.expm(dt*A),u_i)
    u_.vector()[:]=u_i
    u_.rename("u_exp", "u_exp");vtkfile_u_exp.write(u_, t)
    
    u_analytical.t=t
    u_ana = project(u_analytical,V)
    u_ana.rename("u_an", "u_an");vtkfile_u.write(u_ana, t)

    solve(a == L, u_BDF)
    u_nnn.assign(u_nn)
    u_nn.assign(u_n)
    u_n.assign(u_BDF)
    u_n.rename("u_BDF", "u_BDF");vtkfile_ubdf.write(u_n, t)

end_code = time.time()
