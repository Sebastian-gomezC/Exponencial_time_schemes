import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

types=['BDF1','BDF2','BDF3','exp']
dt=0.01
CFL=3.4
error=pd.DataFrame()
df1 = pd.read_csv(f'results_dt_{dt}/error_scheme_{'BDF1'}.csv')
error = pd.concat([error,df1["tiempo"]],axis=1)
for scheme in types:
    df1 = pd.read_csv(f'results_dt_{dt}/error_scheme_{scheme}.csv')
    error = pd.concat([error,df1[scheme]],axis=1)

plot = error.plot(x='tiempo',xlim=[0,np.pi/2],ylim=[1E-8,1],title=f"Error convergence for CFL = {CFL}",logy=True,grid=True,xlabel='time',ylabel='L2 norm')
plt.grid(True,which="both",alpha=0.3, ls='-')
plt.savefig(f'L2norm_CFL_{CFL}.png',dpi=300)

