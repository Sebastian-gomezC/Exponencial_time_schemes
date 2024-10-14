import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys
import math
types=['BDF1','BDF2','exp']
type_plot = sys.argv[1]
if type_plot == "t-L2":
    dt=float(sys.argv[2])
    error=pd.DataFrame()
    df1 = pd.read_csv(f'results_dt_{dt}/error_scheme_BDF1.csv')
    error = pd.concat([error,df1["tiempo"]],axis=1)
    for scheme in types:
        df1 = pd.read_csv(f'results_dt_{dt}/error_scheme_{scheme}.csv')
        error = pd.concat([error,df1[scheme]],axis=1)

    plot = error.plot(x='tiempo',xlim=[0,10],ylim=[1e-5,1],title=f"Error convergence for dt = {dt}",logy=True,grid=True,xlabel='time',ylabel='L2 norm')
    plt.grid(True,which="both",alpha=0.3, ls='-')
    plt.savefig(f'L2norm_dt_{dt}.png',dpi=300)
elif type_plot == "dt-sumL2":
    error=pd.DataFrame()
    it = 0
    for scheme in types:
        df1 = pd.read_csv(f'time_error_{scheme}.csv')
        df1=df1.rename(columns={"L2_norm_sum": scheme})
        if it == 0:
            error = pd.concat([error,df1[scheme],df1['dt']],sort=True,axis=1)
        else:            
            error = pd.concat([error,df1[scheme]],sort=True,axis=1)
        it += 1
    print(error)
    plot = error.plot(x='dt',title=f"Error convergence ",loglog=True,grid=True,xlabel='time',ylabel='$L_\infty$ norm',marker='o')
    plt.grid(True,which="both",alpha=0.3, ls='-')
    plt.xlabel("$dt$ ",fontsize=15)
    plt.ylabel("$L_\infty$ norm",fontsize=15)
    plt.savefig(f'dt-L2.png',dpi=300)
elif type_plot == "cpu-sumL2":
    error=pd.DataFrame()
    ig, ax = plt.subplots()
    for scheme in types:
        df1 = pd.read_csv(f'time_error_{scheme}.csv')
        df1=df1.rename(columns={"L2_norm_sum": scheme, "cpu_time": "cpu_time "+scheme})            
        error = pd.concat([error,df1[scheme],df1["cpu_time "+scheme]],sort=False,axis=1)
        ax.loglog(error["cpu_time "+scheme],error[scheme],label=scheme, marker='o')
    print(error)
    ax.set_yscale('log') 
    ax.set_xscale('log') 
    plt.xlabel("cpu time [s] ",fontsize=15)
    plt.ylabel("$L_\infty$ norm",fontsize=15)
    ax.legend(loc= 'upper right')
    plt.grid(True,which="both",alpha=0.3, ls='-')
    plt.savefig(f'cpu-L2.png',dpi=300)
elif type_plot == "Hdim":
    error=pd.DataFrame()
    ig, ax = plt.subplots()
    
    error = pd.read_csv('Hdim_error_exp.csv')
    error = error.sort_values(by=['krylov_dim'])
    ax.plot(error["krylov_dim"],error['l2_norm_sum'],marker='o')
    print(error)
    ax.set_yscale('log') 
    #ax.set_xscale('log') 
    plt.xlabel("$H_m$ dimension ",fontsize=10)
    plt.ylabel("$ L_\infty$ error",fontsize=10)

    ax.legend(loc= 'upper right')
    plt.grid(True,which="both",alpha=0.3, ls='-')
    plt.savefig(f'H_dim.png',dpi=600)
elif type_plot == "Horden":
    dt=float(sys.argv[2])
    error=pd.DataFrame()
    df1 = pd.read_csv(f'results_dt_{dt}/error_scheme_exp.csv')
    error = pd.concat([error,df1["tiempo"],df1["exp"]],axis=1)
    types=[40,70,110,115,120,125,130]
    plt.figure(figsize=(12,6))
    for scheme in types:
        df1 = pd.read_csv(f'results_dt_{dt}/error_scheme_exp_H_{str(scheme)}.csv')
        error = pd.concat([error,df1['exp_'+str(scheme)]],axis=1)
        
    print(error)
    for scheme in types:
        plt.plot(error['tiempo'],error['exp_'+str(scheme)],label='exp_'+str(scheme))
    plt.xlabel('Time[s]')
    plt.ylabel('$L_2$ norm')
    plt.loglog(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.grid(True,which="both",alpha=0.3, ls='-')
    plt.savefig(f'L2norm_H_dim_dt_{dt}.png',dpi=300)
elif type_plot == "solution":
    dt=float(sys.argv[2])
    sol=pd.DataFrame()
    df1 = pd.read_csv(f'results_dt_{dt}/solution.csv')
    sol = pd.concat([sol,df1["x"]],axis=1)
    sol = pd.concat([sol,df1["Analytical"]],axis=1)
    for scheme in types:
        df1 = pd.read_csv(f'results_dt_{dt}/solution.csv')
        sol = pd.concat([sol,df1[scheme]],axis=1)

    plot = sol.plot(x='x',xlim=[0,20],title=f"Solution problem for dt = {dt}",grid=True,xlabel='$x$',ylabel='u')
    plt.grid(True,which="both",alpha=0.3, ls='-')
    plt.savefig(f'solution_dt_{dt}.png',dpi=300)

else:
    print('type plot error')