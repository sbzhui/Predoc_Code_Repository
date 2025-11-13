#生成仿真数据
import pandas as pd
import numpy as np
from scipy.stats import invgamma
np.random.seed(42)
paras=pd.DataFrame(columns=['tao','rhog','rhoz','lny','lnpi','beta','kappa','lnr','rhor','phi1','phi2','random_std_g','random_std_z','random_std_r'])
for time in range(20000):
    #calculate x_t, pi_t, R_t
    tao=np.random.gamma((2/0.5)**2, (0.5**2)/2, size=1)[0]
    rhog=np.random.beta(((1-0.8)/(0.1**2)-1/0.8)*(0.8**2), ((1-0.8)/(0.1**2)-1/0.8)*(0.8**2)*(1/0.8-1), size=1)[0]

    rhoz=np.random.beta(((1-0.3)/(0.1**2)-1/0.3)*(0.3**2), ((1-0.3)/(0.1**2)-1/0.3)*(0.3**2)*(1/0.3-1), size=1)[0]

    lny=np.random.normal(0.005, 0.0025, size=1)[0]
    lnpi=np.random.normal(0.01, 0.005, size=1)[0]
    y=np.exp(lny)
    beta=np.random.uniform(0.985, 0.995)
    kappa=np.random.gamma((0.3/0.15)**2, (0.15**2)/0.3, size=1)[0]

    lnr=np.random.gamma((0.005/0.0025)**2, (0.0025**2)/0.005, size=1)[0]
    r=np.exp(lnr)

    rhor=np.random.beta(((1-0.5)/(0.2**2)-1/0.5)*(0.5**2), ((1-0.5)/(0.2**2)-1/0.5)*(0.5**2)*(1/0.5-1), size=1)[0]

    phi1=np.random.gamma((1.5/0.25)**2, (0.25**2)/1.5, size=1)[0]
    phi2=np.random.gamma((0.125/0.1)**2, (0.1**2)/0.125, size=1)[0]
    
    mean = 0.00630  # 均值
    std_dev = 0.00323 # 标准差
    # 计算逆伽玛分布的参数 shape 和 scale
    shape = mean**2 / std_dev**2
    scale = std_dev**2 / mean
    random_std_g = invgamma.rvs(shape, scale=scale, size=1)[0]
    
    mean = 0.00875  # 均值
    std_dev = 0.00430 # 标准差
    # 计算逆伽玛分布的参数 shape 和 scale
    shape = mean**2 / std_dev**2
    scale = std_dev**2 / mean
    random_std_z = invgamma.rvs(shape, scale=scale, size=1)[0]
    
    mean = 0.00251  # 均值
    std_dev = 0.00139 # 标准差
    # 计算逆伽玛分布的参数 shape 和 scale
    shape = mean**2 / std_dev**2
    scale = std_dev**2 / mean
    random_std_r = invgamma.rvs(shape, scale=scale, size=1)[0]
    
    para=[tao,rhog,rhoz,lny,lnpi,beta,kappa,lnr,rhor,phi1,phi2,random_std_g,random_std_z,random_std_r]
    paras=pd.concat([paras,pd.DataFrame(para,index=paras.columns).T])
    data_save=pd.DataFrame(columns=['t','x','pi','R'])
    for t in range(1,101):
        
        e_gt=np.random.normal(0, random_std_g, size=1)[0]
        e_zt=np.random.normal(0, random_std_z, size=1)[0]
        e_rt=np.random.normal(0, random_std_r, size=1)[0]
        
        if t==1:
            x_mean=0
            #pi_mean=-lnpi
            pi_mean=0
            #g_t=-np.log(0.03/4)
            g_t=e_gt
            #z_t=-lny
            z_t=e_zt
            r_t1=0
        else:
            x_mean=data_save['x'].mean()
            pi_mean=data_save['pi'].mean()
            g_t=rhog*g_t1+e_gt
            z_t=rhoz*z_t1+e_zt
            r_t1=data_save['R'].iloc[-1]
        # 定义系数矩阵 A 和右侧向量 b
        A = np.array([[1,0,1/tao], [-kappa,1, 0], [-(1-rhor)*phi2,-(1-rhor)*phi1,1]])
        b = np.array([x_mean+pi_mean/tao+(1-rhog)*g_t+rhoz*z_t/tao, y*pi_mean/r-kappa*g_t, rhor*r_t1+e_rt])

        # 使用 numpy.linalg.solve 解线性方程组
        sol = np.linalg.solve(A, b)

        g_t1=g_t
        z_t1=z_t
        data_save=pd.concat([data_save,pd.DataFrame({'t':[t], 'x':[sol[0]], 'pi':[sol[1]], 'R':[sol[2]],'z_t':z_t})])
        #print("解为 x =", sol)
    data_save['x']=data_save['x'].diff()+lny+data_save['z_t']
    data_save['pi']=data_save['pi']+lnpi
    data_save['R']=4*(data_save['R']+lnr+lnpi)
        
    data_save.loc[data_save['t']>10].to_csv('group'+str(time+1)+'.csv',index=False)