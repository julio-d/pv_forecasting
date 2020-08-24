################################
# PV Probabilistic Forecasting #
################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import scipy as sp
import statsmodels as sm
#from sklearn import 
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3


##### Importar dados ##########################################################

filepath = r'C:\Users\Júlio\Desktop\pv_forecasting\dados\Dados_sincronizados_Alentejo2.xlsx'

#extrair info de "Dados_sincronizados_Alentejo2" ______________________________
dados = {
        "dia0": pd.read_excel(filepath, sheet_name = "Dia_Actual", header = [1]),
        "dia-1": pd.read_excel(filepath, sheet_name = "Dia-1", header = [1]),
        "dia-2": pd.read_excel(filepath, sheet_name = "Dia-2", header = [1]),
        "dia-3": pd.read_excel(filepath, sheet_name = "Dia-3", header = [1]),
        "locais": pd.read_excel(filepath, sheet_name = "Posiciones", header = [0])
}

dados["dia0"]["Data"] = dados["dia0"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["dia0"].set_index("Data", inplace = True)

dados["dia-1"]["Data"] = dados["dia-1"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["dia-1"].set_index("Data", inplace = True)

dados["dia-2"]["Data"] = dados["dia-2"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["dia-2"].set_index("Data", inplace = True)

dados["dia-3"]["Data"] = dados["dia-3"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["dia-3"].set_index("Data", inplace = True)

##### Testar dados ############################################################

#testar datas dia 0 -> FALTA DIA 27/02/2019 E 05/04/2019 + TODAS AS 00H
teste=pd.date_range(start='2017/1/1', end='2020/5/31', freq='1H', closed='left')
n=teste.size
for k in range(n//24):
    teste = teste.drop(teste[k*23])
    n=n-1
teste=teste.drop(labels=pd.date_range(start='2019/2/27', end='2019/2/28', freq='1H', closed='left')[1:])
teste=teste.drop(labels=pd.date_range(start='2019/4/5', end='2019/4/6', freq='1H', closed='left')[1:])
if (dados["dia0"].index.equals(teste)==True):
    dados['dia0'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model

#testar datas dia -1 -> FALTA 28/02/2019 E 06/04/2019
teste=pd.date_range(start='2017/1/2', end='2020/6/1', freq='1H', closed='left').drop(labels=pd.date_range(start='2019/2/28', end='2019/3/1', freq='1H', closed='left'))
teste = teste.drop(labels=pd.date_range(start='2019/4/6', end='2019/4/7', freq='1H', closed='left'))
if (dados["dia-1"].index.equals(teste)==True):
    dados['dia-1'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model

#testar datas dia -2 -> FALTA 01/03/2019 E 06/04/2019
teste=pd.date_range(start='2017/1/3', end='2020/6/2', freq='1H', closed='left')
teste=teste.drop(labels=pd.date_range(start='2019/3/1', end='2019/3/2', freq='1H', closed='left'))
teste=teste.drop(labels=pd.date_range(start='2019/4/7', end='2019/4/8', freq='1H', closed='left'))
if (dados["dia-2"].index.equals(teste)==True):
    dados['dia-2'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model

#testar datas dia -3 -> FALTA 02/03/2019 E 07/04/2019
teste=pd.date_range(start='2017/1/4', end='2020/6/3', freq='1H', closed='left')
teste=teste.drop(labels=pd.date_range(start='2019/3/2', end='2019/3/3', freq='1H', closed='left'))
teste=teste.drop(labels=pd.date_range(start='2019/4/8', end='2019/4/9', freq='1H', closed='left'))
if (dados["dia-3"].index.equals(teste)==True):
    dados['dia-3'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model



#testar limites sup e inf das variaveis meteorologicas
teste_lims=pd.concat([dados['dia0'], dados['dia-1'], dados['dia-2'], dados['dia-3']], axis=1, sort=False)

teste_lims.filter(like='Radiación', axis=1).max().max(axis=0)
teste_lims.filter(like='Fraccion total', axis=1).max().max(axis=0)
teste_lims.filter(like='Frac. Baja', axis=1).max().max(axis=0)
teste_lims.filter(like='Frac. Media', axis=1).max().max(axis=0)
teste_lims.filter(like='Frac. Alta', axis=1).max().max(axis=0)
teste_lims.filter(like='Temperatura', axis=1).max().max(axis=0)-272.15
teste_lims.filter(like='Presión', axis=1).max().max(axis=0) #valores normais de mmHg diferentes destes
teste_lims.filter(like='Velocidad', axis=1).max().max(axis=0)
teste_lims.filter(like='Dirección', axis=1).max().max(axis=0)
teste_lims.filter(like='Humedad rel', axis=1).max().max(axis=0)
teste_lims.filter(like='Precipitación', axis=1).max().max(axis=0)
teste_lims.filter(like='Elev. Solar (º)', axis=1).max().max(axis=0)
teste_lims.filter(like='Produção (MW)', axis=1).max().max(axis=0)

teste_lims.filter(like='Radiación', axis=1).min().min(axis=0)
teste_lims.filter(like='Fraccion total', axis=1).min().min(axis=0)
teste_lims.filter(like='Frac. Baja', axis=1).min().min(axis=0)
teste_lims.filter(like='Frac. Media', axis=1).min().min(axis=0)
teste_lims.filter(like='Frac. Alta', axis=1).min().min(axis=0)
teste_lims.filter(like='Temperatura', axis=1).min().min(axis=0)-272.15
teste_lims.filter(like='Presión', axis=1).min().min(axis=0) #valores normais de mmHg diferentes destes
teste_lims.filter(like='Velocidad', axis=1).min().min(axis=0)
teste_lims.filter(like='Dirección', axis=1).min().min(axis=0)
teste_lims.filter(like='Humedad rel', axis=1).min().min(axis=0)
teste_lims.filter(like='Precipitación', axis=1).min().min(axis=0) #numero negativo - considerar 0?
teste_lims.filter(like='Elev. Solar (º)', axis=1).min().min(axis=0)
teste_lims.filter(like='Produção (MW)', axis=1).min().min(axis=0)


#testar NaN 
dados["dia0"].isnull().sum().sum(axis=0)
dados["dia-1"].isnull().sum().sum(axis=0)
dados["dia-2"].isnull().sum().sum(axis=0)
dados["dia-3"].isnull().sum().sum(axis=0)

##### Descritive statistics ###################################################

report = pp.ProfileReport(dados['dia0'])
report.to_file('profile_report.html') #erro ?


##### Organizacao dados #######################################################

#NWP grid _____________________________________________________________________

#dia0 
nwp_grid0 = {
        "central": dados["dia0"].iloc[:, 183:194],
        "elev_solar": dados["dia0"].iloc[:, 194]
}
i=3
cols = [0,1, i, i+1, i+2, i+3, i+4]
for a in range(1,7):
    for b in range(1,7):
        nwp_grid0[str(a)+'x'+str(b)] = dados["dia0"].iloc[:, cols]
        i=i+5
        

#dia -1
nwp_grid1 = {
        "central": dados["dia-1"].iloc[:, 183:194],
        "elev_solar": dados["dia-1"].iloc[:, 194]
}
i=3
cols = [0,1, i, i+1, i+2, i+3, i+4]
for a in range(1,7):
    for b in range(1,7):
        nwp_grid1[str(a)+'x'+str(b)] = dados["dia-1"].iloc[:, cols]
        i=i+5


#dia -2
nwp_grid2 = {
        "central": dados["dia-2"].iloc[:, 183:194],
        "elev_solar": dados["dia-2"].iloc[:, 194]
}
i=3
cols = [0,1, i, i+1, i+2, i+3, i+4]
for a in range(1,7):
    for b in range(1,7):
        nwp_grid2[str(a)+'x'+str(b)] = dados["dia-2"].iloc[:, cols]
        i=i+5


#dia -3
nwp_grid3 = {
        "central": dados["dia-3"].iloc[:, 183:194],
        "elev_solar": dados["dia-3"].iloc[:, 194]
}
i=3
cols = [0,1, i, i+1, i+2, i+3, i+4]
for a in range(1,7):
    for b in range(1,7):
        nwp_grid3[str(a)+'x'+str(b)] = dados["dia-3"].iloc[:, cols]
        i=i+5
        
nwpgrid = {
        "dia0": nwp_grid0,
        "dia-1": nwp_grid1,
        "dia-2": nwp_grid2,
        "dia-3": nwp_grid3
}

        
#potencia gerada ______________________________________________________________
producao = {
        "dia0": dados["dia0"].iloc[:, 195],
        "dia-1": dados["dia-1"].iloc[:, 195],
        "dia-2": dados["dia-2"].iloc[:, 195],
        "dia-3": dados["dia-3"].iloc[:, 195]
}

#localizacao dos pontos da grid
location = np.zeros((7,7,2)) #indices dos eixos 0 e 1 sao o ponto da nwp grid
i=0
for a in [1,2,3,4,5,6]:
    for b in [1,2,3,4,5,6]:
        location[a, b, 0]=dados['locais']['Latitud (º)'][i]
        location[a, b, 1]=dados['locais']['Longitud (º)'][i]
        i+=1
      
#indice 0,0,0 e 0,0,1 sao lat e long da central respetivamente
location[0,0,0]=dados['locais']['Latitud (º)'][36]
location[0,0,1]=dados['locais']['Longitud (º)'][36]

''' testar array location
for a in [0,1,2,3,4,5,6]:
    for b in [0,1,2,3,4,5,6]:
        print(location[a,b,0], location[a,b,1])'''


##### Feature engineering #####################################################

#Clear-sky model
        
for dia in [0,1,2,3]:
    for a in [1,2,3,4,5,6]:
        for b in [1,2,3,4,5,6]:
            local = Location(location[a,b,0], location[a,b,1])
            
            if dia==0:
                time = dados['dia0'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                nwpgrid['dia0'][str(a)+'x'+str(b)]=nwpgrid['dia0'][str(a)+'x'+str(b)].assign(csm_ghi=cs['ghi'].values)
            elif dia==1:
                time = dados['dia-1'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                nwpgrid['dia-1'][str(a)+'x'+str(b)]=nwpgrid['dia-1'][str(a)+'x'+str(b)].assign(csm_ghi=cs['ghi'].values)
            elif dia==2:
                time = dados['dia-2'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                nwpgrid['dia-2'][str(a)+'x'+str(b)]=nwpgrid['dia-2'][str(a)+'x'+str(b)].assign(csm_ghi=cs['ghi'].values)
            elif dia==3:
                time = dados['dia-3'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                nwpgrid['dia-3'][str(a)+'x'+str(b)]=nwpgrid['dia-3'][str(a)+'x'+str(b)].assign(csm_ghi=cs['ghi'].values)
            
            
#plot do ghi do csm
cs.plot();

plt.ylabel('Irradiance $W/m^2$');

plt.title('Ineichen, climatological turbidity');

#Weighted quantile regression


