################################
# PV Probabilistic Forecasting #
################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling as pp
import statsmodels.api as sm
from pandas_profiling import ProfileReport
import scipy as sp
import statsmodels.formula.api as smf
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import properscoring as ps
from scipy.stats import norm
import lightgbm as lgb


##### Importar dados ##########################################################


filepath = r'C:\Users\Júlio\Desktop\pv_forecasting\dados\Dados_sincronizados_Alentejo_V3.xlsx'

dados = {
        "D": pd.read_excel(filepath, sheet_name = "D", header = [1]),
        "D+1": pd.read_excel(filepath, sheet_name = "D+1", header = [1]),
        "D+2": pd.read_excel(filepath, sheet_name = "D+2", header = [1]),
        "D+3": pd.read_excel(filepath, sheet_name = "D+3", header = [1]),
        "locais": pd.read_excel(filepath, sheet_name = "Posiciones", header = [0])
}

dados["D"]["Data"] = dados["D"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["D"].set_index("Data", inplace = True)

dados["D+1"]["Data"] = dados["D+1"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["D+1"].set_index("Data", inplace = True)

dados["D+2"]["Data"] = dados["D+2"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["D+2"].set_index("Data", inplace = True)

dados["D+3"]["Data"] = dados["D+3"][['Fecha', 'Hora']].agg(' '.join, axis=1)
dados["D+3"].set_index("Data", inplace = True)





##### Testar dados ############################################################


#testar datas + colocar datas como index_______________________________________


#testar datas dia 0 -> FALTA DIA 27/02/2019 E 05/04/2019 + TODAS AS 00H
teste=pd.date_range(start='2017/1/1', end='2020/5/31', freq='1H', closed='left')
n=teste.size
for k in range(n//24):
    teste = teste.drop(teste[k*23])
    n=n-1
teste=teste.drop(labels=pd.date_range(start='2019/2/27', end='2019/2/28', freq='1H', closed='left')[1:])
teste=teste.drop(labels=pd.date_range(start='2019/4/5', end='2019/4/6', freq='1H', closed='left')[1:])
if (dados["D"].index.equals(teste)==True):
    dados['D'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model

#testar datas dia -1 -> FALTA 28/02/2019 E 06/04/2019
teste=pd.date_range(start='2017/1/2', end='2020/6/1', freq='1H', closed='left').drop(labels=pd.date_range(start='2019/2/28', end='2019/3/1', freq='1H', closed='left'))
teste = teste.drop(labels=pd.date_range(start='2019/4/6', end='2019/4/7', freq='1H', closed='left'))
if (dados["D+1"].index.equals(teste)==True):
    dados['D+1'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model

#testar datas dia -2 -> FALTA 01/03/2019 E 06/04/2019
teste=pd.date_range(start='2017/1/3', end='2020/6/1', freq='1H', closed='left')
teste=teste.drop(labels=pd.date_range(start='2019/3/1', end='2019/3/2', freq='1H', closed='left'))
teste=teste.drop(labels=pd.date_range(start='2019/4/7', end='2019/4/8', freq='1H', closed='left'))
if (dados["D+2"].index.equals(teste)==True):
    dados['D+2'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model

#testar datas dia -3 -> FALTA 02/03/2019 E 07/04/2019
teste=pd.date_range(start='2017/1/4', end='2020/6/1', freq='1H', closed='left')
teste=teste.drop(labels=pd.date_range(start='2019/3/2', end='2019/3/3', freq='1H', closed='left'))
teste=teste.drop(labels=pd.date_range(start='2019/4/8', end='2019/4/9', freq='1H', closed='left'))
if (dados["D+3"].index.equals(teste)==True):
    dados['D+3'].set_index(teste, inplace=True) #preciso index em datatimeindex p/ clearsky model



#testar limites sup e inf das variaveis meteorologicas_________________________
    
#uniformizar nome coluna da precipitacao - passar valores negativos de precipitacao a 0
df=dados['D+1']
df=df.rename(columns={"Precipitación (mm)": "Precipitación"})
dados['D+1']=df

#passar valores negativos de precipitacao a 0

df=dados['D']['Precipitación']
df=np.where(df<0, 0, df)  
dados['D']['Precipitación']=df

df=dados['D+1']['Precipitación']
df=np.where(df<0, 0, df)  
dados['D+1']['Precipitación']=df

df=dados['D+2']['Precipitación']
df=np.where(df<0, 0, df)  
dados['D+2']['Precipitación']=df

df=dados['D+3']['Precipitación']
df=np.where(df<0, 0, df)  
dados['D+3']['Precipitación']=df    
    
teste_lims=pd.concat([dados['D'], dados['D+1'], dados['D+2'], dados['D+3']], axis=1, sort=False)

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
teste_lims.filter(like='Precipitación', axis=1).min().min(axis=0)
teste_lims.filter(like='Elev. Solar (º)', axis=1).min().min(axis=0)
teste_lims.filter(like='Produção (MW)', axis=1).min().min(axis=0)


#testar NaN ___________________________________________________________________

dados["D"].isnull().sum().sum(axis=0)
dados["D+1"].isnull().sum().sum(axis=0)
dados["D+2"].isnull().sum().sum(axis=0)
dados["D+3"].isnull().sum().sum(axis=0)




##### Descritive statistics ###################################################


#report=ProfileReport(dados['D'])  
#report.to_file(output_file='report.html')

'''DUVIDA da erro o primeiro destes 2 comandos - concat() got an unexpected keyword 
argument 'join_axes' '''

##### Organizacao dados #######################################################

#NWP grid _____________________________________________________________________

#D 
nwp_grid0 = {
        "central": dados["D"].iloc[:, 183:194],
        "elev_solar": dados["D"].iloc[:, 194]
}
nwp_grid0['central']=nwp_grid0['central'].assign(producao=dados["D"].iloc[:, 195].values)

#retirar dias sem producao
from_ts = '2017-01-20 00:00:00'
to_ts = '2017-02-02 23:00:00'
nwp_grid0['central'] = nwp_grid0['central'][(nwp_grid0['central'].index < from_ts) | (nwp_grid0['central'].index > to_ts)]
nwp_grid0['elev_solar'] = nwp_grid0['elev_solar'][(nwp_grid0['elev_solar'].index < from_ts) | (nwp_grid0['elev_solar'].index > to_ts)]


i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid0[str(a)+'x'+str(b)] = dados["D"].iloc[:, cols]
        i=i+5
        nwp_grid0[str(a)+'x'+str(b)] = nwp_grid0[str(a)+'x'+str(b)][(nwp_grid0[str(a)+'x'+str(b)].index < from_ts) | (nwp_grid0[str(a)+'x'+str(b)].index > to_ts)]



#D+1
nwp_grid1 = {
        "central": dados["D+1"].iloc[:, 183:194],
        "elev_solar": dados["D+1"].iloc[:, 194]
}
nwp_grid1['central']=nwp_grid1['central'].assign(producao=dados["D+1"].iloc[:, 195].values)

#retirar dias sem producao
nwp_grid1['central'] = nwp_grid1['central'][(nwp_grid1['central'].index < from_ts) | (nwp_grid1['central'].index > to_ts)]
nwp_grid1['elev_solar'] = nwp_grid1['elev_solar'][(nwp_grid1['elev_solar'].index < from_ts) | (nwp_grid1['elev_solar'].index > to_ts)]



i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid1[str(a)+'x'+str(b)] = dados["D+1"].iloc[:, cols]
        i=i+5
        nwp_grid1[str(a)+'x'+str(b)] = nwp_grid1[str(a)+'x'+str(b)][(nwp_grid1[str(a)+'x'+str(b)].index < from_ts) | (nwp_grid1[str(a)+'x'+str(b)].index > to_ts)]



#D+2
nwp_grid2 = {
        "central": dados["D+2"].iloc[:, 183:194],
        "elev_solar": dados["D+2"].iloc[:, 194]
}
nwp_grid2['central']=nwp_grid2['central'].assign(producao=dados["D+2"].iloc[:, 195].values)

#retirar dias sem producao
nwp_grid2['central'] = nwp_grid2['central'][(nwp_grid2['central'].index < from_ts) | (nwp_grid2['central'].index > to_ts)]
nwp_grid2['elev_solar'] = nwp_grid2['elev_solar'][(nwp_grid2['elev_solar'].index < from_ts) | (nwp_grid2['elev_solar'].index > to_ts)]



i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid2[str(a)+'x'+str(b)] = dados["D+2"].iloc[:, cols]
        i=i+5
        nwp_grid2[str(a)+'x'+str(b)] = nwp_grid2[str(a)+'x'+str(b)][(nwp_grid2[str(a)+'x'+str(b)].index < from_ts) | (nwp_grid2[str(a)+'x'+str(b)].index > to_ts)]



#D+3
nwp_grid3 = {
        "central": dados["D+3"].iloc[:, 183:194],
        "elev_solar": dados["D+3"].iloc[:, 194]
}
nwp_grid3['central']=nwp_grid3['central'].assign(producao=dados["D+3"].iloc[:, 195].values)

#retirar dias sem producao
nwp_grid3['central'] = nwp_grid3['central'][(nwp_grid3['central'].index < from_ts) | (nwp_grid3['central'].index > to_ts)]
nwp_grid3['elev_solar'] = nwp_grid3['elev_solar'][(nwp_grid3['elev_solar'].index < from_ts) | (nwp_grid3['elev_solar'].index > to_ts)]



i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid3[str(a)+'x'+str(b)] = dados["D+3"].iloc[:, cols]
        i=i+5
        nwp_grid3[str(a)+'x'+str(b)] = nwp_grid3[str(a)+'x'+str(b)][(nwp_grid3[str(a)+'x'+str(b)].index < from_ts) | (nwp_grid3[str(a)+'x'+str(b)].index > to_ts)]

        
        
        
nwpgrid = {
        "D": nwp_grid0,
        "D+1": nwp_grid1,
        "D+2": nwp_grid2,
        "D+3": nwp_grid3
}


#normalizacao dos dados da central
scaler = MinMaxScaler()

df=nwpgrid['D']['central']
df.iloc[:,0]=scaler.fit_transform(df.iloc[:,0].values.reshape(-1,1))
df.iloc[:,2]=scaler.fit_transform(df.iloc[:,2].values.reshape(-1,1))
df.iloc[:,3]=scaler.fit_transform(df.iloc[:,3].values.reshape(-1,1))
df.iloc[:,4]=scaler.fit_transform(df.iloc[:,4].values.reshape(-1,1))
df.iloc[:,10]=scaler.fit_transform(df.iloc[:,10].values.reshape(-1,1))
nwpgrid['D']['central']=df

df=nwpgrid['D']['elev_solar']
df=scaler.fit_transform(df.values.reshape(-1,1))
nwpgrid['D']['elev_solar']=pd.Series(df[:,0], index=nwpgrid['D']['elev_solar'].index)




df=nwpgrid['D+1']['central']
df.iloc[:,0]=scaler.fit_transform(df.iloc[:,0].values.reshape(-1,1))
df.iloc[:,2]=scaler.fit_transform(df.iloc[:,2].values.reshape(-1,1))
df.iloc[:,3]=scaler.fit_transform(df.iloc[:,3].values.reshape(-1,1))
df.iloc[:,4]=scaler.fit_transform(df.iloc[:,4].values.reshape(-1,1))
df.iloc[:,10]=scaler.fit_transform(df.iloc[:,10].values.reshape(-1,1))
nwpgrid['D+1']['central']=df

df=nwpgrid['D+1']['elev_solar']
df=scaler.fit_transform(df.values.reshape(-1,1))
nwpgrid['D+1']['elev_solar']=pd.Series(df[:,0], index=nwpgrid['D+1']['elev_solar'].index)



df=nwpgrid['D+2']['central']
df.iloc[:,0]=scaler.fit_transform(df.iloc[:,0].values.reshape(-1,1))
df.iloc[:,2]=scaler.fit_transform(df.iloc[:,2].values.reshape(-1,1))
df.iloc[:,3]=scaler.fit_transform(df.iloc[:,3].values.reshape(-1,1))
df.iloc[:,4]=scaler.fit_transform(df.iloc[:,4].values.reshape(-1,1))
df.iloc[:,10]=scaler.fit_transform(df.iloc[:,10].values.reshape(-1,1))
nwpgrid['D+2']['central']=df

df=nwpgrid['D+2']['elev_solar']
df=scaler.fit_transform(df.values.reshape(-1,1))
nwpgrid['D+2']['elev_solar']=pd.Series(df[:,0], index=nwpgrid['D+2']['elev_solar'].index)



df=nwpgrid['D+3']['central']
df.iloc[:,0]=scaler.fit_transform(df.iloc[:,0].values.reshape(-1,1))
df.iloc[:,2]=scaler.fit_transform(df.iloc[:,2].values.reshape(-1,1))
df.iloc[:,3]=scaler.fit_transform(df.iloc[:,3].values.reshape(-1,1))
df.iloc[:,4]=scaler.fit_transform(df.iloc[:,4].values.reshape(-1,1))
df.iloc[:,10]=scaler.fit_transform(df.iloc[:,10].values.reshape(-1,1))
nwpgrid['D+3']['central']=df

df=nwpgrid['D+3']['elev_solar']
df=scaler.fit_transform(df.values.reshape(-1,1))
nwpgrid['D+3']['elev_solar']=pd.Series(df[:,0], index=nwpgrid['D+3']['elev_solar'].index)



#localizacao dos pontos da grid________________________________________________

location = np.zeros((7,7,2)) #indices dos eixos 0 e 1 sao o ponto da nwp grid
i=0
for a in range(1,7):
    for b in range(1,7):
        location[a, b, 0]=dados['locais']['Latitud (º)'][i]
        location[a, b, 1]=dados['locais']['Longitud (º)'][i]
        i+=1
      
#indice 0,0,0 e 0,0,1 sao lat e long da central respetivamente
location[0,0,0]=dados['locais']['Latitud (º)'][36]
location[0,0,1]=dados['locais']['Longitud (º)'][36]

''' testar array location
for a in range(0,7):
    for b in range(0,7):
        print(location[a,b,0], location[a,b,1])'''




##### Feature engineering #####################################################


#Clear-sky model - adicionar colunas Ics e Inorm ______________________________

'''utilizo o indice das colunas em vez do nome porque ao importar do excel, as colunas com
nome igual sao lhe colocadas indicacoes '.X' sendo X um numero crescente. tentei alterar 
pd.read_excel para alterar isso mas nao consegui pelo que optei por chamar as colunas pelo
indice porqe esse mantem-se para qualquer ponto da grid'''

#central
local = Location(location[0,0,0], location[0,0,1]) #localizacao da central

df=nwpgrid['D']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[1]]+10)/(df[df.columns[12]]+10), df['irr_norm'])  
nwpgrid['D']['central']=df 
  

df=nwpgrid['D+1']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[1]]+10)/(df[df.columns[12]]+10), df['irr_norm'])  
nwpgrid['D+1']['central']=df 
  

df=nwpgrid['D+2']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[1]]+10)/(df[df.columns[12]]+10), df['irr_norm'])  
nwpgrid['D+2']['central']=df         
  
  
df=nwpgrid['D+3']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[1]]+50)/(df[df.columns[12]]+50), df['irr_norm'])  
nwpgrid['D+3']['central']=df 
              
        
#pontos nwp grid        
for dia in range(0,4):
    for a in range(1,7):
        for b in range(1,7):
            local = Location(location[a,b,0], location[a,b,1])
            
            if dia==0:
                df=nwpgrid['D'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[2]]+10)/(df[df.columns[7]]+10), df['irr_norm'])  
                nwpgrid['D'][str(a)+'x'+str(b)]=df 
                
            elif dia==1:
                df=nwpgrid['D+1'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[2]]+10)/(df[df.columns[7]]+10), df['irr_norm'])  
                nwpgrid['D+1'][str(a)+'x'+str(b)]=df 
                
            elif dia==2:
                df=nwpgrid['D+2'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[2]]+10)/(df[df.columns[7]]+10), df['irr_norm'])  
                nwpgrid['D+2'][str(a)+'x'+str(b)]=df 
            
            elif dia==3:
                df=nwpgrid['D+3'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, (df[df.columns[2]]+10)/(df[df.columns[7]]+10), df['irr_norm'])  
                nwpgrid['D+3'][str(a)+'x'+str(b)]=df 
        

        
        
#Weighted quantile regression _________________________________________________

aux = {'irradiancia': nwpgrid['D']['central']['Radiación.36'],
       'potencia': nwpgrid['D']['central']['producao']
}

#scatter plot dos pts  I e P na central
plt.scatter(aux['irradiancia'], aux['potencia'])
plt.xlabel('Irradiancia')
plt.ylabel('Potencia')


data = pd.DataFrame (aux, columns = ['irradiancia','potencia'])
mod = smf.quantreg('potencia ~ irradiancia', data)
res = mod.fit(q=.5)
#print(res.summary())

intercept = res.params['Intercept']
beta = res.params['irradiancia']






#Obter Pcs  ___________________________________________________________________

#P = intercept + beta . I 
if intercept<10e-05:
  intercept=0

#D
df=nwpgrid['D']['central']
df=df.assign(Pcs = intercept + beta*df[df.columns[12]].values) #coluna 12 = Ics  
nwpgrid['D']['central']=df 
    
#D+1
df=nwpgrid['D+1']['central']
df=df.assign(Pcs = intercept + beta*df[df.columns[12]].values) #coluna 12 = Ics  
nwpgrid['D+1']['central']=df 

#D+2
df=nwpgrid['D+2']['central']
df=df.assign(Pcs = intercept + beta*df[df.columns[12]].values) #coluna 12 = Ics  
nwpgrid['D+2']['central']=df 

#D+3
df=nwpgrid['D+3']['central']
df=df.assign(Pcs = intercept + beta*df[df.columns[12]].values) #coluna 12 = Ics  
nwpgrid['D+3']['central']=df 




#Normalizar P _________________________________________________________________

#D
df=nwpgrid['D']['central']
df=df.assign(pot_norm=df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, (df[df.columns[11]]+1)/(df[df.columns[14]]+1), df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D']['central']=df 
    
#D+1
df=nwpgrid['D+1']['central']
df=df.assign(pot_norm=df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, (df[df.columns[11]]+1)/(df[df.columns[14]]+1), df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D+1']['central']=df 

#D+2
df=nwpgrid['D+2']['central']
df=df.assign(pot_norm =df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, (df[df.columns[11]]+1)/(df[df.columns[14]]+1), df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D+2']['central']=df 

#D+3
df=nwpgrid['D+3']['central']
df=df.assign(pot_norm =df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, (df[df.columns[11]]+1)/(df[df.columns[14]]+1), df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D+3']['central']=df 







#Apenas dia D _________________________________________________________________



#variancia temporal -> apliquei a todas as variaveis

#central
df=pd.DataFrame()
aux=nwpgrid['D']['central']
    
#temperatura
df=df.assign(temp_var3=aux[aux.columns[0]].rolling(window=3, center=True).var())
df=df.assign(temp_var7=aux[aux.columns[0]].rolling(window=7, center=True).var())
df=df.assign(temp_var11=aux[aux.columns[0]].rolling(window=11, center=True).var())

#pressao
df=df.assign(pres_var3=aux[aux.columns[2]].rolling(window=3, center=True).var())
df=df.assign(pres_var7=aux[aux.columns[2]].rolling(window=7, center=True).var())
df=df.assign(pres_var11=aux[aux.columns[2]].rolling(window=11, center=True).var())

#velociade vento
df=df.assign(vel_var3=aux[aux.columns[3]].rolling(window=3, center=True).var())
df=df.assign(vel_var7=aux[aux.columns[3]].rolling(window=7, center=True).var())
df=df.assign(vel_var11=aux[aux.columns[3]].rolling(window=11, center=True).var())

#direcao vento
df=df.assign(dir_var3=aux[aux.columns[4]].rolling(window=3, center=True).var())
df=df.assign(dir_var7=aux[aux.columns[4]].rolling(window=7, center=True).var())
df=df.assign(dir_var11=aux[aux.columns[4]].rolling(window=11, center=True).var())

#humidade relativa
df=df.assign(humr_var3=aux[aux.columns[5]].rolling(window=3, center=True).var())
df=df.assign(humr_var7=aux[aux.columns[5]].rolling(window=7, center=True).var())
df=df.assign(humr_var11=aux[aux.columns[5]].rolling(window=11, center=True).var())

#frac total
df=df.assign(ftotal_var3=aux[aux.columns[6]].rolling(window=3, center=True).var())
df=df.assign(ftotal_var7=aux[aux.columns[6]].rolling(window=7, center=True).var())
df=df.assign(ftotal_var11=aux[aux.columns[6]].rolling(window=11, center=True).var())

#frac baixa
df=df.assign(fbaixa_var3=aux[aux.columns[7]].rolling(window=3, center=True).var())
df=df.assign(fbaixa_var7=aux[aux.columns[7]].rolling(window=7, center=True).var())
df=df.assign(fbaixa_var11=aux[aux.columns[7]].rolling(window=11, center=True).var())

#frac media
df=df.assign(fmedia_var3=aux[aux.columns[8]].rolling(window=3, center=True).var())
df=df.assign(fmedia_var7=aux[aux.columns[8]].rolling(window=7, center=True).var())
df=df.assign(fmedia_var11=aux[aux.columns[8]].rolling(window=11, center=True).var())

#frac alta
df=df.assign(falta_var3=aux[aux.columns[9]].rolling(window=3, center=True).var())
df=df.assign(falta_var7=aux[aux.columns[9]].rolling(window=7, center=True).var())
df=df.assign(falta_var11=aux[aux.columns[9]].rolling(window=11, center=True).var())

#precipitacao
df=df.assign(prec_var3=aux[aux.columns[10]].rolling(window=3, center=True).var())
df=df.assign(prec_var7=aux[aux.columns[10]].rolling(window=7, center=True).var())
df=df.assign(prec_var11=aux[aux.columns[10]].rolling(window=11, center=True).var())

#irradiancia normalizada
df=df.assign(irrn_var3=aux[aux.columns[13]].rolling(window=3, center=True).var())
df=df.assign(irrn_var7=aux[aux.columns[13]].rolling(window=7, center=True).var())
df=df.assign(irrn_var11=aux[aux.columns[13]].rolling(window=11, center=True).var())


#Retirar nan iniciais e finais - colocar igual a previsao imediatamente seguinte/anterior 
df=df.ffill()
df=df.bfill()

dados_calc={'var_temp': {'D': {'central': df}}}




#nwp grid
for a in range(1,7):
  for b in range(1,7):
    
    aux=nwpgrid['D'][str(a)+'x'+str(b)]
    df=pd.DataFrame()
    
    #frac. total
    df=df.assign(ftotal_var3=aux[aux.columns[3]].rolling(window=3, center=True).var())
    df=df.assign(ftotal_var7=aux[aux.columns[3]].rolling(window=7, center=True).var())
    df=df.assign(ftotal_var11=aux[aux.columns[3]].rolling(window=11, center=True).var())
    
    #frac. baixa
    df=df.assign(fbaixa_var3=aux[aux.columns[4]].rolling(window=3, center=True).var())
    df=df.assign(fbaixa_var7=aux[aux.columns[4]].rolling(window=7, center=True).var())
    df=df.assign(fbaixa_var11=aux[aux.columns[4]].rolling(window=11, center=True).var())
    
    #frac. media
    df=df.assign(fmedia_var3=aux[aux.columns[5]].rolling(window=3, center=True).var())
    df=df.assign(fmedia_var7=aux[aux.columns[5]].rolling(window=7, center=True).var())
    df=df.assign(fmedia_var11=aux[aux.columns[5]].rolling(window=11, center=True).var())
    
    #frac. alta
    df=df.assign(falta_var3=aux[aux.columns[6]].rolling(window=3, center=True).var())
    df=df.assign(falta_var7=aux[aux.columns[6]].rolling(window=7, center=True).var())
    df=df.assign(falta_var11=aux[aux.columns[6]].rolling(window=11, center=True).var())
    
    #irradiancia normalizada
    df=df.assign(irrn_var3=aux[aux.columns[8]].rolling(window=3, center=True).var())
    df=df.assign(irrn_var7=aux[aux.columns[8]].rolling(window=7, center=True).var())
    df=df.assign(irrn_var11=aux[aux.columns[8]].rolling(window=11, center=True).var())
    
    #Retirar nan iniciais e finais - colocar igual a previsao imediatamente seguinte/anterior 
    df=df.ffill()
    df=df.bfill()
  

    dados_calc['var_temp']['D'][str(a)+'x'+str(b)]=df



#lags 
    
#central
aux=nwpgrid['D']['central']
df=pd.DataFrame()
 
#temperatura
df=df.assign(temp_lag1=aux[aux.columns[0]].shift(periods=1))
df=df.assign(temp_lag2=aux[aux.columns[0]].shift(periods=2))
df=df.assign(temp_lag3=aux[aux.columns[0]].shift(periods=3))

#pressao
df=df.assign(pres_lag1=aux[aux.columns[2]].shift(periods=1))
df=df.assign(pres_lag2=aux[aux.columns[2]].shift(periods=2))
df=df.assign(pres_lag3=aux[aux.columns[2]].shift(periods=3))

#velociade vento
df=df.assign(vel_lag1=aux[aux.columns[3]].shift(periods=1))
df=df.assign(vel_lag2=aux[aux.columns[3]].shift(periods=2))
df=df.assign(vel_lag3=aux[aux.columns[3]].shift(periods=3))

#direcao vento
df=df.assign(dir_lag1=aux[aux.columns[4]].shift(periods=1))
df=df.assign(dir_lag2=aux[aux.columns[4]].shift(periods=2))
df=df.assign(dir_lag3=aux[aux.columns[4]].shift(periods=3))

#humidade relativa
df=df.assign(humr_lag1=aux[aux.columns[5]].shift(periods=1))
df=df.assign(humr_lag2=aux[aux.columns[5]].shift(periods=2))
df=df.assign(humr_lag3=aux[aux.columns[5]].shift(periods=3))

#frac total
df=df.assign(ftotal_lag1=aux[aux.columns[6]].shift(periods=1))
df=df.assign(ftotal_lag2=aux[aux.columns[6]].shift(periods=2))
df=df.assign(ftotal_lag3=aux[aux.columns[6]].shift(periods=3))

#frac baixa
df=df.assign(fbaixa_lag1=aux[aux.columns[7]].shift(periods=1))
df=df.assign(fbaixa_lag2=aux[aux.columns[7]].shift(periods=2))
df=df.assign(fbaixa_lag3=aux[aux.columns[7]].shift(periods=3))

#frac media
df=df.assign(fmedia_lag1=aux[aux.columns[8]].shift(periods=1))
df=df.assign(fmedia_lag2=aux[aux.columns[8]].shift(periods=2))
df=df.assign(fmedia_lag3=aux[aux.columns[8]].shift(periods=3))

#frac alta
df=df.assign(falta_lag1=aux[aux.columns[9]].shift(periods=1))
df=df.assign(falta_lag2=aux[aux.columns[9]].shift(periods=2))
df=df.assign(falta_lag3=aux[aux.columns[9]].shift(periods=3))

#precipitacao
df=df.assign(prec_lag1=aux[aux.columns[10]].shift(periods=1))
df=df.assign(prec_lag2=aux[aux.columns[10]].shift(periods=2))
df=df.assign(prec_lag3=aux[aux.columns[10]].shift(periods=3))

#irradiancia normalizada
df=df.assign(irrn_lag1=aux[aux.columns[13]].shift(periods=1))
df=df.assign(irrn_lag2=aux[aux.columns[13]].shift(periods=2))
df=df.assign(irrn_lag3=aux[aux.columns[13]].shift(periods=3))



#Retirar nan iniciais - colocar igual a previsao imediatamente seguinte
df = df.bfill()

dados_calc['lags']={'D': {'central': df}}




#nwp grid
for a in range(1,7):
  for b in range(1,7):
    
    aux=nwpgrid['D'][str(a)+'x'+str(b)]
    df=pd.DataFrame()
    
    ##frac. total
    df=df.assign(ftotal_lag1=aux[aux.columns[3]].shift(periods=1))
    df=df.assign(ftotal_lag2=aux[aux.columns[3]].shift(periods=2))
    df=df.assign(ftotal_lag3=aux[aux.columns[3]].shift(periods=3))
    
    #frac. baixa
    df=df.assign(fbaixa_lag1=aux[aux.columns[4]].shift(periods=1))
    df=df.assign(fbaixa_lag2=aux[aux.columns[4]].shift(periods=2))
    df=df.assign(fbaixa_lag3=aux[aux.columns[4]].shift(periods=3))
    
    #frac. media
    df=df.assign(fmedia_lag1=aux[aux.columns[5]].shift(periods=1))
    df=df.assign(fmedia_lag2=aux[aux.columns[5]].shift(periods=2))
    df=df.assign(fmedia_lag3=aux[aux.columns[5]].shift(periods=3))
    
    #frac. alta
    df=df.assign(falta_lag1=aux[aux.columns[6]].shift(periods=1))
    df=df.assign(falta_lag2=aux[aux.columns[6]].shift(periods=2))
    df=df.assign(falta_lag3=aux[aux.columns[6]].shift(periods=3))
    
    #irradiancia normalizada
    df=df.assign(irrn_lag1=aux[aux.columns[8]].shift(periods=1))
    df=df.assign(irrn_lag2=aux[aux.columns[8]].shift(periods=2))
    df=df.assign(irrn_lag3=aux[aux.columns[8]].shift(periods=3))
    
    
    #Retirar nan iniciais - colocar igual a ao valor imediatamente seguinte
    df=df.bfill()
    
    dados_calc['lags']['D'][str(a)+'x'+str(b)]=df






#leads 
    
#central
df=pd.DataFrame()
aux=nwpgrid['D']['central']
    
#temperatura
df=df.assign(temp_lead1=aux[aux.columns[0]].shift(periods=-1))
df=df.assign(temp_lead2=aux[aux.columns[0]].shift(periods=-2))
df=df.assign(temp_lead3=aux[aux.columns[0]].shift(periods=-3))

#pressao
df=df.assign(pres_lead1=aux[aux.columns[2]].shift(periods=-1))
df=df.assign(pres_lead2=aux[aux.columns[2]].shift(periods=-2))
df=df.assign(pres_lead3=aux[aux.columns[2]].shift(periods=-3))

#velociade vento
df=df.assign(vel_lead1=aux[aux.columns[3]].shift(periods=-1))
df=df.assign(vel_lead2=aux[aux.columns[3]].shift(periods=-2))
df=df.assign(vel_lead3=aux[aux.columns[3]].shift(periods=-3))

#direcao vento
df=df.assign(dir_lead1=aux[aux.columns[4]].shift(periods=-1))
df=df.assign(dir_lead2=aux[aux.columns[4]].shift(periods=-2))
df=df.assign(dir_lead3=aux[aux.columns[4]].shift(periods=-3))

#humidade relativa
df=df.assign(humr_lead1=aux[aux.columns[5]].shift(periods=-1))
df=df.assign(humr_lead2=aux[aux.columns[5]].shift(periods=-2))
df=df.assign(humr_lead3=aux[aux.columns[5]].shift(periods=-3))

#frac total
df=df.assign(ftotal_lead1=aux[aux.columns[6]].shift(periods=-1))
df=df.assign(ftotal_lead2=aux[aux.columns[6]].shift(periods=-2))
df=df.assign(ftotal_lead3=aux[aux.columns[6]].shift(periods=-3))

#frac baixa
df=df.assign(fbaixa_lead1=aux[aux.columns[7]].shift(periods=-1))
df=df.assign(fbaixa_lead2=aux[aux.columns[7]].shift(periods=-2))
df=df.assign(fbaixa_lead3=aux[aux.columns[7]].shift(periods=-3))

#frac media
df=df.assign(fmedia_lead1=aux[aux.columns[8]].shift(periods=-1))
df=df.assign(fmedia_lead2=aux[aux.columns[8]].shift(periods=-2))
df=df.assign(fmedia_lead3=aux[aux.columns[8]].shift(periods=-3))

#frac alta
df=df.assign(falta_lead1=aux[aux.columns[9]].shift(periods=-1))
df=df.assign(falta_lead2=aux[aux.columns[9]].shift(periods=-2))
df=df.assign(falta_lead3=aux[aux.columns[9]].shift(periods=-3))

#precipitacao
df=df.assign(prec_lead1=aux[aux.columns[10]].shift(periods=-1))
df=df.assign(prec_lead2=aux[aux.columns[10]].shift(periods=-2))
df=df.assign(prec_lead3=aux[aux.columns[10]].shift(periods=-3))

#irradiancia normalizada
df=df.assign(irrn_lead1=aux[aux.columns[13]].shift(periods=-1))
df=df.assign(irrn_lead2=aux[aux.columns[13]].shift(periods=-2))
df=df.assign(irrn_lead3=aux[aux.columns[13]].shift(periods=-3))



#Retirar nan finais - colocar igual a previsao imediatamente anterior
df=df.ffill()


dados_calc['leads']={'D': {'central': df}}

#nwp grid
for a in range(1,7):
  for b in range(1,7):
    
    df=pd.DataFrame()
    aux=nwpgrid['D'][str(a)+'x'+str(b)]
    
    ##frac. total
    df=df.assign(ftotal_lead1=aux[aux.columns[3]].shift(periods=-1))
    df=df.assign(ftotal_lead2=aux[aux.columns[3]].shift(periods=-2))
    df=df.assign(ftotal_lead3=aux[aux.columns[3]].shift(periods=-3))
    
    #frac. baixa
    df=df.assign(fbaixa_lead1=aux[aux.columns[4]].shift(periods=-1))
    df=df.assign(fbaixa_lead2=aux[aux.columns[4]].shift(periods=-2))
    df=df.assign(fbaixa_lead3=aux[aux.columns[4]].shift(periods=-3))
    
    #frac. media
    df=df.assign(fmedia_lead1=aux[aux.columns[5]].shift(periods=-1))
    df=df.assign(fmedia_lead2=aux[aux.columns[5]].shift(periods=-2))
    df=df.assign(fmedia_lead3=aux[aux.columns[5]].shift(periods=-3))
    
    #frac. alta
    df=df.assign(falta_lead1=aux[aux.columns[6]].shift(periods=-1))
    df=df.assign(falta_lead2=aux[aux.columns[6]].shift(periods=-2))
    df=df.assign(falta_lead3=aux[aux.columns[6]].shift(periods=-3))
    
    #irradiancia normalizada
    df=df.assign(irrn_lead1=aux[aux.columns[8]].shift(periods=-1))
    df=df.assign(irrn_lead2=aux[aux.columns[8]].shift(periods=-2))
    df=df.assign(irrn_lead3=aux[aux.columns[8]].shift(periods=-3))
    
    
    #Retirar nan finais - colocar igual a previsao imediatamente anterior
    df=df.ffill()
    
    dados_calc['leads']['D'][str(a)+'x'+str(b)]=df















#EXTRA: MEDIAS TEMPORAIS (janelas de 3, 7 e 11 horas)


#central
df=pd.DataFrame()
aux=nwpgrid['D']['central']
    

'''df=df.assign(temp_med3=aux[aux.columns[0]].rolling(window=3, center=True).mean())
df=df.assign(pres_med3=aux[aux.columns[2]].rolling(window=3, center=True).mean())
df=df.assign(vel_med3=aux[aux.columns[3]].rolling(window=3, center=True).mean())
df=df.assign(hum_med3=aux[aux.columns[5]].rolling(window=3, center=True).mean())'''


#direcao vento
df=df.assign(dir_med3=aux[aux.columns[4]].rolling(window=3, center=True).mean())
df=df.assign(dir_med7=aux[aux.columns[4]].rolling(window=7, center=True).mean())
df=df.assign(dir_med11=aux[aux.columns[4]].rolling(window=11, center=True).mean())


#frac total
df=df.assign(ftotal_med3=aux[aux.columns[6]].rolling(window=3, center=True).mean())
df=df.assign(ftotal_med7=aux[aux.columns[6]].rolling(window=7, center=True).mean())
df=df.assign(ftotal_med11=aux[aux.columns[6]].rolling(window=11, center=True).mean())


#frac baixa
df=df.assign(fbaixa_med3=aux[aux.columns[7]].rolling(window=3, center=True).mean())
df=df.assign(fbaixa_med7=aux[aux.columns[7]].rolling(window=7, center=True).mean())
df=df.assign(fbaixa_med11=aux[aux.columns[7]].rolling(window=11, center=True).mean())

#frac media
df=df.assign(fmedia_med3=aux[aux.columns[8]].rolling(window=3, center=True).mean())
df=df.assign(fmedia_med7=aux[aux.columns[8]].rolling(window=7, center=True).mean())
df=df.assign(fmedia_med11=aux[aux.columns[8]].rolling(window=11, center=True).mean())

#frac alta
df=df.assign(falta_med3=aux[aux.columns[9]].rolling(window=3, center=True).mean())
df=df.assign(falta_med7=aux[aux.columns[9]].rolling(window=7, center=True).mean())
df=df.assign(falta_med11=aux[aux.columns[9]].rolling(window=11, center=True).mean())


#precipitacao
df=df.assign(prec_med3=aux[aux.columns[10]].rolling(window=3, center=True).mean())
df=df.assign(prec_med7=aux[aux.columns[10]].rolling(window=7, center=True).mean())
df=df.assign(prec_med11=aux[aux.columns[10]].rolling(window=11, center=True).mean())

#irradiancia normalizada
df=df.assign(irrn_med3=aux[aux.columns[13]].rolling(window=3, center=True).mean())
df=df.assign(irrn_med7=aux[aux.columns[13]].rolling(window=7, center=True).mean())
df=df.assign(irrn_med11=aux[aux.columns[13]].rolling(window=11, center=True).mean())
df=df.assign(irrn_med25=aux[aux.columns[13]].rolling(window=25, center=True).mean())

#Retirar nan iniciais e finais - colocar igual a previsao imediatamente anterior/seguinte
df=df.ffill()
df=df.bfill()


dados_calc['media_temp']={'D': {'central': df}}




#nwp grid
for a in range(1,7):
  for b in range(1,7):
    
    df=pd.DataFrame()
    aux=nwpgrid['D'][str(a)+'x'+str(b)]
    
    #frac. total
    df=df.assign(ftotal_med3=aux[aux.columns[3]].rolling(window=3, center=True).mean())
    df=df.assign(ftotal_med7=aux[aux.columns[3]].rolling(window=7, center=True).mean())
    df=df.assign(ftotal_med11=aux[aux.columns[3]].rolling(window=11, center=True).mean())
    
    #frac. baixa
    df=df.assign(fbaixa_med3=aux[aux.columns[4]].rolling(window=3, center=True).mean())
    df=df.assign(fbaixa_med7=aux[aux.columns[4]].rolling(window=7, center=True).mean())
    df=df.assign(fbaixa_med11=aux[aux.columns[4]].rolling(window=11, center=True).mean())
    
    #frac. media
    df=df.assign(fmedia_med3=aux[aux.columns[5]].rolling(window=3, center=True).mean())
    df=df.assign(fmedia_med7=aux[aux.columns[5]].rolling(window=7, center=True).mean())
    df=df.assign(fmedia_med11=aux[aux.columns[5]].rolling(window=11, center=True).mean())

    #frac. alta
    df=df.assign(falta_med3=aux[aux.columns[6]].rolling(window=3, center=True).mean())
    df=df.assign(falta_med7=aux[aux.columns[6]].rolling(window=7, center=True).mean())
    df=df.assign(falta_med11=aux[aux.columns[6]].rolling(window=11, center=True).mean())

    #irradiancia normalizada
    df=df.assign(irrn_med3=aux[aux.columns[8]].rolling(window=3, center=True).mean())
    df=df.assign(irrn_med7=aux[aux.columns[8]].rolling(window=7, center=True).mean())
    df=df.assign(irrn_med11=aux[aux.columns[8]].rolling(window=11, center=True).mean())
    df=df.assign(irrn_med25=aux[aux.columns[8]].rolling(window=25, center=True).mean())


    #Retirar nan iniciais e finais - colocar igual a previsao imediatamente anterior/seguinte
    df=df.ffill()
    df=df.bfill()
  

    dados_calc['media_temp']['D'][str(a)+'x'+str(b)]=df








#desvio padrao espacial


aux=pd.DataFrame()
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([df, nwpgrid['D'][str(a)+'x'+str(b)]], axis=1, sort=True)

#frac. total    
df1=df.filter(like='Fraccion total', axis=1)
aux=aux.assign(ftotal_dps=df1.std(axis=1))

    
#frac. baixa
df1=df.filter(like='Frac. Baja', axis=1)
aux=aux.assign(fbaixa_dps=df1.std(axis=1))
    
#frac. media
df1=df.filter(like='Frac. Media', axis=1)
aux=aux.assign(fmedia_dps=df1.std(axis=1))

#frac. alta
df1=df.filter(like='Frac. Alta', axis=1)
aux=aux.assign(falta_dps=df1.std(axis=1))


#irradiancia normalizada
df1=df.filter(like='irr_norm', axis=1)
aux=aux.assign(irrn_dps=df1.std(axis=1))

    
dados_calc['dp_esp']={'D': aux}   
 
    
    
    







#Média 4 previsoes ____________________________________________________________


#pesos
span=4
alpha=2/(span+1)
w=[(1-alpha)**0, (1-alpha)**1, (1-alpha)**2, (1-alpha)**3]


aux=pd.DataFrame()
dados_calc['media_prev']={}
#media dos pts da grid
  
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([nwpgrid['D'][str(a)+'x'+str(b)], nwpgrid['D+1'][str(a)+'x'+str(b)], nwpgrid['D+2'][str(a)+'x'+str(b)], nwpgrid['D+3'][str(a)+'x'+str(b)]], axis=1, sort=False)
    
    
    #irradiancia normalizada
    df1=df.filter(like='irr_norm', axis=1)
    aux=aux.assign(irrn_med_prev=np.average(df1, axis=1, weights=w), )
    
    #frac. total
    df1=df.filter(like='Fraccion total', axis=1)
    aux=aux.assign(ftotal_med_prev=np.average(df1, axis=1, weights=w))
    
    #frac. baixa
    df1=df.filter(like='Frac. Baja', axis=1)
    aux=aux.assign(fbaixa_med_prev=np.average(df1, axis=1, weights=w))
    
    #frac. media
    df1=df.filter(like='Frac. Media', axis=1)
    aux=aux.assign(fmedia_med_prev=np.average(df1, axis=1, weights=w))
    
    #frac. alta
    df1=df.filter(like='Frac. Alta', axis=1)
    aux=aux.assign(falta_med_prev=np.average(df1, axis=1, weights=w))
    
    dados_calc['media_prev'][str(a)+'x'+str(b)]= aux
    dados_calc['media_prev'][str(a)+'x'+str(b)].index= df.index

    


aux=pd.DataFrame()    
    
#media central
df=pd.concat([nwpgrid['D']['central'], nwpgrid['D+1']['central'], nwpgrid['D+2']['central'], nwpgrid['D+3']['central']], axis=1, sort=True)

#temperatura
df1=df.filter(like='Temperatura', axis=1)
aux=aux.assign(temp_med_prev=np.average(df1, axis=1, weights=w)) #com .mean() funciona

#pressao
df1=df.filter(like='Presión', axis=1) 
aux=aux.assign(pres_med_prev=np.average(df1, axis=1, weights=w)) #erro

#velocidade
df1=df.filter(like='Velocidad', axis=1)
aux=aux.assign(vel_med_prev=np.average(df1, axis=1, weights=w))
   
#velocidade
df1=df.filter(like='Dirección', axis=1)
aux=aux.assign(dir_med_prev=np.average(df1, axis=1, weights=w))

#humidade relativa
df1=df.filter(like='Humedad rel', axis=1)
aux=aux.assign(hum_med_prev=np.average(df1, axis=1, weights=w))

#frac. total
df1=df.filter(like='Fraccion total.36', axis=1)
aux=aux.assign(ftotal_med_prev=np.average(df1, axis=1, weights=w))
    
#frac. baixa
df1=df.filter(like='Frac. Baja.36', axis=1)
aux=aux.assign(fbaixa_med_prev=np.average(df1, axis=1, weights=w))
    
#frac. media
df1=df.filter(like='Frac. Media.36', axis=1)
aux=aux.assign(fmedia_med_prev=np.average(df1, axis=1, weights=w))

#frac. alta
df1=df.filter(like='Frac. Alta', axis=1)
aux=aux.assign(falta_med_prev=np.average(df1, axis=1, weights=w))

#precipitacao
df1=df.filter(like='Precipitación', axis=1)
aux=aux.assign(prec_med_prev=np.average(df1, axis=1, weights=w))

#irradiancia normalizada
df1=df.filter(like='irr_norm', axis=1)
aux=aux.assign(irrn_med_prev=np.average(df1, axis=1, weights=w))

dados_calc['media_prev']['central']=aux
dados_calc['media_prev']['central'].index= df.index



#Retirar nan dos dias 1-1-2017, 2-1-2017 e 3-1-2017 
#colocar igual a previsao do dia seguinte a mesma hora
dados_calc['media_prev']['central'].iloc[47:71,:] = dados_calc['media_prev']['central'].iloc[71:95,:].values
dados_calc['media_prev']['central'].iloc[23:47,:] = dados_calc['media_prev']['central'].iloc[71:95,:].values
dados_calc['media_prev']['central'].iloc[0:23,:] = dados_calc['media_prev']['central'].iloc[72:95,:].values

for a in range(1,7):
  for b in range(1,7):
    dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[47:71,:] = dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[71:95,:].values
    dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[23:47,:] = dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[71:95,:].values
    dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[0:23,:] = dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[72:95,:].values


#Retirar nan do dia 31/05/2020
#colocar igual a previsao do dia anterior a mesma hora
dados_calc['media_prev']['central'].iloc[-24:,:] = dados_calc['media_prev']['central'].iloc[-48:-24,:].values

for a in range(1,7):
  for b in range(1,7):
    dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[-24:,:] = dados_calc['media_prev'][str(a)+'x'+str(b)].iloc[-48:-24,:].values




#Retirar nan das 00:00:00 horas
#media de toda a df preenchida com o elemento anterior com toda a df preenchida com o elemento seguinte - valores nan dao a media dos 2 e os outros mantem-se
dados_calc['media_prev']['central'] = (dados_calc['media_prev']['central'].ffill()+dados_calc['media_prev']['central'].bfill())/2

for a in range(1,7):
  for b in range(1,7):
    dados_calc['media_prev'][str(a)+'x'+str(b)] = (dados_calc['media_prev'][str(a)+'x'+str(b)].ffill()+dados_calc['media_prev'][str(a)+'x'+str(b)].bfill())/2







#Spatial Smoothing ____________________________________________________________

#dividir em camadas
layer1=pd.DataFrame() #df layer 1
layer2=pd.DataFrame() #df layer 2
layer3=pd.DataFrame() #df layer 3

for a in range(1,7):
  for b in range(1,7):
    if(a==1 or b==1 or a==6 or b==6):
      layer3=pd.concat((layer3, nwpgrid['D'][str(a)+'x'+str(b)]), axis=1, sort=True)
    elif(str(a)+str(b)==str(33) or str(a)+str(b)==str(34) or str(a)+str(b)==str(44) or str(a)+str(b)==str(43)):
      layer1=pd.concat((layer1, nwpgrid['D'][str(a)+'x'+str(b)]), axis=1, sort=True)
    else:
      layer2=pd.concat((layer2, nwpgrid['D'][str(a)+'x'+str(b)]), axis=1, sort=True)

#media horaria da irradiancia em cada layer
layer1=layer1.assign(irr_med1=layer1.mean(axis=1))
layer2=layer2.assign(irr_med2=layer2.mean(axis=1))
layer3=layer3.assign(irr_med3=layer3.mean(axis=1))


aux=pd.concat((layer1['irr_med1'], layer2['irr_med2'], layer3['irr_med3']), axis=1, sort=True)

#pesos
span=4
alpha=2/(span+1)
w=[(1-alpha)**0, (1-alpha)**1, (1-alpha)**2]

#media
aux['media']=np.average(aux, axis=1, weights=w) 
dados_calc['spatial_smoothing']= {'D': {'layers': aux[['irr_med1', 'irr_med2', 'irr_med3']],
                                        'media': aux['media']}}














#PCA __________________________________________________________________________

'''indicacao paper: melhores resultados obtidos para PCA aplicado individualmente
a cada serie de dados de cada ponto da grid - tambem inclui o da central'''


#3282 variáveis -> PCA -> 983 variáveis

prin_comp95 = {
    'D': {},
    'D+1': {},
    'D+2': {},
    'D+3': {}
}

prin_comp89 = {
    'D': {},
    'D+1': {},
    'D+2': {},
    'D+3': {}
}


#D central 
cols = [0,2,3,4,5,6,7,8,9,10,13] #so variaveis meteorologicas

df=nwpgrid['D']['central'].iloc[:, cols]
df=df.assign(elev_solar=nwpgrid['D']['elev_solar'])
df.set_index(nwpgrid['D']['central'].index, inplace = True)


pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)


pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['D']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)


#D grid
cols = [3,4,5,6,8] #so variaveis meteorologicas

for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D'][str(a)+'x'+str(b)].iloc[:, cols]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['D'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)


#D producao
df=nwpgrid['D']['central']['pot_norm']
prin_comp89['D']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D']['producao']=pd.DataFrame(data=df, index=df.index)







#D+1 central
cols = [0,2,3,4,5,6,7,8,9,10,13] 
df=nwpgrid['D+1']['central'].iloc[:, cols]
df=df.assign(elev_solar=nwpgrid['D+1']['elev_solar'])
df.set_index(nwpgrid['D+1']['central'].index, inplace = True)


pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D+1']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['D+1']['central']=pd.DataFrame(data=pca.transform(df), index=df.index) 


#D+1 grid                 
cols = [3,4,5,6,8] #colunas das variaveis 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D+1'][str(a)+'x'+str(b)].iloc[:, cols]
      df.set_index(nwpgrid['D+1'][str(a)+'x'+str(b)].index, inplace = True)

      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D+1'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['D+1'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

#D+1 producao
df=nwpgrid['D+1']['central']['pot_norm']
prin_comp89['D+1']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D+1']['producao']=pd.DataFrame(data=df, index=df.index)




#D+2 central
cols = [0,2,3,4,5,6,7,8,9,10,13]
df=nwpgrid['D+2']['central'].iloc[:, cols]
df=df.assign(elev_solar=nwpgrid['D+2']['elev_solar'])
df.set_index(nwpgrid['D+2']['central'].index, inplace = True)


pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D+2']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['D+2']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)  
      
      
#D+2 grid
cols = [3,4,5,6,8] #colunas das variaveis 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D+2'][str(a)+'x'+str(b)].iloc[:, cols]
      df.set_index(nwpgrid['D+2'][str(a)+'x'+str(b)].index, inplace = True)

      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D+2'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['D+2'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

#D+2 producao
df=nwpgrid['D+2']['central']['pot_norm']
prin_comp89['D+2']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D+2']['producao']=pd.DataFrame(data=df, index=df.index)



#D+3 central
cols = [0,2,3,4,5,6,7,8,9,10,13]
df=nwpgrid['D+3']['central'].iloc[:, cols]
df=df.assign(elev_solar=nwpgrid['D+3']['elev_solar'])
df.set_index(nwpgrid['D+3']['central'].index, inplace = True)


pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D+3']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['D+3']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)      
      
#D+3 grid
cols = [3,4,5,6,8] #colunas das variaveis 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D+3'][str(a)+'x'+str(b)].iloc[:, cols]
      df.set_index(nwpgrid['D+3'][str(a)+'x'+str(b)].index, inplace = True)


      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D+3'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['D+3'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)


#D+3 producao
df=nwpgrid['D+3']['central']['pot_norm']
prin_comp89['D+3']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D+3']['producao']=pd.DataFrame(data=df, index=df.index)







#dados calculados

prin_comp95['dados_calc']={}
prin_comp89['dados_calc']={}
prin_comp95['dados_calc']['lags']={}
prin_comp89['dados_calc']['lags']={}
prin_comp95['dados_calc']['leads']={}
prin_comp89['dados_calc']['leads']={}
prin_comp95['dados_calc']['media_prev']={}
prin_comp89['dados_calc']['media_prev']={}
prin_comp95['dados_calc']['media_temp']={}
prin_comp89['dados_calc']['media_temp']={}
prin_comp95['dados_calc']['var_temp']={}
prin_comp89['dados_calc']['var_temp']={}

df=dados_calc['dp_esp']['D']


pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['dados_calc']['dp_esp']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['dados_calc']['dp_esp']=pd.DataFrame(data=pca.transform(df), index=df.index)



for a in range(1,7):
    for b in range(1,7):      
      df=dados_calc['lags']['D'][str(a)+'x'+str(b)]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['dados_calc']['lags'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['dados_calc']['lags'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
          
      df=dados_calc['leads']['D'][str(a)+'x'+str(b)]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['dados_calc']['leads'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['dados_calc']['leads'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      

      df=dados_calc['media_prev'][str(a)+'x'+str(b)]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['dados_calc']['media_prev'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['dados_calc']['media_prev'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      

      df=dados_calc['media_temp']['D'][str(a)+'x'+str(b)]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['dados_calc']['media_temp'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['dados_calc']['media_temp'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      

      df=dados_calc['var_temp']['D'][str(a)+'x'+str(b)]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['dados_calc']['var_temp'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

      pca = PCA(.89) #numero componentes que prefazem 89%
      pca.fit(df)
      prin_comp89['dados_calc']['var_temp'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      

      
df=dados_calc['lags']['D']['central']

pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['dados_calc']['lags']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['dados_calc']['lags']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)
      
    
df=dados_calc['leads']['D']['central']
      
pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['dados_calc']['leads']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['dados_calc']['leads']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)


df=dados_calc['media_prev']['central']
      
pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['dados_calc']['media_prev']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['dados_calc']['media_prev']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)



df=dados_calc['media_temp']['D']['central']
      
pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['dados_calc']['media_temp']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['dados_calc']['media_temp']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)


df=dados_calc['var_temp']['D']['central']
      
pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['dados_calc']['var_temp']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.89) #numero componentes que prefazem 89%
pca.fit(df)
prin_comp89['dados_calc']['var_temp']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)


prin_comp95['dados_calc']['spatial_smoothing']=dados_calc['spatial_smoothing']['D']['media']
prin_comp89['dados_calc']['spatial_smoothing']=dados_calc['spatial_smoothing']['D']['media']




#print(pca.explained_variance_ratio_)







##### Weather to Power Model ##################################################
      
      
#treino - 1/1/2020-31/5/2020; teste - 1/1/2017-31/12/2019 _____________________
#principal components em todos os pontos - X
#potencia produzida D+1 - Y


#TREINO
d0=0 #numero colunas dia 0
d1=0 #numero colunas dia 1
d2=0 #numero colunas dia 2
d3=0 #numero colunas dia 3

#D
df=pd.DataFrame()
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      d0+=prin_comp89['D'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
d0+=prin_comp89['D']['central'].shape[1]


#D+1
d1+=d0      
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D+1'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      d1+=prin_comp89['D+1'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D+1']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
d1+=prin_comp89['D+1']['central'].shape[1]


#D+2
d2+=d1  
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D+2'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      d2+=prin_comp89['D+2'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D+2']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
d2+=prin_comp89['D+2']['central'].shape[1]

#D+3
d3+=d2   
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D+3'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
      d3+=prin_comp89['D+3'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D+3']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
d3+=prin_comp89['D+3']['central'].shape[1]


df.columns = range(df.shape[1])      


#Retirar nan das 00:00:00 horas do dia D
#media de toda a df preenchida com o elemento anterior com toda a df preenchida com o elemento seguinte - valores nan dao a media dos 2 e os outros mantem-se
df.iloc[:,0:d0] = (df.iloc[:,0:d0].ffill()+df.iloc[:,0:d0].bfill())/2

#Retirar nan do dia 1-1-2017 na previsao D+1
#colocar igual a previsao do dia seguinte a mesma hora
df.iloc[0:23,d0:d1] = df.iloc[24:47,d0:d1].values

#Retirar nan dos dias 1-1-2017 e 2-1-2017 na previsao D+2
#colocar igual a previsao do dia seguinte a mesma hora
df.iloc[23:47,d1:d2] = df.iloc[47:71,d1:d2].values
df.iloc[0:23,d1:d2] = df.iloc[48:71,d1:d2].values

#Retirar nan dos dias 1-1-2017, 2-1-2017 e 3-1-2017 na previsao D+3
#colocar igual a previsao do dia seguinte a mesma hora
df.iloc[47:71,d2:d3] = df.iloc[71:95,d2:d3].values
df.iloc[23:47,d2:d3] = df.iloc[71:95,d2:d3].values
df.iloc[0:23,d2:d3] = df.iloc[72:95,d2:d3].values

#tirar a x_train os dias que faltam a y_train para ficar sincrono
df=pd.concat([df.loc['2017-01-02':'2019-02-27'], df.loc['2019-03-03':'2019-04-05'], df.loc['2019-04-09':'2019/12/31']])






df1=pd.DataFrame()
#dados calculados
for a in range(1,7):
    for b in range(1,7):
      df1=pd.concat([df1, prin_comp89['dados_calc']['lags'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['leads'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['media_prev'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['media_temp'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['var_temp'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      

df1=pd.concat([df1, prin_comp89['dados_calc']['lags']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['leads']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['media_prev']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['media_temp']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['var_temp']['central'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      

df1=pd.concat([df1, prin_comp89['dados_calc']['dp_esp'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
df1=pd.concat([df1, prin_comp89['dados_calc']['spatial_smoothing'].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)



#tirar a x_train os dias que faltam a y_train para ficar sincrono
df1=pd.concat([df1.loc['2017-01-02':'2019-02-27'], df1.loc['2019-03-03':'2019-04-05'], df1.loc['2019-04-09':'2019/12/31']])
df1 = (df1.ffill()+df1.bfill())/2

df=pd.concat([df, df1], axis=1, sort=True)      

df=df.loc['2017-01-04':'2019/12/31']


df.columns = range(df.shape[1])      
x_train=df 


#x_train.isnull().sum().sum()
''' onde estao os nulls
df_null = x_train.isnull().unstack()
t = df_null[df_null]'''


y_train = prin_comp89['D+1']['producao'].loc['2017-01-01':'2019/12/31']
y_train=pd.concat([y_train.loc['2017-01-04':'2019-02-27'], y_train.loc['2019-03-03':'2019-04-05'], y_train.loc['2019-04-09':'2019/12/31']])

#y_train.isnull().sum().sum()

'''Detetar diferencas entre x_train e y_train
y=np.where(x_train.index.isin(y_train.index) == False)'''






#TESTE


#D
df=pd.DataFrame()
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      d0+=prin_comp89['D'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
d0+=prin_comp89['D']['central'].shape[1]


#D+1
d1+=d0      
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D+1'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      d1+=prin_comp89['D+1'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D+1']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
d1+=prin_comp89['D+1']['central'].shape[1]


#D+2
d2+=d1  
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D+2'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      d2+=prin_comp89['D+2'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D+2']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
d2+=prin_comp89['D+2']['central'].shape[1]

#D+3
d3+=d2   
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp89['D+3'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
      d3+=prin_comp89['D+3'][str(a)+'x'+str(b)].shape[1]

df=pd.concat([df, prin_comp89['D+3']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
d3+=prin_comp89['D+3']['central'].shape[1]


df.columns = range(df.shape[1])      


#Retirar nan das 00:00:00 horas do dia D
#media de toda a df preenchida com o elemento anterior com toda a df preenchida com o elemento seguinte - valores nan dao a media dos 2 e os outros mantem-se
df.iloc[:,0:d0] = (df.iloc[:,0:d0].ffill()+df.iloc[:,0:d0].bfill())/2

#dia 31/05/2020 da previsao D
df.iloc[-24:,0:d0] = df.iloc[-48:-24,0:d0].values

#primeiro dia as 00:00:00 igual a 01:00:00 desse dia
df.iloc[:,0:d0] = df.iloc[:,0:d0].bfill()







df1=pd.DataFrame()
#dados calculados
for a in range(1,7):
    for b in range(1,7):
      df1=pd.concat([df1, prin_comp89['dados_calc']['lags'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
      df1=pd.concat([df1, prin_comp89['dados_calc']['leads'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['media_prev'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['media_temp'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      df1=pd.concat([df1, prin_comp89['dados_calc']['var_temp'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      

df1=pd.concat([df1, prin_comp89['dados_calc']['lags']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['leads']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['media_prev']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['media_temp']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
df1=pd.concat([df1, prin_comp89['dados_calc']['var_temp']['central'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      

df1=pd.concat([df1, prin_comp89['dados_calc']['dp_esp'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
df1=pd.concat([df1, prin_comp89['dados_calc']['spatial_smoothing'].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)


#retirar os nan
df1 = (df1.ffill()+df1.bfill())/2
df1 = df1.loc['2020-01-01':'2020/05/30']
df1 = df1.bfill()

df=pd.concat([df, df1], axis=1, sort=True)      
df=df.loc['2020-01-01':'2020/05/30']
df.columns = range(df.shape[1])      
x_test=df 


y_test = prin_comp89['D+1']['producao'].loc['2020-01-01':'2020/05/30']


































def mape(test_array, pred_array): 
    test_array, pred_array = np.array(test_array), np.array(pred_array)
    return np.mean(np.abs((test_array - pred_array) / np.mean(test_array)))
  
  
  
#XGBoost_______________________________________________________________________

          
'''teste automatico
i=0
results = np.ones((5,20)) 
for a in [0,0.5,0.9,1]:
  params = {'eta': 0.07,
            'gamma': 0.01,
            'max_depth': 8, 
            
            'min_child_weight': 1,
            'max_delta_step': 0,
            'lambda': 1,
            'alpha': 0.15,
            }   
  xg = xgb.XGBRegressor(objective ='reg:squarederror', **params)
  model=xg.fit(x_train.values, y_train.values)
  y_pred = model.predict(x_test.values)  
  y_pred = y_pred = y_pred.reshape(-1, 1)   
  xg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  xg_mae=mean_absolute_error(y_test, y_pred)    
  xg_mape=mape(y_test, y_pred)
  xg_crps=ps.crps_gaussian(y_pred, mu=0, sig=1).mean()

  results[0,i]=a
  results[1,i]=xg_rmse
  results[2,i]=xg_mae
  results[3,i]=xg_mape
  results[4,i]=xg_crps
  i+=1
  
results[3,:].min() #mape
results[3,:].argmin() #mape

results[4,:].min() #crps
results[4,:].argmin() #crps
'''



params = {'eta': 0.07,
          'gamma': 0.01,
          'max_depth': 8, 
          
          'min_child_weight': 1,
          'max_delta_step': 0,
          'lambda': 1,
          'alpha': 0.15,
          }    

  
xg = xgb.XGBRegressor(objective ='reg:squarederror', **params)

model=xg.fit(x_train.values, y_train.values)

y_pred = model.predict(x_test.values) 
y_pred = y_pred.reshape(-1, 1)   
    
xg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
xg_rmse

xg_mae=mean_absolute_error(y_test, y_pred)      
xg_mae

xg_mape=mape(y_test, y_pred)
xg_mape

xg_crps=ps.crps_gaussian(y_pred, mu=0, sig=1).mean()
xg_crps

    





#CatBoost______________________________________________________________________
   
params = {'loss_function': 'MAPE', # objective function
          'eval_metric': 'MAPE', # metric
          
          'learning_rate': 0.07,
          'depth': 8,
          'l2_leaf_reg': 1
         }
cb = CatBoostRegressor(**params)
model=cb.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True, plot=True);
   
y_pred = model.predict(x_test.values)
y_pred = y_pred.reshape(-1, 1)   

cb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cb_rmse

cb_mae=mean_absolute_error(y_test, y_pred)      
cb_mae   
    
cb_mape=mape(y_test, y_pred)
cb_mape

cb_crps=ps.crps_gaussian(y_pred, mu=0, sig=1).mean()
cb_crps
    







#LightGBM______________________________________________________________________

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

params = {
    'application': 'mape',
    'objective': 'mape',
    'learning_rate': 0.07,
    'max_depth': 8,
    'min_sum_hessian_in_leaf':1,
    'max_delta_step': 0,
    'lambda_l1': 0.15,
    'lambda_l2':1
}

model = lgb.train(params, train_data, valid_sets=test_data, num_boost_round=5000)

y_pred = model.predict(x_test.values)
y_pred = y_pred.reshape(-1, 1)   

lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lgb_rmse

lgb_mae=mean_absolute_error(y_test, y_pred)      
lgb_mae
    
lgb_mape=mape(y_test, y_pred)
lgb_mape

lgb_crps=ps.crps_gaussian(y_pred, mu=0, sig=1).mean()
lgb_crps
    

  
    

  
#GBT___________________________________________________________________________

#definir hyperparameters que fala no paper
params = {'max_depth': 8, #entre 5 e 9
          'min_samples_split': 150, #entre 150 e 350
          'min_samples_leaf': 20, #entre 20 e 80
          'max_features': 'sqrt', #square root of total number of features
          'learning_rate': 0.07, #entre 0.01 e 0.05 
          'n_estimators': 500, #entre 500 e 800 
          'subsample': 0.8 #80%
          }


#criar regressor
gbt = GradientBoostingRegressor(**params)
#treinar modelo
model=gbt.fit(x_train, y_train.values.ravel())
#prever resposta ao dataset de teste
y_pred=model.predict(x_test)
y_pred = y_pred.reshape(-1, 1)   



gbt_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
gbt_rmse

gbt_mae=mean_absolute_error(y_test, y_pred)      
gbt_mae

gbt_mape=mape(y_test, y_pred)
gbt_mape

gbt_crps=ps.crps_gaussian(y_pred, mu=0, sig=1).mean()
gbt_crps



    
    
    
    
    
    
