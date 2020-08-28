################################
# PV Probabilistic Forecasting #
################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
#from sklearn import 
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3




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
teste_lims.filter(like='Precipitación', axis=1).min().min(axis=0) #numero negativo - considerar 0?
teste_lims.filter(like='Elev. Solar (º)', axis=1).min().min(axis=0)
teste_lims.filter(like='Produção (MW)', axis=1).min().min(axis=0)


#testar NaN ___________________________________________________________________

dados["D"].isnull().sum().sum(axis=0)
dados["D+1"].isnull().sum().sum(axis=0)
dados["D+2"].isnull().sum().sum(axis=0)
dados["D+3"].isnull().sum().sum(axis=0)




##### Descritive statistics ###################################################


report = pp.ProfileReport(dados['D'])
report.to_file('profile_report.html') #erro ?




##### Organizacao dados #######################################################

#NWP grid _____________________________________________________________________

#D 
nwp_grid0 = {
        "central": dados["D"].iloc[:, 183:194],
        "elev_solar": dados["D"].iloc[:, 194]
}
i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid0[str(a)+'x'+str(b)] = dados["D"].iloc[:, cols]
        i=i+5
        

#D+1
nwp_grid1 = {
        "central": dados["D+1"].iloc[:, 183:194],
        "elev_solar": dados["D+1"].iloc[:, 194]
}
i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid1[str(a)+'x'+str(b)] = dados["D+1"].iloc[:, cols]
        i=i+5


#D+2
nwp_grid2 = {
        "central": dados["D+2"].iloc[:, 183:194],
        "elev_solar": dados["D+2"].iloc[:, 194]
}
i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid2[str(a)+'x'+str(b)] = dados["D+2"].iloc[:, cols]
        i=i+5


#D+3
nwp_grid3 = {
        "central": dados["D+3"].iloc[:, 183:194],
        "elev_solar": dados["D+3"].iloc[:, 194]
}
i=3
for a in range(1,7):
    for b in range(1,7):
        cols = [0,1, i, i+1, i+2, i+3, i+4]
        nwp_grid3[str(a)+'x'+str(b)] = dados["D+3"].iloc[:, cols]
        i=i+5
        
nwpgrid = {
        "D": nwp_grid0,
        "D+1": nwp_grid1,
        "D+2": nwp_grid2,
        "D+3": nwp_grid3
}

        
#potencia gerada ______________________________________________________________

producao = {
        "D": dados["D"].iloc[:, 195],
        "D+1": dados["D+1"].iloc[:, 195],
        "D+2": dados["D+2"].iloc[:, 195],
        "D+3": dados["D+3"].iloc[:, 195]
}


#localizacao dos pontos da grid________________________________________________

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


#Clear-sky model - adicionar colunas Ics e Inorm ______________________________
        
for dia in [0,1,2,3]:
    for a in [1,2,3,4,5,6]:
        for b in [1,2,3,4,5,6]:
            local = Location(location[a,b,0], location[a,b,1])
            
            if dia==0:
                df=nwpgrid['D'][str(a)+'x'+str(b)] 
                time = dados['D'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D'][str(a)+'x'+str(b)]=df 
                
            elif dia==1:
                df=nwpgrid['D+1'][str(a)+'x'+str(b)] 
                time = dados['D+1'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D+1'][str(a)+'x'+str(b)]=df 
                
            elif dia==2:
                df=nwpgrid['D+2'][str(a)+'x'+str(b)] 
                time = dados['D+2'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D+2'][str(a)+'x'+str(b)]=df 
            
            elif dia==3:
                df=nwpgrid['D+3'][str(a)+'x'+str(b)] 
                time = dados['D+3'].index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D+3'][str(a)+'x'+str(b)]=df 
        
        
    
'''teste do csm num ponto aleatorio da grid -> OK!
local = Location(location[6,6,0], location[6,6,1])
time = dados['D+2'].index
cs = local.get_clearsky(time)
'''       
        
        
#Weighted quantile regression _________________________________________________

aux = {'irradiancia': dados['D']['Radiación.36'],
       'potencia': dados['D']['Produção (MW)']
}

#scatter plot dos pts  I e P na central
plt.scatter(aux['irradiancia'], aux['potencia'])
plt.xlabel('Irradiancia')
plt.ylabel('Potencia')


data = pd.DataFrame (aux, columns = ['irradiancia','potencia'])
mod = smf.quantreg('potencia ~ irradiancia', data)
res = mod.fit(q=.5)
print(res.summary())

quantiles = np.arange(.05, .96, .1)

def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['irradiancia']] + res.conf_int().loc['irradiancia'].tolist()

models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b', 'cf_lower_bound', 'cf_upper_bound'])

#minimos quadrados ordinarios p/ comparar com qr
ols = smf.ols('potencia ~ irradiancia', data).fit()
ols_ci = ols.conf_int().loc['irradiancia'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['irradiancia'],
           lb = ols_ci[0],
           ub = ols_ci[1])

#comparacao ols vs qr
print(models)
print(ols)

#plot1
x = np.arange(data.irradiancia.min(), data.irradiancia.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')

y = get_y(ols['a'], ols['b'])

ax.plot(x, y, color='red', label='OLS')
ax.scatter(data['irradiancia'], data['potencia'], alpha=.05)
legend = ax.legend()
ax.set_xlabel('Irradiancia')
ax.set_ylabel('Potência');


#plot2
n = models.shape[0]
p1 = plt.plot(models.q, models.b, color='black', label='Quantile Reg.')
p2 = plt.plot(models.q, models.cf_upper_bound, linestyle='dotted', color='black')
p3 = plt.plot(models.q, models.cf_lower_bound, linestyle='dotted', color='black')
p4 = plt.plot(models.q, [ols['b']] * n, color='red', label='OLS')
p5 = plt.plot(models.q, [ols['lb']] * n, linestyle='dotted', color='red')
p6 = plt.plot(models.q, [ols['ub']] * n, linestyle='dotted', color='red')
plt.ylabel(r'$\beta_{irradiancia}$')
plt.xlabel('Quantis')
plt.legend()
plt.show()




#Apenas dia D _________________________________________________________________

#variancia temporal -> apliquei a todas as variaveis

for a in [1,2,3,4,5,6]:
  for b in [1,2,3,4,5,6]:
    
    df=nwpgrid['D'][str(a)+'x'+str(b)]
    
    #irradiancia
    df=df.assign(irr_var3=df[df.columns[2]].rolling(window=3, center=True).var())
    df=df.assign(irr_var7=df[df.columns[2]].rolling(window=7, center=True).var())
    df=df.assign(irr_var11=df[df.columns[2]].rolling(window=11, center=True).var())
    
    #frac. total
    df=df.assign(ftotal_var3=df[df.columns[3]].rolling(window=3, center=True).var())
    df=df.assign(ftotal_var7=df[df.columns[3]].rolling(window=7, center=True).var())
    df=df.assign(ftotal_var11=df[df.columns[3]].rolling(window=11, center=True).var())
    
    #frac. baixa
    df=df.assign(fbaixa_var3=df[df.columns[4]].rolling(window=3, center=True).var())
    df=df.assign(fbaixa_var7=df[df.columns[4]].rolling(window=7, center=True).var())
    df=df.assign(fbaixa_var11=df[df.columns[4]].rolling(window=11, center=True).var())
    
    #frac. media
    df=df.assign(fmedia_var3=df[df.columns[5]].rolling(window=3, center=True).var())
    df=df.assign(fmedia_var7=df[df.columns[5]].rolling(window=7, center=True).var())
    df=df.assign(fmedia_var11=df[df.columns[5]].rolling(window=11, center=True).var())
    
    #frac. alta
    df=df.assign(falta_var3=df[df.columns[6]].rolling(window=3, center=True).var())
    df=df.assign(falta_var7=df[df.columns[6]].rolling(window=7, center=True).var())
    df=df.assign(falta_var11=df[df.columns[6]].rolling(window=11, center=True).var())
    
    
    nwpgrid['D'][str(a)+'x'+str(b)]=df



#lags -> apliquei a todas as variaveis
for a in [1,2,3,4,5,6]:
  for b in [1,2,3,4,5,6]:
    
    df=nwpgrid['D'][str(a)+'x'+str(b)]
    
    #irradiancia
    df=df.assign(irr_lag1=df[df.columns[2]].shift(periods=1))
    df=df.assign(irr_lag2=df[df.columns[2]].shift(periods=2))
    df=df.assign(irr_lag3=df[df.columns[2]].shift(periods=3))
    
    ##frac. total
    df=df.assign(ftotal_lag1=df[df.columns[3]].shift(periods=1))
    df=df.assign(ftotal_lag2=df[df.columns[3]].shift(periods=2))
    df=df.assign(ftotal_lag3=df[df.columns[3]].shift(periods=3))
    
    #frac. baixa
    df=df.assign(fbaixa_lag1=df[df.columns[4]].shift(periods=1))
    df=df.assign(fbaixa_lag2=df[df.columns[4]].shift(periods=2))
    df=df.assign(fbaixa_lag3=df[df.columns[4]].shift(periods=3))
    
    #frac. media
    df=df.assign(fmedia_lag1=df[df.columns[5]].shift(periods=1))
    df=df.assign(fmedia_lag2=df[df.columns[5]].shift(periods=2))
    df=df.assign(fmedia_lag3=df[df.columns[5]].shift(periods=3))
    
    #frac. alta
    df=df.assign(falta_lag1=df[df.columns[6]].shift(periods=1))
    df=df.assign(falta_lag2=df[df.columns[6]].shift(periods=2))
    df=df.assign(falta_lag3=df[df.columns[6]].shift(periods=3))
    
    nwpgrid['D'][str(a)+'x'+str(b)]=df


    
    





