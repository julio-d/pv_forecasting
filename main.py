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
from pvlib.iotools import read_tmy3
from sklearn.decomposition import PCA




##### Importar dados ##########################################################


filepath = r'C:\Users\Júlio\Desktop\pv_forecasting\dados\Dados_sincronizados_Alentejo_V3.xlsx'

dados = {
        "D": pd.read_excel(filepath, sheet_name = "D", header = [1]),
        "D+1": pd.read_excel(filepath, sheet_name = "D+1", header = [1]),
        "D0+2": pd.read_excel(filepath, sheet_name = "D+2", header = [1]),
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


report=ProfileReport(dados['D'])  
report.to_file(output_file='report.html')

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
nwp_grid1['central']=nwp_grid1['central'].assign(producao=dados["D+1"].iloc[:, 195].values)

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
nwp_grid2['central']=nwp_grid2['central'].assign(producao=dados["D+2"].iloc[:, 195].values)

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
nwp_grid3['central']=nwp_grid3['central'].assign(producao=dados["D+3"].iloc[:, 195].values)

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

''' DUVIDA: na serie de potencia ha varios dias sem producao - 20/1/2017-3/2/2017
ignoro estes dados nulos de producao ou tomo alguma medida? '''


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
df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[1]]/df[df.columns[12]], df['irr_norm'])  
nwpgrid['D']['central']=df 
  

df=nwpgrid['D+1']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[1]]/df[df.columns[12]], df['irr_norm'])  
nwpgrid['D+1']['central']=df 
  
df=nwpgrid['D+2']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[1]]/df[df.columns[12]], df['irr_norm'])  
nwpgrid['D+2']['central']=df         
  
  
df=nwpgrid['D+3']['central'] 
time = df.index
cs = local.get_clearsky(time)  # df com ghi, dni, dhi
df=df.assign(csm_ghi=cs['ghi'].values)
df=df.assign(irr_norm=df['csm_ghi'].values)
df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[1]]/df[df.columns[12]], df['irr_norm'])  
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
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D'][str(a)+'x'+str(b)]=df 
                
            elif dia==1:
                df=nwpgrid['D+1'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D+1'][str(a)+'x'+str(b)]=df 
                
            elif dia==2:
                df=nwpgrid['D+2'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D+2'][str(a)+'x'+str(b)]=df 
            
            elif dia==3:
                df=nwpgrid['D+3'][str(a)+'x'+str(b)] 
                time = df.index
                cs = local.get_clearsky(time)  # df com ghi, dni, dhi
                df=df.assign(csm_ghi=cs['ghi'].values)
                df=df.assign(irr_norm=df['csm_ghi'].values)
                df['irr_norm'] = np.where(df['irr_norm']>0, df[df.columns[2]]/df[df.columns[7]], df['irr_norm'])  
                nwpgrid['D+3'][str(a)+'x'+str(b)]=df 
        
''' DUVIDA: valores de irradiancia normalizada anormalmente altos atingindo valores
de 5 - normalmente ocorrem para a primeira hora do dia em que o valor deixa de ser 0'''
    
'''teste do csm num ponto aleatorio da grid -> OK!
local = Location(location[6,6,0], location[6,6,1])
time = dados['D+2'].index
cs = local.get_clearsky(time)
'''       
        
        
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
df['pot_norm']=np.where(df['pot_norm']>0, df[df.columns[11]]/df[df.columns[14]], df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D']['central']=df 
    
#D+1
df=nwpgrid['D+1']['central']
df=df.assign(pot_norm=df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, df[df.columns[11]]/df[df.columns[14]], df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D+1']['central']=df 

#D+2
df=nwpgrid['D+2']['central']
df=df.assign(pot_norm =df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, df[df.columns[11]]/df[df.columns[14]], df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D+2']['central']=df 

#D+3
df=nwpgrid['D+3']['central']
df=df.assign(pot_norm =df['Pcs'].values)
df['pot_norm']=np.where(df['pot_norm']>0, df[df.columns[11]]/df[df.columns[14]], df['pot_norm']) #11=pot e 14=Pcs   
nwpgrid['D+3']['central']=df 


'''DUVIDA: valores de potencia normalizada apresenta valores anormalmente altos 
chegando a valores de 11 e 12 - normalmente ocorrem para a primeira hora do dia 
em que o valor deixa de ser 0 '''





#Apenas dia D _________________________________________________________________
'''DUVIDA: calculo da variancia temporal e dos lags - aplicar a irradiancia observada ou 
a normalizada (que ja foi calculada) ou as duas? '''



#variancia temporal -> apliquei a todas as variaveis

#central
df=nwpgrid['D']['central']
    
#temperatura
df=df.assign(temp_var3=df[df.columns[0]].rolling(window=3, center=True).var())
df=df.assign(temp_var7=df[df.columns[0]].rolling(window=7, center=True).var())
df=df.assign(temp_var11=df[df.columns[0]].rolling(window=11, center=True).var())

#irradiancia
df=df.assign(irr_var3=df[df.columns[1]].rolling(window=3, center=True).var())
df=df.assign(irr_var7=df[df.columns[1]].rolling(window=7, center=True).var())
df=df.assign(irr_var11=df[df.columns[1]].rolling(window=11, center=True).var())

#pressao
df=df.assign(pres_var3=df[df.columns[2]].rolling(window=3, center=True).var())
df=df.assign(pres_var7=df[df.columns[2]].rolling(window=7, center=True).var())
df=df.assign(pres_var11=df[df.columns[2]].rolling(window=11, center=True).var())

#velociade vento
df=df.assign(vel_var3=df[df.columns[3]].rolling(window=3, center=True).var())
df=df.assign(vel_var7=df[df.columns[3]].rolling(window=7, center=True).var())
df=df.assign(vel_var11=df[df.columns[3]].rolling(window=11, center=True).var())

#direcao vento
df=df.assign(dir_var3=df[df.columns[4]].rolling(window=3, center=True).var())
df=df.assign(dir_var7=df[df.columns[4]].rolling(window=7, center=True).var())
df=df.assign(dir_var11=df[df.columns[4]].rolling(window=11, center=True).var())

#humidade relativa
df=df.assign(humr_var3=df[df.columns[5]].rolling(window=3, center=True).var())
df=df.assign(humr_var7=df[df.columns[5]].rolling(window=7, center=True).var())
df=df.assign(humr_var11=df[df.columns[5]].rolling(window=11, center=True).var())

#frac total
df=df.assign(ftotal_var3=df[df.columns[6]].rolling(window=3, center=True).var())
df=df.assign(ftotal_var7=df[df.columns[6]].rolling(window=7, center=True).var())
df=df.assign(ftotal_var11=df[df.columns[6]].rolling(window=11, center=True).var())

#frac baixa
df=df.assign(fbaixa_var3=df[df.columns[7]].rolling(window=3, center=True).var())
df=df.assign(fbaixa_var7=df[df.columns[7]].rolling(window=7, center=True).var())
df=df.assign(fbaixa_var11=df[df.columns[7]].rolling(window=11, center=True).var())

#frac media
df=df.assign(fmedia_var3=df[df.columns[8]].rolling(window=3, center=True).var())
df=df.assign(fmedia_var7=df[df.columns[8]].rolling(window=7, center=True).var())
df=df.assign(fmedia_var11=df[df.columns[8]].rolling(window=11, center=True).var())

#frac alta
df=df.assign(falta_var3=df[df.columns[9]].rolling(window=3, center=True).var())
df=df.assign(falta_var7=df[df.columns[9]].rolling(window=7, center=True).var())
df=df.assign(falta_var11=df[df.columns[9]].rolling(window=11, center=True).var())

#precipitacao
df=df.assign(prec_var3=df[df.columns[10]].rolling(window=3, center=True).var())
df=df.assign(prec_var7=df[df.columns[10]].rolling(window=7, center=True).var())
df=df.assign(prec_var11=df[df.columns[10]].rolling(window=11, center=True).var())

#irradiancia normalizada
df=df.assign(irrn_var3=df[df.columns[13]].rolling(window=3, center=True).var())
df=df.assign(irrn_var7=df[df.columns[13]].rolling(window=7, center=True).var())
df=df.assign(irrn_var11=df[df.columns[13]].rolling(window=11, center=True).var())

nwpgrid['D']['central']=df




#nwp grid
for a in range(1,7):
  for b in range(1,7):
    
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
    
    #irradiancia normalizada
    df=df.assign(irrn_var3=df[df.columns[8]].rolling(window=3, center=True).var())
    df=df.assign(irrn_var7=df[df.columns[8]].rolling(window=7, center=True).var())
    df=df.assign(irrn_var11=df[df.columns[8]].rolling(window=11, center=True).var())
    
    nwpgrid['D'][str(a)+'x'+str(b)]=df



#lags 
    
#central
df=nwpgrid['D']['central']
    
#temperatura
df=df.assign(temp_lag1=df[df.columns[0]].shift(periods=1))
df=df.assign(temp_lag2=df[df.columns[0]].shift(periods=2))
df=df.assign(temp_lag3=df[df.columns[0]].shift(periods=3))

#irradiancia
df=df.assign(irr_lag1=df[df.columns[1]].shift(periods=1))
df=df.assign(irr_lag2=df[df.columns[1]].shift(periods=2))
df=df.assign(irr_lag3=df[df.columns[1]].shift(periods=3))

#pressao
df=df.assign(pres_lag1=df[df.columns[2]].shift(periods=1))
df=df.assign(pres_lag2=df[df.columns[2]].shift(periods=2))
df=df.assign(pres_lag3=df[df.columns[2]].shift(periods=3))

#velociade vento
df=df.assign(vel_lag1=df[df.columns[3]].shift(periods=1))
df=df.assign(vel_lag2=df[df.columns[3]].shift(periods=2))
df=df.assign(vel_lag3=df[df.columns[3]].shift(periods=3))

#direcao vento
df=df.assign(dir_lag1=df[df.columns[4]].shift(periods=1))
df=df.assign(dir_lag2=df[df.columns[4]].shift(periods=2))
df=df.assign(dir_lag3=df[df.columns[4]].shift(periods=3))

#humidade relativa
df=df.assign(humr_lag1=df[df.columns[5]].shift(periods=1))
df=df.assign(humr_lag2=df[df.columns[5]].shift(periods=2))
df=df.assign(humr_lag3=df[df.columns[5]].shift(periods=3))

#frac total
df=df.assign(ftotal_lag1=df[df.columns[6]].shift(periods=1))
df=df.assign(ftotal_lag2=df[df.columns[6]].shift(periods=2))
df=df.assign(ftotal_lag3=df[df.columns[6]].shift(periods=3))

#frac baixa
df=df.assign(fbaixa_lag1=df[df.columns[7]].shift(periods=1))
df=df.assign(fbaixa_lag2=df[df.columns[7]].shift(periods=2))
df=df.assign(fbaixa_lag3=df[df.columns[7]].shift(periods=3))

#frac media
df=df.assign(fmedia_lag1=df[df.columns[8]].shift(periods=1))
df=df.assign(fmedia_lag2=df[df.columns[8]].shift(periods=2))
df=df.assign(fmedia_lag3=df[df.columns[8]].shift(periods=3))

#frac alta
df=df.assign(falta_lag1=df[df.columns[9]].shift(periods=1))
df=df.assign(falta_lag2=df[df.columns[9]].shift(periods=2))
df=df.assign(falta_lag3=df[df.columns[9]].shift(periods=3))

#precipitacao
df=df.assign(prec_lag1=df[df.columns[10]].shift(periods=1))
df=df.assign(prec_lag2=df[df.columns[10]].shift(periods=2))
df=df.assign(prec_lag3=df[df.columns[10]].shift(periods=3))

#irradiancia normalizada
df=df.assign(irrn_lag1=df[df.columns[13]].shift(periods=1))
df=df.assign(irrn_lag2=df[df.columns[13]].shift(periods=2))
df=df.assign(irrn_lag3=df[df.columns[13]].shift(periods=3))


nwpgrid['D']['central']=df

#nwp grid
for a in range(1,7):
  for b in range(1,7):
    
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
    
    #irradiancia normalizada
    df=df.assign(irrn_lag1=df[df.columns[8]].shift(periods=1))
    df=df.assign(irrn_lag2=df[df.columns[8]].shift(periods=2))
    df=df.assign(irrn_lag3=df[df.columns[8]].shift(periods=3))
    
    nwpgrid['D'][str(a)+'x'+str(b)]=df


############################# EM DESENVOLVIMENTO ##############################

#Média 4 previsoes ____________________________________________________________
'''falta saber como aplicar o ewm().mean sem erros. a estrutura ja esta montada'''


#para ter o index sincronizado
df=pd.concat([nwpgrid['D']['1x1'], nwpgrid['D+1']['1x1'], nwpgrid['D+2']['1x1'], nwpgrid['D+3']['1x1']], axis=1, sort=False)

#criar df media para central e cada pto da grid
cols=['temperatura', 'irr_norm', 'pressao', 'velocidade', 'direcao', 'hum. relativa',
      'frac. total', 'frac. baixa', 'frac. media', 'frac. alta', 'precipitacao', 'pot_norm']

media = {'central': pd.DataFrame(index=[df.index], columns=cols)
        }
media['central'].set_index(df.index, inplace = True)

cols=['irr_norm', 'frac. total', 'frac. baixa', 'frac. media', 'frac. alta']

for a in range(1,7):
    for b in range(1,7):
        media[str(a)+'x'+str(b)] = pd.DataFrame(index=[df.index], columns=cols)
        media[str(a)+'x'+str(b)].set_index(df.index, inplace = True)
        
'''aux=pd.DataFrame({'A': [0,1,2,3],
                  'B': [2,3,4,5]})
aux=aux.assign(media=aux.ewm(span=4, axis=1).mean())  exemplo simplificado p/ duvida '''

#media dos pts da grid
        
#irradiancia
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([nwpgrid['D'][str(a)+'x'+str(b)], nwpgrid['D+1'][str(a)+'x'+str(b)], nwpgrid['D+2'][str(a)+'x'+str(b)], nwpgrid['D+3'][str(a)+'x'+str(b)]], axis=1, sort=False)
    aux=df.filter(like='irr_norm', axis=1)
    aux=aux.assign(media=aux.mean(axis=1))
    media[str(a)+'x'+str(b)]['irr_norm'] = aux['media']
   
#frac. total
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([nwpgrid['D'][str(a)+'x'+str(b)], nwpgrid['D+1'][str(a)+'x'+str(b)], nwpgrid['D+2'][str(a)+'x'+str(b)], nwpgrid['D+3'][str(a)+'x'+str(b)]], axis=1, sort=False)
    aux=df.filter(like='frac. total', axis=1)
    media[str(a)+'x'+str(b)]['frac. total'] = aux.ewm(span=4).mean(axis=1)     
    
#frac. baixa
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([nwpgrid['D'][str(a)+'x'+str(b)], nwpgrid['D+1'][str(a)+'x'+str(b)], nwpgrid['D+2'][str(a)+'x'+str(b)], nwpgrid['D+3'][str(a)+'x'+str(b)]], axis=1, sort=False)
    aux=df.filter(like='frac. baixa', axis=1)
    media[str(a)+'x'+str(b)]['frac. baixa'] = aux.ewm(span=4).mean(axis=1)   
    
#frac. media
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([nwpgrid['D'][str(a)+'x'+str(b)], nwpgrid['D+1'][str(a)+'x'+str(b)], nwpgrid['D+2'][str(a)+'x'+str(b)], nwpgrid['D+3'][str(a)+'x'+str(b)]], axis=1, sort=False)
    aux=df.filter(like='frac. media', axis=1)
    media[str(a)+'x'+str(b)]['frac. media'] = aux.ewm(span=4).mean(axis=1)

#frac. alta
for a in range(1,7):
  for b in range(1,7):
    df=pd.concat([nwpgrid['D'][str(a)+'x'+str(b)], nwpgrid['D+1'][str(a)+'x'+str(b)], nwpgrid['D+2'][str(a)+'x'+str(b)], nwpgrid['D+3'][str(a)+'x'+str(b)]], axis=1, sort=False)
    aux=df.filter(like='frac. alta', axis=1)
    media[str(a)+'x'+str(b)]['frac. alta'] = aux.ewm(span=4).mean(axis=1)
    
'''DUVIDA: aplicar o comando da media pesada sem erros. como aplicar o comando 
.ewm(span=4).mean() sem erro. '''
    
    
    
#media central
df=pd.concat([nwpgrid['D']['central'], nwpgrid['D+1']['central'], nwpgrid['D+2']['central'], nwpgrid['D+3']['central']], axis=1, sort=True)

#temperatura
aux=df.filter(like='Temperatura', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['temperatura'] = aux['media']

#pressao
aux=df.filter(like='Presión', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['pressao'] = aux['media']

#velocidade
aux=df.filter(like='Velocidad', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['velocidade'] = aux['media']
   
#velocidade
aux=df.filter(like='Dirección', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['direcao'] = aux['media']

#humidade relativa
aux=df.filter(like='Humedad rel', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['hum. relativa'] = aux['media']

#frac. total
aux=df.filter(like='Fraccion total.36', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['frac. total'] = aux['media']    
    
#frac. baixa
aux=df.filter(like='Frac. Baja.36', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['frac. baixa'] = aux['media']     
    
#frac. media
aux=df.filter(like='Frac. Media.36', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['frac. media'] = aux['media']    

#frac. alta
aux=df.filter(like='Frac. Alta.36', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['frac. alta'] = aux['media']    

#precipitacao
aux=df.filter(like='Precipitación', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['precipitacao'] = aux['media']

#irradiancia
aux=df.filter(like='irr_norm', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['irr_norm'] = aux['media']

#producao
aux=df.filter(like='pot_norm', axis=1)
aux=aux.assign(media=aux.mean(axis=1))
media['central']['pot_norm'] = aux['media']


#PCA __________________________________________________________________________

'''indicacao paper: melhores resultados obtidos para PCA aplicado individualmente
a cada serie de dados de cada ponto da grid

DUVIDA: segundo o paper so se aplica aos pontos da grid - incluir os da central?

DUVIDA: o pca usa-se sobre os dados observados ou sobre as medias calculadas acima?

DUVIDA: ao aplicar o pca no dia D, usar apenas as variaveis observadas como nos outros
dias ou usar tambem as variancias temporais e lags? se for para os incluir tenho de retirar
nan provenientes da sincronizacao - coloco esses valores a zero?

DUVIDA: eu nao inclui a producao no pca por ser o target e por ser apenas uma coluna. 
esta correto?'''

prin_comp95 = {
    'D': {},
    'D+1': {},
    'D+2': {},
    'D+3': {}
}

prin_comp90 = {
    'D': {},
    'D+1': {},
    'D+2': {},
    'D+3': {}
}


#D central
cols = [0,2,3,4,5,6,7,8,9,10,13]
df=nwpgrid['D']['central'].iloc[:, cols]

pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.90) #numero componentes que prefazem 90%
pca.fit(df)
prin_comp90['D']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)
  

#D grid
cols = [3,4,5,6,8] 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D'][str(a)+'x'+str(b)].iloc[:, cols]
      
      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.90) #numero componentes que prefazem 90%
      pca.fit(df)
      prin_comp90['D'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

#D producao
df=nwpgrid['D']['central']['pot_norm']
prin_comp90['D']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D']['producao']=pd.DataFrame(data=df, index=df.index)




#D+1 central
cols = [0,2,3,4,5,6,7,8,9,10,13] 
df=nwpgrid['D+1']['central'].iloc[:, cols]
df.set_index(nwpgrid['D+1']['central'].index, inplace = True)

pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D+1']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.90) #numero componentes que prefazem 90%
pca.fit(df)
prin_comp90['D+1']['central']=pd.DataFrame(data=pca.transform(df), index=df.index) 


#D+1 grid                 
cols = [3,4,5,6,8] #colunas das variaveis 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D+1'][str(a)+'x'+str(b)].iloc[:, cols]
      df.set_index(nwpgrid['D+1'][str(a)+'x'+str(b)].index, inplace = True)

      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D+1'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.90) #numero componentes que prefazem 90%
      pca.fit(df)
      prin_comp90['D+1'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

#D+1 producao
df=nwpgrid['D+1']['central']['pot_norm']
prin_comp90['D+1']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D+1']['producao']=pd.DataFrame(data=df, index=df.index)




#D+2 central
cols = [0,2,3,4,5,6,7,8,9,10,13]
df=nwpgrid['D+2']['central'].iloc[:, cols]
df.set_index(nwpgrid['D+2']['central'].index, inplace = True)

pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D+2']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.90) #numero componentes que prefazem 90%
pca.fit(df)
prin_comp90['D+2']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)  
      
      
#D+2 grid
cols = [3,4,5,6,8] #colunas das variaveis 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D+2'][str(a)+'x'+str(b)].iloc[:, cols]
      df.set_index(nwpgrid['D+2'][str(a)+'x'+str(b)].index, inplace = True)

      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D+2'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.90) #numero componentes que prefazem 90%
      pca.fit(df)
      prin_comp90['D+2'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)

#D+2 producao
df=nwpgrid['D+2']['central']['pot_norm']
prin_comp90['D+2']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D+2']['producao']=pd.DataFrame(data=df, index=df.index)



#D+3 central
cols = [0,2,3,4,5,6,7,8,9,10,13]
df=nwpgrid['D+3']['central'].iloc[:, cols]
df.set_index(nwpgrid['D+3']['central'].index, inplace = True)

pca = PCA(.95) #numero componentes que prefazem 95%
pca.fit(df)
prin_comp95['D+3']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)

pca = PCA(.90) #numero componentes que prefazem 90%
pca.fit(df)
prin_comp90['D+3']['central']=pd.DataFrame(data=pca.transform(df), index=df.index)      
      
#D+3 grid
cols = [3,4,5,6,8] #colunas das variaveis 
for a in range(1,7):
    for b in range(1,7):
      df=nwpgrid['D+3'][str(a)+'x'+str(b)].iloc[:, cols]
      df.set_index(nwpgrid['D+3'][str(a)+'x'+str(b)].index, inplace = True)


      pca = PCA(.95) #numero componentes que prefazem 95%
      pca.fit(df)
      prin_comp95['D+3'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)
      
      pca = PCA(.90) #numero componentes que prefazem 90%
      pca.fit(df)
      prin_comp90['D+3'][str(a)+'x'+str(b)]=pd.DataFrame(data=pca.transform(df), index=df.index)


#D+3 producao
df=nwpgrid['D+3']['central']['pot_norm']
prin_comp90['D+3']['producao']=pd.DataFrame(data=df, index=df.index)

prin_comp95['D+3']['producao']=pd.DataFrame(data=df, index=df.index)




#print(pca.explained_variance_ratio_)







##### Weather to Power Model ##################################################
      
      
#treino - 1/1/2020-31/5/2020; teste - 1/1/2017-31/12/2019 _____________________
#principal components em todos os pontos - X
#potencia produzida D+1 - Y

'''DUVIDA: abaixo passo os dados provenientes do pca para uma df a parte para servir
de X_train. a sincronizacao dos dados cria nan - coloco-os a zero? '''

#TREINO
df=pd.DataFrame()
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D+1'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D+2'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)
      
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D+3'][str(a)+'x'+str(b)].loc['2017-01-01':'2019/12/31']], axis=1, sort=True)      
      
      
x_train=df #ainda tem nans - preciso tirar - por zero?

y_train = prin_comp90['D+1']['producao'].loc['2017-01-01':'2019/12/31']

'''DUVIDA: y_train e apenas a potencia no dia D+1 uma vez que a nossa intencao e prever 
apenas o dia D+1, certo? '''

#TESTE

df=pd.DataFrame()
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D+1'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D+2'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)
      
for a in range(1,7):
    for b in range(1,7):
      df=pd.concat([df, prin_comp90['D+3'][str(a)+'x'+str(b)].loc['2020-01-01':'2020/05/31']], axis=1, sort=True)      
      
      
x_test=df #ainda tem nans - preciso tirar - por zero?

y_test = prin_comp90['D+1']['producao'].loc['2020-01-01':'2020/05/31']

      
      
#GBT___________________________________________________________________________

      
      
      
      
      
      


#XGBoost_______________________________________________________________________
      
      
    
    
    
    
    
    
    
#CatBoost______________________________________________________________________
      

    
    
    
    
    
    
    
    
    
    
    
    
    
