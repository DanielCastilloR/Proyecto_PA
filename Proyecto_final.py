# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:13:23 2022

@author: Daniel
"""


import astsadata as astsa
from astsadata import chicken, globtemp,birth
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pmdarima as arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import scipy.stats as cs
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
from pandas_datareader.data import DataReader


class Proyecto:
    
    def __init__(self):
        data = pd.read_csv("Serie_desocupacion.csv")
        data.index = data['Periodos']
        data = data.drop(['Periodos'],axis=1)
        self.data_total = data['Total']
        self.crecimiento = self.data_total.diff().dropna()
          
    def desc(self):
        print(self.data_total.describe())
        
    def datos(self):
        plt.figure(figsize=(20,10))
        self.data_total.plot(color='darkred')
        plt.title("Tasa de desocupación")
        plt.xlabel("Tiempo")
        plt.ylabel("Tasa")
        plt.show()
        
    def datos_dif(self):
        plt.figure(figsize=(20,10))
        self.crecimiento.plot(color='darkblue')
        plt.title("Tasa de desocupación")
        plt.xlabel("Tiempo")
        plt.ylabel("Tasa")
        plt.show()
        
    def Dickey_Fuller(self,serie):
        if serie == 'original':
            fuller = adfuller(self.data_total,autolag='AIC')
            print('estadistico ADF: %f' % fuller[0])
            print('p-valor: %f' % fuller[1])
            print('Critical Values:')
            for key, value in fuller[4].items():
            	print('\t%s: %.3f' % (key, value))
                
        if serie == 'crecimiento':
            fuller = adfuller(self.crecimiento,autolag='AIC')
            print('estadistico ADF: %f' % fuller[0])
            print('p-valor: %f' % fuller[1])
            print('Critical Values:')
            for key, value in fuller[4].items():
            	print('\t%s: %.3f' % (key, value))
        
    def acf(self,serie):
        if serie == 'original':
            x = acf(self.data_total,nlags=100)
            plt.stem(x,markerfmt=' ',linefmt ='darkred')                                     
            plt.xlabel("Lag")
            plt.ylabel("ACF")
            plt.hlines((1.96/(np.sqrt(len(self.data_total)))), 0, 100,linestyles='dashed',color='darkblue')
            plt.hlines((-1.96/(np.sqrt(len(self.data_total)))), 0, 100,linestyles='dashed',color='darkblue')
            plt.title("Función de auto-correlación")
            plt.show()
        
        if serie == 'crecimiento':
            x = acf(self.crecimiento,nlags=100)
            plt.title("Crecieminto")
            plt.stem(x,markerfmt=' ',linefmt ='darkred')
            plt.xlabel("Lag")
            plt.ylabel("ACF")
            plt.hlines((1.96/(np.sqrt(len(self.crecimiento)))), 0, 100,linestyles='dashed',color='darkblue')
            plt.hlines((-1.96/(np.sqrt(len(self.crecimiento)))), 0, 100,linestyles='dashed',color='darkblue')
            plt.title("Función de auto-correlación")
            plt.show()
            
    def pacf(self,serie):
        
        if serie == 'original':
            x = pacf(self.data_total,nlags=80)
            plt.stem(x[1:],markerfmt=' ',linefmt ='darkred')                                     
            plt.xlabel("Lag")
            plt.ylabel("PACF")
            plt.hlines((1.96/(np.sqrt(len(self.data_total)))), 0, 80,linestyles='dashed',color='darkblue')
            plt.hlines((-1.96/(np.sqrt(len(self.data_total)))), 0, 80,linestyles='dashed',color='darkblue')
            plt.title("Función de auto-correlación parcial")
            plt.show()
        
        if serie == 'crecimiento':
            x = pacf(self.crecimiento,nlags=80)
            plt.title("Crecieminto")
            plt.stem(x[1:],markerfmt=' ',linefmt ='darkred')
            plt.xlabel("Lag")
            plt.ylabel("PACF")
            plt.hlines((1.96/(np.sqrt(len(self.crecimiento)))), 0, 80,linestyles='dashed',color='darkblue')
            plt.hlines((-1.96/(np.sqrt(len(self.crecimiento)))), 0, 80,linestyles='dashed',color='darkblue')
            plt.title("Función de auto-correlación parcial")      
            
    def modelar(self):
        self.modelo = SARIMAX(self.data_total, order=(1,1,2),seasonal_order=(1, 0, 2, 6))
        self.modelo = self.modelo.fit()
        print(self.modelo.summary())     
        
    def raices_plot(self):
        raices = 1/self.modelo.arroots
        real = np.array( [ele.real for ele in raices])
        imag = np.array([ele.imag for ele in raices])
        
        raices2 = 1/self.modelo.maroots
        real2 = np.array( [ele.real for ele in raices2])
        imag2 = np.array([ele.imag for ele in raices2])
        
        theta = np.linspace(0, 2*np.pi, 100)
        r = np.sqrt(1.0)
        x1 = r*np.cos(theta)
        x2 = r*np.sin(theta)
        fig, ax = plt.subplots(1,figsize=(10,10))
        ax.plot(x1, x2,color='black')
        plt.xlim(-1.25,1.25)
        plt.ylim(-1.25,1.25)
        plt.scatter(real,imag,color='darkred',s=150,label='AR')
        plt.scatter(real2,imag2,color='darkblue',s=150,label='MA')
        plt.vlines(0,-2,2,color='black',linewidth=0.6)
        plt.hlines(0,-2,2,color='black',linewidth=0.6)
        plt.xlabel("Reales")
        plt.ylabel("Imaginarios")
        plt.title("Inversa de las Raíces",fontsize=15,color='black')
        plt.legend()
        plt.show()       
    
    def residuales_analisis(self):
        residuales = (self.modelo.resid - self.modelo.resid.mean())/self.modelo.resid.std()
        residuales = residuales[1:]
        residuales.plot()
        plt.title("Residuales estandarizados")
        plt.show()
        
        x = acf(self.modelo.resid,nlags=50)
        plt.stem(x[1:],markerfmt=' ',linefmt ='darkred')
        plt.title("Función de auto-correlación")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.hlines((1.96/(np.sqrt(len(self.modelo.resid)))), 0, 50,linestyles='dashed',color='darkblue')
        plt.hlines((-1.96/(np.sqrt(len(self.modelo.resid)))), 0, 50,linestyles='dashed',color='darkblue')
        plt.show()
        
        sns.histplot(residuales,stat='density', kde=True,color='darkred')
        plt.title("Dist. residuales estandarizados")
        plt.show()
        
        qqplot(residuales, line='s')
        plt.title("QQ-plot de resiudales estandarizados")
        plt.show()
        
        p_v = acorr_ljungbox(self.modelo.resid[1:],lags=np.arange(1,20), return_df=True,
                             model_df=3) # corrigiendo los p + q grados de libertad
        plt.subplots(1,figsize=(10,4))
        plt.scatter(p_v.index,p_v.lb_pvalue,color='darkred')
        plt.ylabel("P_valor")
        plt.xlabel("Lags")
        plt.title("Ljung-Box")
        plt.hlines(0.05,4,20, linestyles ="dashed",color='black')
        plt.show()
        print(cs.shapiro(residuales))
        
    def fit(self):
        pred = self.modelo.predict()
        
        plt.figure(figsize=(20,10))
        self.data_total.plot(color='darkblue',label='Datos')
        plt.plot(pred.iloc[1:],color='darkred',label='Ajuste')
        plt.xlabel("Tiempo")
        plt.legend()
        plt.show()
  
    def descomp(self):
        self.result = seasonal_decompose(self.data_total, model='multiplicable', period=12)
        self.result.plot()
        plt.show()

    def des_estacion(self):
        des_Estacion = self.data_total - self.result.seasonal
        plt.figure(figsize=(25,8))
        plt.bar(x=des_Estacion.index[12:],height=des_Estacion[12:])
        plt.xticks(rotation=90)
        plt.show()

    def tendencia_ciclo(self):
        
        hp_cycle, hp_trend = sm.tsa.filters.hpfilter(self.data_total, lamb=129600)

        plt.figure(figsize=(20,10))
        self.data_total.plot(color='darkred',label='Serie')
        plt.plot(hp_cycle,color='orange',label='Ciclo')
        plt.plot(hp_trend,color='green',label='Tendencia')
        plt.legend()
        plt.show()
        
    def forecast(self):
        forcast = self.modelo.forecast(4)
        conf = self.modelo.get_forecast(4).conf_int(alpha=0.05)
        
        date = np.array(("01/02/2022","01/03/2022","01/04/2022","01/05/2022"))
        forcast.index = date
        conf.index = date
        conf = pd.concat((conf,forcast),axis=1)
        
        tam = len(self.data_total)
        na = np.empty(tam)
        na[:] = np.nan
        na[-1] = self.data_total[-1]
        na = np.hstack((na,forcast))
        date2 = np.hstack((self.data_total.index , date))
        fore = pd.DataFrame(na)
        fore.index = date2
        data_total2 = np.hstack((self.data_total.values,np.nan,np.nan,np.nan,np.nan))
        data_total2 = pd.DataFrame(data_total2)
        data_total2.index = date2
        pronostico = pd.concat((data_total2,fore),axis=1)
        pronostico.columns = ("Serie Real","Pronóstico")
        
        mat = np.empty((tam,3))
        mat[:,:] = np.nan
        mat = pd.DataFrame(mat)
        
        IC = pd.concat((mat,conf))
        IC = IC.drop([0,1,2],axis=1)
        IC.index = date2
        
        fig, axes = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(25,15))
        pronostico.plot(color=('darkred','darkblue'),ax=axes)
        axes.fill_between(IC["predicted_mean"].index, IC['lower Total'], IC['upper Total'],facecolor='green', alpha=0.4, label='Prediction Interval')
        plt.show()
        
        
        
Inicio = Proyecto()  
Inicio.desc()     
Inicio.datos() 
Inicio.Dickey_Fuller("original") 
Inicio.acf("original")
Inicio.pacf("original")
Inicio.datos_dif()
Inicio.Dickey_Fuller("crecimiento") 
Inicio.modelar()
Inicio.raices_plot()
Inicio.fit()
Inicio.residuales_analisis()
Inicio.acf("crecimiento")
Inicio.pacf("crecimiento")
Inicio.descomp()
Inicio.des_estacion()
Inicio.tendencia_ciclo()
Inicio.forecast()



