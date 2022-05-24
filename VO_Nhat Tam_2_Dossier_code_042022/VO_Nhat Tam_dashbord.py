

import streamlit as st
import shap
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import time
import requests, json
from urllib.request import urlopen




#url = "http://127.0.0.1:5000/{}/{}"

url = 'http://ntv-api.herokuapp.com/{}/{}'


#fenetre input
st.title('Évaluation des demandes de Crédit')
st.subheader("Prédictions du score bancaire du client")
id_input = st.text_input('Merci de saisir l\'identifiant du client:', )
st.write('Exemple d\'ID client: 454293; 105865; 365820')

@st.cache
def requet_ID(ID):
    
    ID_client=int(ID)
    
    response = requests.get(url.format('prediction', ID_client)).json()
    message = response['message']
    return message

def profil_client(ID):
    
    ID_client=int(ID)
    response = requests.get(url.format('profile_client', ID_client)).json()
    response = pd.read_json(response)    
    return response

def credit (ID):
    ID_client=int(ID)

    response = requests.get(url.format('credit', ID_client)).json() 
    response = pd.read_json(response)   
    return response

def proba(ID):
    
    ID_client=int(ID)
    
    response = requests.get(url.format('prediction', ID_client)).json()
    pourcentage = response['pourcentage']
    return pourcentage

            

def gauge_plot(score):
    
    fig = go.Figure(go.Indicator(domain = {'x': [0, 1], 'y': [0, 1]},
                                 value = score,
                                 mode = "gauge+number",
                                 title = {'text': "Rique de crédit"},
                                 delta = {'reference': 0.3},gauge = {'axis': {'range': [None, 100]},
                                                                     'steps' : [{'range': [0, 50], 'color': "whitesmoke"},
                                                                                {'range': [50, 1], 'color': "lightgray"}],
                                                                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))

    st.plotly_chart(fig)

def plot_shap(ID):
    ID_client=int(ID)
    
    response = requests.get(url.format('shap_values', ID_client)).json()
    shap_values= response['shap_values']
    shap_values = np.asarray(shap_values)
    lst_id = []
    for i in  shap_values:
        lst_id.append(i)
    X = response['data_client']
    X = pd.read_json(X)
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(lst_id, X, max_display=10, plot_type ="bar",  color_bar=True)
    st.pyplot(fig)
    
def plot_feature_important(ID):
    ID_client=int(ID)
    
    feature_importance = requests.get(url.format('feature_importance', ID_client)).json()

    fig, ax= plt.subplots(figsize=(12,8))
    feature_importance.plot.bar()
    ax.set_title("Feature importances du modèle Bagging sur l'émission  CO2",
             fontsize = 24)
    ax.set_ylabel("Features Importances", fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.xticks( fontsize = 14)
def plot_radars(ID):
    
    ID_client=int(ID)
    
    response1= requests.get(url.format('shap_values', ID_client)).json()
    frame_cl= response1['data_client']
    frame_cl= pd.read_json(frame_cl)
    response2= requests.get(url.format('data_comparaison', ID_client)).json()
    frame_all= pd.read_json(response2)
    
    fig2 = go.Figure()

    trace0=fig2.add_trace(go.Scatterpolar(
        r=frame_cl.iloc[:,:8].values.reshape(-1),
        theta=frame_cl.columns[:8],
        fill='toself',
        name="Client sélectionné"
        ))
    
    trace1=fig2.add_trace(go.Scatterpolar(
        r=frame_all[frame_all["TARGET"]==1].iloc[:,3:11].mean().values.reshape(-1),
        theta=frame_all.columns[3:11],
        fill='toself',
        name="Moyennes des clients avec défaut de paiement"
        ))
    
    trace2=fig2.add_trace(go.Scatterpolar(
        r=frame_all[frame_all["TARGET"]==0].iloc[:,3:11].mean().values.reshape(-1),
        theta=frame_all.columns[3:11],
        fill='toself',
         name="Moyennes des clients sans défaut de paiement"
        ))
    data = [trace0, trace1]
    
    fig2.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=False
          #range=[0, 1]
        )),
        legend=dict(
        yanchor="top",
        y=-0.1,
        xanchor="left",
        x=0.01
        ),
        title={'text': "Comparatif du client",
                'y':0.95,
                'x':0.5,
                'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=17)
    st.plotly_chart(fig2)
  
def hist_plot_global(ID):
  ID_client=int(ID)
  ### Récupération des feat. les plus importants pour le client
  response1 = requests.get(url.format('shap_values', ID_client)).json()
  frame_shap= response1['set_shap']
  frame_shap = pd.read_json(frame_shap)
  set_shap_id=frame_shap[frame_shap['SK_ID_CURR']==ID_client].copy().T
  set_shap_id=set_shap_id.rename({frame_shap[frame_shap["SK_ID_CURR"]==ID_client].index.item(): 'valeur'}, axis=1)
  set_shap_id=set_shap_id.drop(['SK_ID_CURR','Proba', 'TARGET','PREDICTION'],axis=0).sort_values(by='valeur')

  ft_ID_1=[]
  ft_ID_0=[]
  ft_ID_1.append(set_shap_id.index[0])
  ft_ID_1.append(set_shap_id.index[1])
  ft_ID_0.append(set_shap_id.index[-1])
  ft_ID_0.append(set_shap_id.index[-2])

  #Feat augmentant le risque
  response2= requests.get(url.format('shap_values', ID_client)).json()
  frame_cl= response2['data_client']
  frame_cl= pd.read_json(frame_cl)
  response3= requests.get(url.format('data_comparaison', ID_client)).json()
  frame_all= pd.read_json(response3)     
  st.write('Comparaison des informations du client pour les principaux indicateurs augmentant le risque par rapport à l\'ensemble des clients de la base de données.')
  for ft in ft_ID_1:
    #plt.style.use('seaborn-deep')  
    st.subheader(ft)
    fig,ax=plt.subplots( figsize=(10,4))

  
    x = frame_all[frame_all['TARGET']==0][ft]
    y = frame_all[frame_all['TARGET']==1][ft]
    z = frame_all[ft]
    #bins = np.linspace(0, 1, 15)

    risque_client=frame_cl[ft].item()

    plt.hist([x, y,z], label=['Solvable', 'Non solvable','Global'])
    plt.axvline(risque_client,linewidth=4, color='#d62728')

    plt.legend(loc='upper right')
    plt.ylabel('Nb de client')
    plt.xlabel('Valeur non-normalisée')
    plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
    plt.figtext(0.797,0.9,'Client '+str(ID_client))
    plt.show()
    st.pyplot(fig)

  #feat diminuant le risque
  
  st.write('Comparaison des informations du client pour les principaux indicateurs indicateurs diminuant le risque par rapport à l\'ensemble des clients de la base de données.')
  for ft in ft_ID_0:
    st.subheader(ft)
    fig,ax=plt.subplots( figsize=(10,4))
    
    #plt.style.use('seaborn-deep')

    x = frame_all[frame_all['TARGET']==0][ft]
    y = frame_all[frame_all['TARGET']==1][ft]
    z = frame_all[ft]
    #bins = np.linspace(0, 1, 15)

    risque_client=frame_cl[ft].item()

    plt.hist([x, y,z], label=['Solvable', 'Non solvable','Global'])
    plt.axvline(risque_client,linewidth=4, color='#d62728')

    plt.legend(loc='upper right')
    plt.ylabel('Nb de client')
    plt.xlabel('Valeur non-normalisée')
    plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
    plt.figtext(0.797,0.9,'Client '+str(ID_client))
    plt.show() 
    st.pyplot(fig)
    
def plot_bivarie_1(ID):
    ID_client=int(ID)
    response1= requests.get(url.format('data_comparaison', ID_client)).json()
    data_score= pd.read_json(response1)
    response2= requests.get(url.format('shap_values', ID_client)).json()
    infos_client= response2['data_client']
    infos_client= pd.read_json(infos_client)    
     
    st.write('"Visualisez l\'analyse bivarié des revenus et montant du crédit des clients"')
 
    fig=plt.figure(figsize=(8,8))
    target0= data_score[data_score['TARGET']==0]
    target1= data_score[data_score['TARGET']==1]
    ax=plt.scatter(x=target0['AMT_INCOME_TOTAL'], y=target0['AMT_CREDIT'], color="blue", label = 'clients sans défaut de paiement')
    ax=plt.scatter(x=target1['AMT_INCOME_TOTAL'], y=target1['AMT_CREDIT'], color="red", label = 'clients avec défaut de paiement')

    ax= plt.scatter(x=infos_client["AMT_INCOME_TOTAL"], y =infos_client["AMT_CREDIT"], s = 100,  color="green", label = 'client slectionné')

    plt.legend()
    plt.title('Montant crédit en fonction des revenus des Clients', size=15)
    plt.xlabel('Revenu en euros', size=15)
    plt.ylabel('Montant crédit en euros', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.ylim([1e4, 3e6])
    plt.xlim([1e5, 8e5])
    st.pyplot(fig)
    
def plot_bivarie_2(ID):
    ID_client=int(ID)
    response1= requests.get(url.format('data_comparaison', ID_client)).json()
    data_score= pd.read_json(response1)
    response2= requests.get(url.format('shap_values', ID_client)).json()
    infos_client= response2['data_client']
    infos_client= pd.read_json(infos_client)    
     
    st.write('Visualisez l\'analyse bivarié des revenus et âge des clients')
 
    fig=plt.figure(figsize=(8,8))
    target0= data_score[data_score['TARGET']==0]
    target1= data_score[data_score['TARGET']==1]
    ax=plt.scatter(x=target0['DAYS_BIRTH']/-365, y=target0["AMT_INCOME_TOTAL"], color="blue", label = 'clients sans défaut de paiement')
    ax=plt.scatter(x=target1['DAYS_BIRTH']/-365, y=target1["AMT_INCOME_TOTAL"], color="red", label = 'clients avec défaut de paiement')

    ax= plt.scatter(x=infos_client['DAYS_BIRTH']/-365, y =infos_client["AMT_INCOME_TOTAL"], s = 100,  color="green", label = 'client slectionné')

    plt.legend()
    plt.title('Revenu en fonction de lâge des clients', size=15)
    plt.xlabel('Age en ans', size=15)
    plt.ylabel('Revenu en euros', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.ylim([1e5, 8e5])

    st.pyplot(fig)

def plot_bivarie_3(ID):
    ID_client=int(ID)
    response1= requests.get(url.format('data_comparaison', ID_client)).json()
    data_score= pd.read_json(response1)
    response2= requests.get(url.format('shap_values', ID_client)).json()
    infos_client= response2['data_client']
    infos_client= pd.read_json(infos_client)    
     
    st.write('Visualisez l\'analyse bivarié des années emploi et âge des clients')
 
    fig=plt.figure(figsize=(8,8))
    target0= data_score[data_score['TARGET']==0]
    target1= data_score[data_score['TARGET']==1]
    ax=plt.scatter(x=target0['DAYS_BIRTH']/-365, y=target0["DAYS_EMPLOYED"]/-365, color="blue", label = 'clients sans défaut de paiement')
    ax=plt.scatter(x=target1['DAYS_BIRTH']/-365, y=target1["DAYS_EMPLOYED"]/-365, color="red", label = 'clients avec défaut de paiement')

    ax= plt.scatter(x=infos_client['DAYS_BIRTH']/-365, y =infos_client["DAYS_EMPLOYED"]/-365, s = 100,  color="green", label = 'client slectionné')

    plt.legend()
    plt.title('Années emploi en fonction de lâge  des clients', size=15)
    plt.xlabel('Age en ans', size=15)
    plt.ylabel('Années emploi en ans', size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.ylim([0,30])

    st.pyplot(fig)




#######

if id_input == '':
    st.write('Veuillez entrer un ID')
    
else:
    r_ID=requet_ID(id_input)
    st.write(r_ID)
    
   
    
    if r_ID != 'Ce client n\'est pas dans la base de données.':
    
        option = st.sidebar.selectbox('Plus d\'informations',['','Interprétation', 'Comparaison des clients', 'Visualisations univariées', 'Visualisations bivariées']) 
        
        if option == 'Visualisations univariées':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input))
            st.write(hist_plot_global(id_input))
            
        elif option == 'Visualisations bivariées':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input))
            st.write(plot_bivarie_1(id_input))
            st.write(plot_bivarie_2(id_input))
            st.write(plot_bivarie_3(id_input)) 

        elif option == 'Interprétation':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input))
            st.write("Informations du crédit de client : ")
            st.write(credit(id_input))
            st.write('Interprétation du modèle - Importance des variables locale :')
            st.write(plot_shap(id_input))
            
        elif option == 'Comparaison des clients':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input))
            st.write(plot_radars(id_input))
            
        else:
            st.sidebar.write('Chosissez une option')
            score = proba(id_input)
            st.write(gauge_plot(score))