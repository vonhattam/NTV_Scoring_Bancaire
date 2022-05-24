
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
import shap


app = Flask(__name__)

shap_values = pickle.load( open( "shap_values.p", "rb" ) )
pred_frame = pd.read_csv("pred_frame_500.csv")#Chargement dataset
target_frame = pd.read_csv("target_frame_500.csv")
data_dash = pd.read_csv('data_dash.csv')
lgbm= pickle.load( open( "model_lgbm.pkl", "rb" ) )
set_shap = pickle.load( open( "set_shap.p", "rb" ) )
scaler_std = pickle.load( open( "scaler_std.pkl", "rb" ) )
data_api = pd.read_csv("data_api.csv")
data_api.reset_index()

@app.route('/', methods=["GET"])
def home():
    return 'Bienvenue sur l\'API du projet Implémentez un modèle de scoring \n\nVoici les endpoints de cette API: \nprediction: requête pour charger les données avec le score prédit \nprofile_client: requête pour le chargement du profile de client sélectionné \ncredit: requête: pour le chargement du crédit de client sélectionné \nfeature_importance: requête permettant d\'illustrer les importances des différents features du modèle \nshap_values: requête permet de charger les données utilisés pour le calcul des shap values \ndata_comparaison: requête pour charger les données des clients'

@app.route('/prediction/<int:ID>/',  methods=['GET'])
def prediction(ID):
    
    if ID not in list(data_api['SK_ID_CURR']):
        result = 'Ce client n\'est pas dans la base de donnée'
    else:
        data_api_ID=data_api[data_api['SK_ID_CURR']==int(ID)]

        y_proba=lgbm.predict_proba(scaler_std.transform(data_api_ID.drop(['SK_ID_CURR','TARGET'],axis=1)))[:, 1]
        
        seuil=0.5 
       
        if y_proba >= seuil:
            y_Target=1
        elif y_proba < seuil:
             y_Target=0
   
   
        if y_Target == 0:
           result=('Ce client est solvable avec un taux de risque de {} %'.format(np.around(y_proba*100,2).item()))

        elif y_Target == 1:
            result=('Ce client est non solvable avec un taux de risque de {} %'.format(np.around(y_proba*100,2).item()))
    results = {'message': result, 
               'solvable': False if y_proba > seuil else True, 
               'pourcentage': np.around(y_proba*100,2).item()}
    return jsonify(results)

@app.route('/feature_importance/<int:ID>/',  methods=['GET'])
def feature_importance(model):
    features_importance = model.feature_importances_    
    features = data_api.drop(['SK_ID_CURR','TARGET'],axis=1).columns
    feature_importance_df = pd.DataFrame({'Feature':features,'Importance':feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by=['Importance'], ascending=False)
    feature_importance_df=feature_importance_df.to_json()
    return jsonify(feature_importance_df)
    
@app.route('/profile_client/<int:ID>/',  methods=['GET'])
def profile_client(ID):    
    
    data_dash['AGE']=round(abs(data_dash['DAYS_BIRTH']/365)).astype(int)
    ID_c=int(ID)  
    info_client_t=data_dash[data_dash['SK_ID_CURR']==ID_c]

    frame_info_client=info_client_t[['AGE','CODE_GENDER','NAME_FAMILY_STATUS','CNT_CHILDREN', 'NAME_HOUSING_TYPE','NAME_EDUCATION_TYPE','AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'DAYS_EMPLOYED']]
    frame_info_client.index = [ID_c]
    frame_info_client['DAYS_EMPLOYED']=frame_info_client.apply(lambda x: round((x["DAYS_EMPLOYED"]/-365),1),axis=1)
    frame_info_client.columns = ['Age', 'Genre', 'Situation', 'Enfant', 'Location',  'Niveau d\'éducation', 'Revenu ($)', 'Type de revenus', 'Durée d\'emploi (ans)']
    frame_info_client=frame_info_client.T
    frame_info_client = frame_info_client.astype(str)
    frame_info_client = frame_info_client.to_json()
  
    return jsonify(frame_info_client) 

@app.route('/credit/<int:ID>/',  methods=['GET'])
def credit(ID):
    ID_c=int(ID)        
    table=data_dash[data_dash['SK_ID_CURR']==ID_c]

        
    table=table[['AMT_CREDIT', 'CNT_PAYMENT','AMT_PAYMENT','AMT_ANNUITY']]
    
    table = table.rename(columns={'AMT_PAYMENT':'Remboursements crédit($)',
                           'AMT_CREDIT':"Montant crédit ($)",
                           'CNT_PAYMENT':"Durée du crédit (ans)",
                           'AMT_ANNUITY':'Annuités emprunt ($)'})
    table.index = ['ID: {}'.format(ID_c)]
    table = table.T
    table = table.astype(str)
    table = table.to_json()
  
    return jsonify(table)

@app.route('/shap_values/<int:ID>/',  methods=['GET'])
def shap_values(ID):
    ID_c=int(ID)    
    X_ID=pred_frame[pred_frame['SK_ID_CURR']==ID_c].copy()
    X_ID=X_ID.reset_index(drop=True)
    X=X_ID.drop(['SK_ID_CURR','Proba', 'TARGET','PREDICTION'],axis=1)
    X_std = scaler_std.transform(X)
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_std)
    shap_values = np.asarray(shap_values)
    shap_values =shap_values.tolist()
    set_shap=pd.DataFrame(shap_values[1],columns=X.columns)
    X = X.to_json()
    
    set_shap['SK_ID_CURR']=X_ID['SK_ID_CURR']
    set_shap['TARGET']=X_ID['TARGET']
    set_shap['Proba']=X_ID['Proba']
    set_shap['PREDICTION']=X_ID['PREDICTION']
    set_shap = set_shap.to_json()
    results = {'data_client': X,
               'shap_values': shap_values, 
               'set_shap': set_shap}
  
    return jsonify(results)

@app.route('/data_comparaison/<int:ID>',  methods=['GET'])
def data_comparaison(ID):
    ID_c=int(ID)    
    data_compa = target_frame.copy()
    data_compa = data_compa.to_json()
 
    return jsonify(data_compa)
if __name__ == '__main__':
  app.run(debug=True)
