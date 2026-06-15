import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from opencage.geocoder import OpenCageGeocode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# import tqdm.notebook as tqdm 
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import os 
import joblib
import json
import pandas as pd 
import numpy as np 
from datetime import date
from fuzzywuzzy import fuzz
from sklearn.metrics import precision_score, \
                            recall_score, f1_score, \
                            roc_auc_score, roc_curve, \
                              auc , confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


def preproccesBeforePipe (X,train = False):
    data = X.copy()
#     data = data.set_index('observation_id')
    if train : 
        # removing nulls stations
        data = data[~data['station'].isin(['lancashire', 'metropolitan'])]
        # dealing with True outcome liked but false outcome
        possible_having_with_premmision_paper = ['Controlled drugs','Fireworks','Game or poaching equipment','Firearms','Goods on which duty has not been paid etc.','Crossbows','Seals or hunting equipment']
        objects = list(data.loc[(data['Outcome']=='A no further action disposal')& (data['Outcome linked to object of search'] ==True),:]['Object of search'].value_counts().index)
        for x in possible_having_with_premmision_paper :
            objects.remove(x)
        data.loc[(data['Object of search'].isin(objects))&(data['Outcome']=='A no further action disposal')& (data['Outcome linked to object of search'] ==True),'Outcome linked to object of search'] = False
        # Removing age range <10 and vechile type 
        values = data.loc[(data['Age range'].isin(['under 10']))&(data['Type'].isin(['Vehicle search','Person and Vehicle search']))& (data['Outcome linked to object of search']==True),:].index
        data.drop(index=list(values),inplace=True)
        # Delecte Self-defiened and Removal 
        data = data.drop(columns=['Self-defined ethnicity','Removal of more than just outer clothing'])
        # making Target
        data['target'] = False 
#         data['target'] = data['target'].mask((data['Outcome']!='A no further action disposal' ),True)
        data['target'] = data['target'].mask((data['Outcome']!='A no further action disposal'),True)
        # drop outcomes columns
        data.drop(columns = ['Outcome', 'Outcome linked to object of search'],inplace=True)
        # changing datatype to be good for the API
    #     categorical_features = list(x for x in df_model.select_dtypes(['object']).columns if x != 'Part of a policing operation')
    #     df_model[categorical_features] = df_model[categorical_features].astype('string')
#         data['Part of a policing operation'] = data['Part of a policing operation'].astype('bool')
    # lowercasing test columns
    lowering = ['station','Type','Age range','Gender','Officer-defined ethnicity','Legislation','Object of search']
    data[lowering] = data[lowering].applymap(lambda x: str(x).lower() if pd.notna(x)  else x)
    return data







 
class Preprocces(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        # Fit the transformer and store it.
        return self
        
    def transform(self, X):
        df_model= X.copy()
#         df_model = df_model.set_index('observation_id')
        df_model['Date'] = pd.to_datetime(df_model['Date'],infer_datetime_format=True)
        holiday_dates = [date(2020,1,1),date(2020,12,25),date(2021,1,1),date(2021,12,25),date(2022,12,25),date(2023,1,1),date(2023,12,25)]
        df_model['is_holiday'] = df_model['Date'].apply(lambda x : 1 if x.date() in holiday_dates else 0)
        df_model['Timeofday'] = pd.cut(df_model['Date'].dt.hour , bins = [0,6,12,18,24], labels =['night','morning','afternoon','evening'])
        df_model['DayofWeek'] = df_model['Date'].dt.day_name()
        df_model['pastmid'] = (df_model['Date'].dt.hour <6 ).astype(int)
        # feature engineering from Date 
        
#         df_model['day_of_week'] = df_model['Date'].dt.day_name()
        df_model['month'] = df_model['Date'].dt.month
        df_model['hour'] = df_model['Date'].dt.hour
        df_model['year'] = df_model['Date'].dt.year
        df_model.drop(columns = ['Date','observation_id'],inplace=True)
        return df_model


    
    
class FullImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        data = X.copy()
        # Fit the transformer and store it.
        self.pipe = joblib.load('pickles/pipeline_legislation2.pickle')
        self.mapping = {}
        for name, group in data.groupby("station")[['Latitude','Longitude']]:
            try :
                
                self.mapping [name] = group.mode().iloc[0].to_dict()  
            except IndexError : 
                continue
        return self
        
    def transform(self, X):
        data = X.copy()
#         # filling outcome linked
#         data.loc[:,'Outcome linked to object of search'].fillna(False,inplace = True)
        # filling part of policing operation
        if data['Part of a policing operation'].isnull().sum() :
            data['Part of a policing operation'].fillna(False,inplace = True)
        # filling legislation 
        if data['Legislation'].isnull().sum() :
            test = data.loc[data['Legislation'].isnull(),['Legislation','Object of search']]
            pred =  self.pipe.predict(test[['Object of search']])
            data.loc[data['Legislation'].isnull(),'Legislation'] = pred
        # filling lat and long 
        if data['Latitude'].isnull().sum() or data['Longitude'].isnull().sum() :  
            known_stations = {'nottinghamshire': {'Latitude': 53.1459288, 'Longitude': -1.0214971},'south-yorkshire': {'Latitude': 53.4815333, 'Longitude': -1.3810422}}
            for name, group in data.groupby("station")[['Latitude','Longitude']]:
                try :
                    idxs = group.index
                    if group['Latitude'].isnull().sum() : 
                        imputer_lat = SimpleImputer(strategy="constant",fill_value = self.mapping[name]['Latitude'])
                        data.loc[idxs,['Latitude']] = imputer_lat.fit_transform(group[['Latitude']])
                    if group['Longitude'].isnull().sum() :
                        imputer_long = SimpleImputer(strategy="constant",fill_value = self.mapping[name]['Longitude'])
                        data.loc[idxs,['Longitude']] = imputer_long.fit_transform(group[['Longitude']])
                except KeyError :
                    if name in known_stations.keys():
                        coordinate = known_stations[name]
                        for key , value in coordinate.items():
                            imputer_fail = SimpleImputer(strategy='constant',fill_value=value)
                            data.loc[idxs,[key]] = imputer_fail.fit_transform(group[[key]])
                    else : 
                        coordinate = getting_lat_long(name)
                        for key , value in coordinate.items():
                            imputer_fail = SimpleImputer(strategy='constant',fill_value=value)
                            data.loc[idxs,[key]] = imputer_fail.fit_transform(group[[key]])
        return data

    
    
    
    
    
    
    
def try_diffrient_classification_models(X,y,X_val,y_val):
    names = [
#         "Logistic regression","GradientBoosting", "Decision Tree",
#          'lgbm',
#         "Random Forest",
#         "KNearest Neighbors",
#         "AdaBoost",
#         "Naive Bayes",
#         "ExtraTree",
#         "sgd",
#         "svc",
        'gaussianprocess',
        'Rbf'
           ]

    classifierse = [
#         LogisticRegression(max_iter=500),
#         GradientBoostingClassifier(random_state=42),
#         DecisionTreeClassifier(),
#         LGBMClassifier(random_state=42),
#         RandomForestClassifier(random_state=42, n_jobs=-1),
#         KNeighborsClassifier(),
#         AdaBoostClassifier(),
#         GaussianNB(),
#         ExtraTreesClassifier(),
#         SGDClassifier(loss= 'log_loss'),
#         SVC(),
        GaussianProcessClassifier(),
        RBF()
    ]
    classifiers =  {x : y for x,y in zip(names,classifierse)}
    numerical_features = ['Latitude', 'Longitude', 'month', 'hour', 'year']
    categorical_features = ['Type', 'Part of a policing operation', 'Gender', 'Age range', 'Officer-defined ethnicity', 'Object of search', 'station']
    numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
    ('onehot', ce.one_hot.OneHotEncoder(handle_unknown='ignore',use_cat_names=True))])

    Dealer = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
    for cal,model in classifiers.items():
        print(f'\nStarting in {cal} ....... \n')
        pipe = Pipeline([
            ('preprocces',Preprocces()),
            ('imputer',FullImputer()),
            ('dealer',Dealer)
            ,('model',model)
                    ])
#         print(pipe.named_steps['model'])
#         scores = cross_validate(pipe, X, y, cv=5, return_train_score=True,scoring='roc_auc')
#         for key, values in scores.items():
#                 print( f"{cal} :", key,' mean ', values.mean())
                
#                 print(f"{cal} :",key,' std ', values.std())
#                 print("=====================================================")
        
        pipe.fit(X,y)
        joblib.dump(pipe, f'pickles/pipes/last_chance/{cal}_pipe.pickle') 
        pred_proba = pipe.predict_proba(X_val)[:,1]
        fpr, tpr, threshold = roc_curve(y_val,pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"for {cal} the roc auc  =  {roc_auc}")
        precision, recall, _ = precision_recall_curve(y_val,pred_proba)

# Calculate average precision
        ap = average_precision_score(y_val,pred_proba)
        print(f"for {cal} the Avrage precison  =  {ap}")
        print(f" the plot for {cal} : /n")
        precision, recall, thresholds = precision_recall_curve(y_val,pred_proba)
        precision = precision[:-1]
        recall = recall[:-1]
        f1_scores = 2*(precision*recall)/(precision + recall)
        min_index = [i for i, rec in enumerate(recall) if rec >= 0.65][-1]
        print(f'recall : {recall[min_index]}')
        print(f'precision : {precision[min_index]}')
        print(f'threshold : {thresholds[min_index]}')

        fig=plt.figure()
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)

        ax1.hlines(y=precision[min_index],xmin=0, xmax=1, colors='red', label=f'precision = {precision[min_index]:.2f}')
        ax2.axvline(thresholds[min_index], color='blue', label=f'threshold = {thresholds[min_index]:.2f}')
        ax2.hlines(y=recall[min_index],xmin=0, xmax=1, colors='red', label=f'recall = {recall[min_index]:.2f}')
        ax1.plot(thresholds,precision)
        ax2.plot(thresholds,recall)

        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])
        ax1.set_ylabel('Precision')
        ax2.set_ylabel('Recall')
        ax1.legend(loc='best')
        ax2.legend(loc='best')

        plt.xlabel('Threshold')
        plt.show()
        print("**************************************************************************************************")
        print("***************************************************************************************************")
        
def GetDiffrentClassifiers(X,y,X_val,y_val):
    names = [
        "Logistic regression","GradientBoosting", "Decision Tree"
        ,"sgd"
        ,"Random Forest",
        "KNearest Neighbors",
        "AdaBoost",
        "Naive Bayes",
        "ExtraTree",
#         "svc",
#         'gaussianprocess'
#         'Rbf'
           ]
    
    for cal in names:
        print(f'\nStarting in {cal} ........ \n')
        pipe = joblib.load(f'pickles/pipes/{cal}_pipe.pickle')
#         print(pipe.named_steps['model'])
#         scores = cross_validate(pipe, X, y, cv=5, return_train_score=True,scoring='roc_auc')
#         for key, values in scores.items():
#                 print( f"{cal} :", key,' mean ', values.mean())
                
#                 print(f"{cal} :",key,' std ', values.std())
#                 print("=====================================================")
        
        pred_proba = pipe.predict_proba(X_val)[:,1]
        fpr, tpr, threshold = roc_curve(y_val,pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"for {cal} the roc auc  =  {roc_auc}")
        threshold = .1         # make this threshold the perfom only the searches more than 0.1 succes probability
        preds = [1 if pred > threshold else 0 for pred in pred_proba]
        print(f'\n The confusion matrix for {cal} ')
        confmat = confusion_matrix(y_true=y_val, y_pred=preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat,
                                       display_labels=pipe.classes_)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

        # add annotations to each square
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(j, i, f"\n{confmat[i, j]:d}\n"
                               f"{'TN' if i == j and i==0 else 'TP' if i== j and i ==1 else 'FP' if j > i else 'FN'}",
                        ha="center", va="center", color="white" if confmat[i, j] > confmat.max() / 2 else "black")

        plt.show()
        print("*************************************************************")
# Replace YOUR_API_KEY with your actual API key
def getting_lat_long(address) : 
    geocoder = OpenCageGeocode("fc77af32e6754b9885e7ee8268bb1d23")

    # Address to be geocoded
    # Perform geocoding
    result = geocoder.geocode(address)

    # Print latitude and longitude
    if result and len(result):
        latitude = result[0]['geometry']['lat']
        longitude = result[0]['geometry']['lng']
#         print(f"Latitude: {latitude}, Longitude: {longitude}")
#     else:
#         print("Unable to geocode the address.")
    return {'Latitude':latitude ,'Longitude' :longitude}



def get_best_match(value, values_list, threshold=60):
    """
    Given an input value and a list of possible matches,
    returns the best match based on fuzzy string matching.
    """
    similarity_scores = [(v, fuzz.token_sort_ratio(value, v)) for v in values_list]
    filtered_scores = [x for x in similarity_scores if x[1] >= threshold]
    if filtered_scores:
        best_match = max(filtered_scores, key=lambda x: x[1])[0]
    else:
        best_match = None
    return best_match


def get_closest_address(latitude, longitude):
    """
    Given a latitude and longitude value, returns the closest address using the OpenCage Geocoding API.
    """
    geocoder = OpenCageGeocode("fc77af32e6754b9885e7ee8268bb1d23")

    # Perform reverse geocoding
    result = geocoder.reverse_geocode(latitude, longitude)

    # Extract the formatted address from the result
    if result and len(result):
        address = result[0]['components']['county']
    else:
        address = None
    return address

def getting_counts(data,col_index , col_columns ):
    group = data.groupby([col_index, 
                col_columns]).count()['Date'].unstack()
    stations_that_all_nulls = [x for x in data[col_index].unique() if x not in group.index]
    second  =data[data[col_columns].isnull()][col_index].value_counts()
    group['missing'] = np.nan
    for id in stations_that_all_nulls:
        group.loc[id] = np.nan
    group.loc[second.index,'missing'] =  second.values
#     return group 
    return round(group.div(group.sum(axis=1 ), axis=0),3).fillna(0)
