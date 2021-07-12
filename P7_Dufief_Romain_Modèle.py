#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator

import seaborn as sns

import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, pipeline


# In[2]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Valeurs Manquantes', 1 : '% des données totales'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% des données totales', ascending=False).round(1)
        
        # Print some summary information
        print ("Le dataframe a " + str(df.shape[1]) + " colonnes.\n"      
            "Il y a  " + str(mis_val_table_ren_columns.shape[0]) +
              " colonnes avec des valeurs manquantes.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
# Affichage de la distribution par subset
def distribSubset(df, subset, numcols):
    if len(numcols)%2 ==0:
        l = len(numcols)//2
    else: l = (len(numcols)//2)+1
        
    fig, ax = plt.subplots(l,2)
    fig.set_size_inches(20, 20)
    i=1

    for a in numcols:
        ax = plt.subplot(l,2,i)
        plt.subplots_adjust()
        plt.xticks(rotation = 90)
        ax = sns.boxplot(x=subset, y=a, data=df)
        i+=1

# Affichage de la matrice de corrélations
def pearsonMatrix(df, annot):
    df_pearson = df.corr(method='pearson').copy()
    plt.figure(figsize=(13,10))
    if annot ==1:
        sns.heatmap(df_pearson, linewidth=0.5,annot=True,annot_kws={"size": 8, "animated":0}, cmap="YlGnBu")
    else:
        sns.heatmap(df_pearson, linewidth=0.5,annot=False, cmap="YlGnBu")
        
    
def encoding_str_data(df_train, df_test):
    '''
    - Assuming train & test have the same columns
    - Lists all columns.
    - The ones that have 2 or less choices are passed on to a LabelEncoder with a fit on train data
    and a transform on both train & test to make sure that transformation is replicable.
    - All other columns with a dtype = 'object' are encoded with OneHotEncoder

    WARNING : this function will replace given columns by encoded ones !
    Make sure you keep a copy of the original data if you need it
    '''

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import pandas as pd

    le = LabelEncoder()
    le_count = 0

    nb_col_str = len(df_train.select_dtypes(include='object').columns)

    for col in df_train:
        if df_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(df_train[col].unique()) <= 2:
                # Train on the training data
                le.fit(df_train[col])
                # Transform both training and testing data
                df_train[col] = le.transform(df_train[col])
                df_test[col] = le.transform(df_test[col])

                # Keep track of how many columns were label encoded
                le_count += 1

    # one-hot encoding of categorical variables
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)

    print('%d columns were label encoded.' % le_count)
    print('%d columns were one-hot encoded.' % (nb_col_str-le_count))
    print('Training Features shape: ', df_train.shape)
    print('Testing Features shape: ', df_test.shape)
    
    return df_train, df_test


def align_columns(df_train, df_test):
    '''
    To be used after string encoding.
    This function will align train & test datasets to make sure they have the same columns.
    '''
    train_labels = df_train['TARGET']

    # Align the training and testing data, keep only columns present in both dataframes
    df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

    # Add the target back in
    df_train['TARGET'] = train_labels

    print('Training Features shape: ', df_train.shape)
    print('Testing Features shape: ', df_test.shape)
    
    return df_train, df_test
    
    
def anomalies_employed_days(df):
    '''
    Detects the anomaly where days_employed has a max of 365 243 days.
    Creates a new column to flag the anomaly and replaces the original values with nan.
    '''
    if df['DAYS_EMPLOYED'].max() == 365243:
        # Create an anomalous flag column
        df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
        # Replace the anomalous values with nan
        df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

        
def printBoxplots(df):
    '''
    Print boxplots for quantitative columns in df.        
    '''
    for col in df:
        df.boxplot(column=col, vert=False)
        plt.show()        


def distributionPlots(df, annot):
    '''
    Plots distributions for quantitative columns in df.
    annot = True if you want AVG & MED values added as a box in the top right corner
    '''
    global fig
    numeric_cols = [col for col in df.columns if df[col].dtype in ('float', 'int')]
    X = df[numeric_cols].values
    if len(df[numeric_cols].columns) > 15:
        fig = plt.figure(figsize=(15,30))
    else: fig = plt.figure(figsize=(15,15))
    
    if annot: 
        for feat_idx in range(df[numeric_cols].shape[1]):
            AVG = X[:, feat_idx].mean()
            MED = np.median(X[:, feat_idx])
            ax = fig.add_subplot((len(df[numeric_cols].columns)//2)+1,3,feat_idx+1)
            h = ax.hist(X[:, feat_idx], bins=50, color='steelblue', density=True, edgecolor='none')
            at = AnchoredText(('AVG = %.0f' % AVG + '\nMED = %.0f' % MED), prop=dict(size=10), frameon=True,loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            ax.set_title(df[numeric_cols].columns[feat_idx], fontsize=14)
        fig.subplots_adjust(hspace=0.9)
    else:
        for feat_idx in range(df[numeric_cols].shape[1]):
            ax = fig.add_subplot((len(df[numeric_cols].columns)//2)+1,3,feat_idx+1)
            h = ax.hist(X[:, feat_idx], bins=50, color='steelblue', density=True, edgecolor='none')
            ax.set_title(df[numeric_cols].columns[feat_idx], fontsize=14)
        fig.subplots_adjust(hspace=0.9)

        
def pipeline_maker(imputer, scaler, estimator):
    '''
    Creates a pipeline based on an imputer, scaler and estimator
    Returns the pipeline ready to fit.
    '''    

    pipe = pipeline.Pipeline([('imputer', imputer),('scaler', scaler), ('estimator', estimator)])
    
    return pipe


# In[3]:


# Nom du dossier où sont stockées les données fournies

path_csv = '/home/romain/Bureau/Formation OpenClassrooms/Projet 07 - Implémentez un modèle de scoring/Projet+Mise+en+prod+-+home-credit-default-risk/'

pd.set_option('display.max_rows', None)


# In[4]:


# Import automatique des fichiers, à l'exception de celui contenant la description des variables

files_list = os.listdir(path_csv)
names = []
files_list.remove('HomeCredit_columns_description.csv')

for file in files_list:
    name = str(file).split('.csv')[0]
    names.append(name)

z=0
files_ready=[]
for f,r in zip(names,files_list):
    globals()[f]=pd.read_csv(path_csv+r)        
    files_ready.append(f)
    z+=1


# In[5]:


#application_train.describe(include='all').T


# In[6]:


#application_train.info(verbose=True)


# In[7]:


cols_to_drop = ['DAYS_EMPLOYED',
                'EXT_SOURCE_1',
                'FLAG_DOCUMENT_6',
                'FLAG_DOCUMENT_8',
                'REGION_POPULATION_RELATIVE',
                'EMERGENCYSTATE_MODE',
                'WALLSMATERIAL_MODE',
                'TOTALAREA_MODE',
                'HOUSETYPE_MODE',
                'FONDKAPREMONT_MODE',
                'NONLIVINGAREA_MEDI',
                'NONLIVINGAPARTMENTS_MEDI',
                'LIVINGAREA_MEDI',
                'LIVINGAPARTMENTS_MEDI',
                'LANDAREA_MEDI',
                'FLOORSMIN_MEDI','FLOORSMAX_MEDI',
                'ENTRANCES_MEDI',
                'ELEVATORS_MEDI','COMMONAREA_MEDI',
                'YEARS_BUILD_MEDI',
                'YEARS_BEGINEXPLUATATION_MEDI',
                'BASEMENTAREA_MEDI','APARTMENTS_MEDI',
                'NONLIVINGAREA_MODE',
                'NONLIVINGAPARTMENTS_MODE',
                'LIVINGAREA_MODE',
                'LIVINGAPARTMENTS_MODE',
                'LANDAREA_MODE',
                'FLOORSMIN_MODE','FLOORSMAX_MODE',
                'ENTRANCES_MODE',
                'ELEVATORS_MODE',
                'COMMONAREA_MODE',
                'YEARS_BUILD_MODE','YEARS_BEGINEXPLUATATION_MODE',
                'BASEMENTAREA_MODE',
                'APARTMENTS_MODE',
                'LIVINGAPARTMENTS_AVG',
                'LIVINGAREA_AVG',
                'APARTMENTS_AVG',
                'FLOORSMIN_AVG',
                'ENTRANCES_AVG',
                'ELEVATORS_AVG',
                'DEF_30_CNT_SOCIAL_CIRCLE',
                'OBS_30_CNT_SOCIAL_CIRCLE',
                'REGION_RATING_CLIENT',
                'REGION_RATING_CLIENT_W_CITY',
                'LIVE_CITY_NOT_WORK_CITY',
                'AMT_ANNUITY',
                'AMT_GOODS_PRICE',
                'CNT_CHILDREN',
                'LIVE_REGION_NOT_WORK_REGION',
                'REG_CITY_NOT_LIVE_CITY',
                'REG_REGION_NOT_WORK_REGION',
                'BASEMENTAREA_AVG',
                'YEARS_BEGINEXPLUATATION_AVG',
                'FLOORSMAX_AVG',
                'FLAG_EMP_PHONE'
]

# On conserve une copie des données avant transformation
app_train = application_train.drop(columns=cols_to_drop).copy()
app_test = application_test.drop(columns=cols_to_drop).copy()


# ## Préparation des données
# 
# * Suppression des anomalies éventuelles
# * Encodage des données texte
# * Préparation des dataset de train & test

# In[8]:


# Suppression des anomalies
#anomalies_employed_days(app_train)
#anomalies_employed_days(app_test)

# Encodage des champs texte
app_train, app_test = encoding_str_data(app_train, app_test)

# Alignement des colonnes train & test
app_train, app_test = align_columns(app_train, app_test)


# In[9]:


# Préparation pour la modélisation

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['SK_ID_CURR','TARGET']).copy()
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.drop(columns = ['SK_ID_CURR']).copy()

X_train, X_test, y_train, y_test = train_test_split(train,app_train['TARGET'],test_size=0.2, random_state=30, stratify=app_train['TARGET'])


# In[10]:


# Avec une recherche en GridSearchCV

imputer = SimpleImputer(strategy = 'median')
scaler = MinMaxScaler(feature_range = (0, 1))

grid = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid = {},cv = 5, scoring='roc_auc')

pipe = pipeline_maker(imputer, scaler, grid)

pipe.fit(X_train, y_train)

print(f"Accuracy = {metrics.accuracy_score(y_test, pipe.predict(X_test)):.3f}")
print(f"ROC_AUC = {pipe.score(X_test, y_test):.2f}")
print(f"F1 Score = {metrics.f1_score(y_test, pipe.predict(X_test), average='weighted'):.2f}")
print(f"Precision = {metrics.precision_score(y_test, pipe.predict(X_test),average='weighted'):.2f}")
print(f"Recall = {metrics.recall_score(y_test, pipe.predict(X_test),average='weighted'):.2f}")


# In[11]:


# Avec une recherche en GridSearchCV

imputer = SimpleImputer(strategy = 'median')
scaler = StandardScaler()

grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=100), param_grid = {},cv = 5, scoring='roc_auc')

pipe = pipeline_maker(imputer, scaler, grid)

pipe.fit(X_train, y_train)

print(f"Accuracy = {metrics.accuracy_score(y_test, pipe.predict(X_test)):.3f}")
print(f"ROC_AUC = {pipe.score(X_test, y_test):.2f}")
print(f"F1 Score = {metrics.f1_score(y_test, pipe.predict(X_test), average='weighted'):.2f}")
print(f"Precision = {metrics.precision_score(y_test, pipe.predict(X_test),average='weighted'):.2f}")
print(f"Recall = {metrics.recall_score(y_test, pipe.predict(X_test),average='weighted'):.2f}")


# In[12]:


#pipe.steps[2][1].best_estimator_.coef_


# In[13]:


# Affichage de la courbe ROC

ns_probs = [0 for _ in range(len(y_test))]
# fit a model
# predict probabilities
lr_probs = pipe.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = metrics.roc_auc_score(y_test, ns_probs)
lr_auc = metrics.roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# In[14]:


#distributionPlots(pd.DataFrame(lr_probs), True)


# In[15]:


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

#features_names = ['input1', 'input2']
#plt.figure(figsize=(15,30))
#f_importances(np.abs(pipe.steps[2][1].best_estimator_.coef_[0]), test.columns)


# In[16]:


y_predict = pipe.predict(test)
y_predict_proba = pipe.predict_proba(test)


# In[17]:


app_test['target'] = y_predict
app_test['proba'] = y_predict_proba[:,1]


# In[18]:


app_test.to_csv('customers_data.csv')

