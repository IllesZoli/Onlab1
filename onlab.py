import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import mean
import os
import datetime
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

register_matplotlib_converters()
ldir=os.chdir(r"C:\Users\illes\Documents\Python jupiter\Mester önlab1\F1 data")
#%%heights
import requests

url = 'https://www.lightsoutblog.com/2021/04/22/the-height-of-every-f1-driver-from-the-2000s/amp/?fbclid=IwAR2jCkAvIPrJXrd9HbEXKDeUOcO2QKLtY3mpsGpq5eau7O0TBkfad07fL9o'

header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}

r = requests.get(url, headers=header)

height = pd.read_html(r.text)
height=height[0] #heights since 2000
height=height.rename(columns={"Driver": "Full name"})

#%% Loading data
circuits= pd.read_csv('circuits.csv')
constructors= pd.read_csv('constructors.csv')
drivers= pd.read_csv('drivers.csv')
pit_stops= pd.read_csv('pit_stops.csv')
races= pd.read_csv('races.csv')
results= pd.read_csv('results.csv')
status= pd.read_csv('status.csv')
weather_info= pd.read_excel('weather_modified.xlsx')
#%% formating data
weather_info=weather_info.drop(['weather','round'],
                               axis = 1)

#calculating lap numbers
laps=results.groupby('raceId').max().reset_index()[['raceId','laps']]
laps=laps.rename(columns={"laps": "Max_laps"})

races=races.drop(races.iloc[:, 5:],axis = 1)
races=races.drop(['round'],axis = 1)

drivers['Full name']=drivers['forename']+' '+drivers['surname']
drivers = pd.merge(drivers, height, how='left', 
               on=['Full name'])
drivers['Height (cm)'].fillna((drivers['Height (cm)'].mean()), inplace=True)

drivers['dob']=drivers['dob'].str[:4]
drivers=drivers[['driverId','Full name','dob','Height (cm)']]
#small fix
drivers.loc[drivers["Full name"] =='Kimi Räikkönen', "Height (cm)"] = 175

results=results[['resultId','raceId','driverId','constructorId','grid','laps','statusId']]
results=results.rename(columns={"grid": "Starting_position","laps":"Completed_laps"})

constructors=constructors.drop(constructors.iloc[:, 3:],axis = 1)
constructors=constructors.drop(['constructorRef'],axis = 1)

pit_stops=pit_stops.drop(['stop','lap','time'],axis = 1)

street_circuits=['adelaide','albert_park','baku','las_vegas','monaco','pedralbes','villeneuve','boavista','monsanto','dallas','detroit','jeddah','long_beach','marina_bay','miami','montjuic','phoenix','valencia'] 
circuits['Street'] = np.where(circuits['circuitRef'].isin(street_circuits), 1, 0)
circuits=circuits[['circuitId','name','Street','Length','Number of turns']]


finished_status_list=[1,11,12,13,14,15,16,17,18,19,45,50,128,53,55,58,88,111,112,113,114,115,116,117,118,119,120,122,123,124,125,127,133,134,81,97] #status Id for did not finishes
#personal_fail_list=[2,3,4,20,104]
#status['DNF'] = np.where(status['statusId'].isin(finished_status_list), '2', )
status['DNF'] = np.where(status['statusId'].isin(finished_status_list), '0', '1')
#%% joining
con1 = pd.merge(races, weather_info, how='inner', 
               on=['raceId','year'])

con2 = pd.merge(results, status, how='inner', 
               on=['statusId'])

con3 = pd.merge(con2, drivers, how='inner', 
               on=['driverId'])

con4 = pd.merge(con3, constructors, how='inner', 
               on=['constructorId'])

df_for_dnf_s = pd.merge(con4, con1, how='inner', 
               on=['raceId'])

df_for_dnf_s=df_for_dnf_s.rename(columns={"name_x": "Constructor", "name_y": "Race name"})

df_for_dnf_s = pd.merge(df_for_dnf_s, circuits, how='inner', 
               on=['circuitId'])

df_for_dnf_s = df_for_dnf_s.drop('Race name', axis=1)
df_for_dnf_s=df_for_dnf_s.rename(columns={"name": "Race name"})
df_for_dnf_s['Age']=df_for_dnf_s['year'].astype(int)-df_for_dnf_s['dob'].astype(int)

df_for_dnf_s = pd.merge(df_for_dnf_s, laps, how='inner', 
               on=['raceId'])

df_for_pits = pd.merge(df_for_dnf_s, pit_stops, how='inner', 
               on=['raceId','driverId'])   #much smaller dataset as data is only from 2011

df_for_pits=df_for_pits.rename(columns={"name_x": "Constructor", "name_y": "Race name"})

df_for_pits = pd.merge(df_for_pits, circuits, how='inner', 
               on=['circuitId'])

df_for_pits = df_for_pits.drop(['Race name','Street_x'], axis=1)
df_for_pits=df_for_pits.rename(columns={'Street_y':'Street',"name": "Race name"})
#%% DNFs/driver for analysis
driver_race_amount=df_for_dnf_s.groupby('Full name').count().reset_index()[['Full name','raceId']]
driver_dnf_amount=df_for_dnf_s.groupby(['Full name', 'DNF']).count()[['raceId']]
driver_dnf_amount=driver_dnf_amount.reset_index()
dnfs=driver_dnf_amount.loc[driver_dnf_amount['DNF'] == '1'][['Full name','raceId']]
driver_race_amount = pd.merge(driver_race_amount, dnfs, how='left', 
               on=['Full name']) 
driver_race_amount = driver_race_amount.fillna(0)
driver_race_amount=driver_race_amount.rename(columns={"raceId_x": "Number of races", "raceId_y": "Number of DNFs"})
driver_race_amount['DNF Percent']=driver_race_amount['Number of DNFs']/driver_race_amount['Number of races']*100
driver_race_amount['Finished']=driver_race_amount['Number of races']-driver_race_amount['Number of DNFs']
##number of finishes vs dnfs
race=df_for_dnf_s['DNF'].value_counts().to_frame().reset_index()


#DNFs/race for analysis
df_for_dnf_s['DNF']=df_for_dnf_s['DNF'].astype(int)
prep_df=pd.merge(races, circuits, how='left', 
               on=['circuitId'])

race_df=prep_df["name_y"].value_counts().to_frame().reset_index()
race_df=race_df.rename(columns={"index": "Race name", "name_y": "Number of races"})
race_dnfs=df_for_dnf_s.loc[df_for_dnf_s['DNF'] == 1]
race_dnfs=race_dnfs["Race name"].value_counts().to_frame().reset_index()
race_dnfs=race_dnfs.rename(columns={"index": "Race name", "Race name": "Number of DNFs"})
race_dnf_df = pd.merge(race_df, race_dnfs, how='left',on=['Race name']) 
race_dnf_df['Number of DNFs per race']=race_dnf_df['Number of DNFs']/race_dnf_df['Number of races']
#DNFs/constuctor for analysis
cons_df=con4['name'].value_counts().to_frame().reset_index()
cons_df=cons_df.rename(columns={"index": "Constructor", "name": "Number of races"})
cons_dnfs=df_for_dnf_s.loc[df_for_dnf_s['DNF'] == 1]
cons_dnfs=cons_dnfs["Constructor"].value_counts().to_frame().reset_index()
cons_dnfs=cons_dnfs.rename(columns={"index": "Constructor", "Constructor": "Number of DNFs"})
cons_dnf_df = pd.merge(cons_df, cons_dnfs, how='left', 
               on=['Constructor']) 
cons_dnf_df['Number of DNFs']=cons_dnf_df['Number of DNFs'].fillna(0)
cons_dnf_df['Number of DNFs per race']=cons_dnf_df['Number of DNFs']/cons_dnf_df['Number of races']

pit_stats=df_for_pits.describe().T
#%%DNF analysis
#finished vs dnf
keys = ['Finished', 'DNF']
plt.pie(race['DNF'].T, labels=keys, autopct='%.0f%%')

#DNFs over the years 
df_for_dnf_s['DNF']=df_for_dnf_s['DNF'].astype(int)
df_dnf_year=df_for_dnf_s.groupby('year').sum().reset_index()[['year','DNF']]
plt.figure(figsize=(15,6))
ax6=sns.barplot(data=df_dnf_year, x='year', y="DNF")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)
#linechart
plt.figure(figsize=(15,6))
ax6=sns.lineplot(data=df_dnf_year, x='year', y="DNF")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)
#DNF per racetrack need to rotate xlabel
ax7=sns.displot(df_for_dnf_s, x="Race name", hue="DNF", multiple="stack")
ax8=sns.barplot(data=race_dnf_df.loc[race_dnf_df['Number of races'] >5].sort_values('Number of DNFs per race', ascending=False).head(10), y='Number of DNFs per race',x='Race name')
ax8.set_xticklabels(ax8.get_xticklabels(), rotation=90)
#driver dnf per racer
v = df_for_dnf_s['Full name'].value_counts().head(20)
#calc1=df_for_dnf_s[df_for_dnf_s['Full name'].isin(v.index[v.gt(60)])]
sns.catplot(data=df_for_dnf_s[df_for_dnf_s['Full name'].isin(v.index[v.gt(20)])], y="Full name", hue="DNF", kind="count")
#constructor overall and one specific team
#ferrari specific dnfs
ferrari=df_for_dnf_s.groupby(['Constructor', 'year']).sum().reset_index()[['Constructor','year','DNF']]
plt.figure(figsize=(15,6))
ax9=sns.lineplot(data=ferrari.loc[ferrari['Constructor'] == 'Ferrari'], x='year', y="DNF")
ax9.yaxis.get_major_locator().set_params(integer=True)

plt.figure(figsize=(15,6))
ax9=sns.barplot(data=df_for_dnf_s.loc[(df_for_dnf_s['Starting_position'] > 0)], x='Starting_position', y="DNF")
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=90)

plt.figure(figsize=(14,6))
df_for_dnf_s.loc[(df_for_dnf_s['Starting_position'] > 0)].groupby('Starting_position').sum()['DNF'].plot(kind="bar")
#crashes per turn number
calc1=df_for_dnf_s.groupby('Number of turns').sum()['DNF']
calc2=df_for_dnf_s['Number of turns'].value_counts()
merged= pd.merge(calc1, calc2, how='inner', left_index=True, right_index=True) 
merged['ratio that crashed']=merged['DNF']/merged['Number of turns']
ax20=sns.scatterplot(data=merged.reset_index(), x="index", y='ratio that crashed')
ax20.set(xlabel='Number of turns')

df_for_dnf_s['Total length']=df_for_dnf_s['Max_laps']*df_for_dnf_s['Length']
calc1=df_for_dnf_s.groupby('Total length').sum()['DNF']
calc2=df_for_dnf_s['Total length'].value_counts()
merged= pd.merge(calc1, calc2, how='inner', left_index=True, right_index=True) 
merged['ratio that did not finish']=merged['DNF']/merged['Total length']
ax20=sns.scatterplot(data=merged.reset_index(), x="index", y='ratio that did not finish')
ax20.set(xlabel='Total length')


from  matplotlib.ticker import FuncFormatter
#monaco kiesők évről évre
monaco=df_for_dnf_s.loc[(df_for_dnf_s['DNF'] == 1) & (df_for_dnf_s['circuitId']==6)]
plt.figure(figsize=(14,6))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
ax8=sns.countplot(data=monaco, x="year")
ax8.set_xticklabels(ax8.get_xticklabels(), rotation=90)


df_for_dnf_s['Length']=df_for_dnf_s['Length'].str[:5].astype(float)

calc1=df_for_dnf_s.groupby('Age').count()['DNF'].reset_index()
calc1['number']=df_for_dnf_s.groupby('Age').sum()['DNF'].reset_index()['DNF']

calc1['calc']=calc1['number']/calc1['DNF']
plt.figure(figsize=(14,6))
ax21=sns.barplot(data=calc1, x='Age', y="calc")
ax21.set(ylabel='ratio that did not finish')
#%%
#for drivers
df_for_dnf_s=df_for_dnf_s.sort_values(['driverId', 'year','raceId'],ascending = [True, True, True])
df_for_dnf_s['Driver DNF accoumulateion']=df_for_dnf_s.groupby(['Full name'])['DNF'].cumsum()
df_for_dnf_s['Driver DNF accoumulateion']=df_for_dnf_s['Driver DNF accoumulateion']+1
df_for_dnf_s['Driver Completed Races']=df_for_dnf_s.groupby(['Full name'])['DNF'].cumcount()
df_for_dnf_s['Driver Completed Races']=df_for_dnf_s['Driver Completed Races']+1
df_for_dnf_s['Driver DNF ratio']=df_for_dnf_s['Driver DNF accoumulateion']/df_for_dnf_s['Driver Completed Races']*100
#for constructors
df_for_dnf_s=df_for_dnf_s.sort_values(['constructorId', 'year','raceId'],ascending = [True, True, True])
df_for_dnf_s['Constructor DNF accoumulateion']=df_for_dnf_s.groupby(['constructorId'])['DNF'].cumsum()
df_for_dnf_s['Constructor DNF accoumulateion']=df_for_dnf_s['Constructor DNF accoumulateion']+1
df_for_dnf_s['Constructor Completed Races']=df_for_dnf_s.groupby(['constructorId'])['DNF'].cumcount()
df_for_dnf_s['Constructor Completed Races']=df_for_dnf_s['Constructor Completed Races']+1
df_for_dnf_s['Constructor DNF ratio']=df_for_dnf_s['Constructor DNF accoumulateion']/df_for_dnf_s['Constructor Completed Races']*100

#for circuits
df_for_dnf_s=df_for_dnf_s.sort_values(['circuitId', 'year','raceId'],ascending = [True, True, True])
df_for_dnf_s['Circuit DNF accoumulateion']=df_for_dnf_s.groupby(['circuitId'])['DNF'].cumsum()
#%%feature selection
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = df_for_dnf_s.select_dtypes(include=numerics)
newdf=newdf.drop(columns=['resultId','statusId','Completed_laps','raceId'])
newdf=newdf.loc[newdf['year'] > 2010]

Y = newdf['DNF']
X= newdf.loc[:, newdf.columns != 'DNF']

# performing preprocessing part
sc = StandardScaler()
  
#%%k best
# ANOVA feature selection for numeric input and categorical output
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# define feature selection
fs = SelectKBest(score_func=f_classif, k=10)
# apply feature selection
X_selected = fs.fit_transform(X, Y)
fs.get_feature_names_out(input_features=None)
x_train, x_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#%% XGBoost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}

clf = XGBClassifier(objective="binary:logistic")
grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=-1, cv=3).fit(X_train, y_train)

print("Param for GS", grid_cv.best_params_)
print("CV score for GS", grid_cv.best_score_)
print("Train AUC ROC Score for GS: ", roc_auc_score(y_train, grid_cv.predict(X_train)))
print("Test AUC ROC Score for GS: ", roc_auc_score(y_test, grid_cv.predict(X_test)))

clf.fit(X_train, y_train)
# make predictions for test data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(confusion_matrix(y_test, y_pred)) 
#print(roc_auc_score(x_train, x_test))
print(roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%smote over 79% 0,55 roc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_resample(X_train,y_train)
rf=RandomForestClassifier(criterion='gini',max_depth=5,min_samples_leaf=1,min_samples_split=2,n_estimators=600)
rf.fit(smote_train, smote_target)
y_pred=rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
#print(roc_auc_score(x_train, x_test))
print(roc_auc_score(y_test, y_pred))

print('Accuracy of classifier on training set: {:.2f}'
     .format(rf.score(smote_train, smote_target)))
print('Accuracy of classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
#%%
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
param_grid = {
    "max_depth": [3,5,10,15,20,None],
    "min_samples_split": [2,5,7,10,20],
    "min_samples_leaf": [1,2,5,8]
}

clf = DecisionTreeClassifier(random_state=42)
grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=-1, cv=3).fit(smote_train, smote_target)

print("Param for GS", grid_cv.best_params_)
print("CV score for GS", grid_cv.best_score_)
print("Train AUC ROC Score: ", roc_auc_score(smote_target, grid_cv.predict(smote_train)))
print("Test AUC ROC Score: ", roc_auc_score(y_test, grid_cv.predict(X_test)))
print(classification_report(y_test, y_pred))
#%%
importance = pd.Series(data=rf.feature_importances_)
#%%over és under sample egyszerre 73-75szazalek 0,55roc
from imblearn.under_sampling import RandomUnderSampler
#oversampler=SMOTE(random_state=0)
oversampler=SMOTE(random_state=0,sampling_strategy=0.5)
smote_train, smote_target = oversampler.fit_resample(x_train,y_train)
under = RandomUnderSampler(sampling_strategy=0.3)
smote_train, smote_target = under.fit_resample(x_train,y_train)
rf=RandomForestClassifier(criterion='gini',max_depth=30,min_samples_leaf=5,min_samples_split=6,n_estimators=800,class_weight='balanced')
rf.fit(smote_train, smote_target)
y_pred=rf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) 
print(roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
range_k = range(1,15)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=3)
   classifier.fit(X_train, y_train)
   y_pred = classifier.predict(X_test)
   scores[k] = roc_auc_score(y_test,y_pred)
   scores_list.append(roc_auc_score(y_test,y_pred))
result = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = roc_auc_score(y_test, y_pred)
print("Classification Report:",)
print (result1)
import matplotlib.pyplot as plt
plt.plot(range_k,scores_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
print(classification_report(y_test, y_pred))