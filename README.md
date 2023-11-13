#In this project, we will analyze the French Open Damir database and build a machine learning 
regression model to predict the reimbursed amount of healthcare benefits. We will train four 
different models based on the concept of decision trees that use boosting or bagging or both. We will 
use grid search to tune their parameters. Finally, we will evaluate them and choose the best one 
according to specified metrics. The report includes an extensive analysis of the database, the social
security system as well as an introduction to ensemble learning and different machine learning 
techniques used



#ommitting FLT columns for functional reasons
cols=['ORG_CLE_REG', 'AGE_BEN_SNDS', 'BEN_RES_REG','BEN_CMU_TOP', 'BEN_QLT_COD', 'BEN_SEX_COD','DDP_SPE_COD','ETE_CAT_SNDS', 'ETE_REG_COD', 'ET
E_TYP_SNDS','ETP_REG_COD','ETP_CAT_SNDS', 'MDT_TYP_COD', 'MFT_COD', 'PRS_FJH_TYP', 'SOI_ANN','SOI_MOI', 'ASU_NAT', 'ATT_NAT', 'CPL_COD', 'CPT_ENV_TYP'
,'DRG_AFF_NAT','ETE_IND_TAA', 'EXO_MTF', 'MTM_NAT', 'PRS_NAT','PRS_PPU_SEC','PRS_REM_TAU','PRS_ACT_NBR', 'PRS_ACT_QTE', 'PRS_PAI_MNT', 'PRS_REM_MNT
','PRS_REM_TYP','PRS_PDS_QCP','EXE_INS_REG', 'PSE_ACT_SNDS', 'PSE_ACT_CAT', 'PSE_SPE_SNDS','PSE_STJ_SNDS', 'PRE_INS_REG', 'PSP_ACT_SNDS', 'PSP_ACT_CAT','P
SP_SPE_SNDS','PSP_STJ_SNDS', 'TOP_PS5_TRG']

#sepcifying explicitly categorical columns

categ_cols=set(cols).difference(set(['PRS_ACT_NBR', 'PRS_ACT_QTE', 'PRS_PAI_MNT', 'PRS_REM_MNT', 'PRS_REM_TAU']))
dic_types={}
for i in categ_cols:
dic_types[i]='category'
%%time

#removing cpl_cod <> 0 and deleting cpl_cod
df=df.drop(df[df.CPL_COD!='0'].index).reset_index(drop=True)
del df['CPL_COD']

#removing rows with BEN _SEX_COD = 0
df=df.drop(df[df.BEN_SEX_COD =='0'].index).reset_index(drop=True)
df.shape
df.info();

#removing missing values (remove_categories() will remove a category class and automatically replace it with np.nan)

for col in ['AGE_BEN_SNDS',
'ASU_NAT',
'BEN_RES_REG',
'PSP_SPE_SNDS',
'PSP_ACT_SNDS',
'PSE_SPE_SNDS',
'PSE_ACT_SNDS',
'PRS_REM_TYP',
'PRS_PDS_QCP',
'PRE_INS_REG',
'ORG_CLE_REG',
'MFT_COD',
'EXO_MTF',
'EXE_INS_REG',
'ETP_REG_COD',
'ETE_REG_COD',
'ETE_TYP_SNDS']:
if '99' in df[col].cat.categories:
df[col]=df[col].cat.remove_categories(['99'])
for col in ['ATT_NAT',
'BEN_CMU_TOP',
'BEN_QLT_COD',
'BEN_SEX_COD',
'CPT_ENV_TYP',
'TOP_PS5_TRG',
'PSP_STJ_SNDS',
'PSE_STJ_SNDS',
'MTM_NAT',
'ETE_IND_TAA']:
if '9' in df[col].cat.categories:
df[col]=df[col].cat.remove_categories(['9'])
for col in ['PSP_SPE_SNDS',
'PSP_ACT_SNDS',
'PSE_SPE_SNDS',
'PSE_ACT_SNDS']:
if '0' in df[col].cat.categories:
df[col]=df[col].cat.remove_categories(['0'])
for col in ['ETE_CAT_SNDS',
'PRS_NAT',
'ETP_CAT_SNDS']:
if '9999' in df[col].cat.categories:
df[col]=df[col].cat.remove_categories(['9999'])
if '24' in df.DRG_AFF_NAT.cat.categories:
df.DRG_AFF_NAT=df.DRG_AFF_NAT.cat.remove_categories(['24'])
#computing null values ratio for each column
df.isnull().mean().round(2).sort_values(ascending = False)
#removing columns with nan ratio > 0.5
a=df.isnull().mean().round(2).sort_values(ascending = False)
df=df[list(a[a < .5].index)]
df.shape
df.SOI_ANN.value_counts()

#removing observations with 0000 or 0001 SOI_ANN values

df=df.drop(df[(df.SOI_ANN=='0000')|(df.SOI_ANN=='0001')].index).reset_in
67
dex(drop=True)
df.SOI_ANN=df.SOI_ANN.cat.remove_categories(['0000', '0001'])
df.shape
#basic imputation
for col in ['PRS_ACT_NBR', 'PRS_ACT_QTE', 'PRS_PAI_MNT', 'PRS_REM_MNT',
'PRS_REM_TAU']:
df[col].fillna((df[col].median()), inplace=True)
df[col]=df[col].round(1)
for col in set(df.columns).difference(set(['PRS_ACT_NBR', 'PRS_ACT_QTE',
'PRS_PAI_MNT','PRS_REM_MNT', 'PRS_REM_TAU'])):
df[col].fillna((df[col].mode()[0]), inplace=True)

#check of nan values

df.isnull().mean().round(2).sort_values(ascending = False)
#ATT_NAT values
df.ATT_NAT.value_counts()

#removal of ATT_NAT and last check of df dimension before dropping duplicates

del df['ATT_NAT']
df.shape
%%time
df_train=df.iloc[0:25000000,:].drop_duplicates()
df_test=df.iloc[25000000:,:].drop_duplicates()
%%time
for col in set(df_train.columns).difference(set(['PRS_ACT_NBR', 'PRS_ACT_QTE', 'PRS_PAI_MNT', 'PRS_REM_MNT', 'PRS_REM_TAU'])):
df_train[col]=df_train[col].astype(int)
df_train=reduce_mem_usage(df_train)

#Exploratory data analysis 
#histograms
df_hist=df_train[['PRS_REM_MNT', 'PRS_PAI_MNT']]

#normalizing data
scaler = StandardScaler()
scaler.fit(df_hist)

#retransforming data from array to dataframe

data=pd.DataFrame(scaler.transform(df_hist).round(2), columns=df_hist.columns)
bin_values = np.arange(start=-10, stop=10, step=1)
ax=df_hist.hist(bins=bin_values, figsize=[22,6]);

#lineplot
fig, ax = plt.subplots(figsize=(22,5))
ax = sns.lineplot(ax=ax, x="SOI_ANN", y='PRS_REM_TAU', data=df_train);

#barplots
fig, ax = plt.subplots(2, 3, figsize=(22, 15))
sns.countplot(x="BEN_RES_REG", hue="BEN_SEX_COD", data=df_train, ax=ax[0,0], palette="dark")
sns.countplot(x="BEN_QLT_COD", hue="BEN_SEX_COD", data=df_train, palette="dark", ax=ax[0,2])
sns.countplot(x="AGE_BEN_SNDS", hue="PRS_PPU_SEC", data=df_train, palette="dark", ax=ax[1,0])
sns.countplot(x="PRS_PPU_SEC", hue="BEN_CMU_TOP", data=df_train, palette="dark", ax=ax[1,2])

plt.tight_layout();
#pie charts
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))
ax1.pie(df_train['BEN_RES_REG'].value_counts().values, labels=df_train['BEN_RES_REG'].value_counts().index, autopct='%1.1f%%', shadow=True, explode =(0.1,0,0,0,0,0,0,0,0,0,0,0,0));
ax2.pie(df_train['AGE_BEN_SNDS'].value_counts().values, labels=df_train['AGE_BEN_SNDS'].value_counts().index, autopct='%1.1f%%', shadow=True, ex
plode = (0.1,0,0,0,0,0,0,0));
ax3.pie(df_train['BEN_SEX_COD'].value_counts().values, labels=df_train['BEN_SEX_COD'].value_counts().index, autopct='%1.1f%%', shadow=True, expl
ode = (0.1,0));
ax1.set_title('BEN_RES_REG');
ax2.set_title('AGE_BEN_SNDS');
ax3.set_title('BEN_SEX_COD');

plt.tight_layout();
plt.show()

#correlation matrix
associations(df_train, figsize=(25, 20));

# Extract X and y
y = df_train['PRS_REM_MNT']
X = df_train.drop('PRS_REM_MNT', axis = 1)

# Training and Testing Sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = .2, 
random_state = 42, shuffle=True)
#Random forest tuned model
#default random forest
rf = RandomForestRegressor(random_state = 42, n_jobs=-1)

# grid to tune over
gs_grid = {
#Number of trees in random forest
'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
#Number of features to consider at every split
'max_features': ['auto', 'sqrt'],
#Maximum number of levels in tree
'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
#Minimum number of samples required to split a node'min_samples_split': [i for i in range(2,8)],
#Minimum number of samples required at each leaf node'min_samples_leaf': [i for i in range(1,4)],
#Method of selecting samples for training each tree
'bootstrap': [True, False]}

#grid search initalization
rf = GridSearchCV(estimator=rf, param_grid=gs_grid, cv = 3, verbose=2, return_train_score=True)
#Fitting the random search model and looking for optimal parameters
rf.fit(Xtrain.iloc[0:6000000,:], ytrain[0:6000000]);
#Xgboost tuned model
#default Xgboost
xg = xgb(random_state = 42, n_jobs=-1, eval_metric='rmse')
#for tuning parameters
gs_grid = {'colsample_bytree':[round(.2*i,1) for i in range(1,5)],
'gamma':[0,.03,.1,.3],
'min_child_weight':[2*i-1 for i in range(1,11)],
'learning_rate':[round(.02*i-.01,2) for i in range(1,6)],
'max_depth':[int(x) for x in np.linspace(10, 110, num = 11)],
'n_estimators':[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
'reg_alpha':[1e-5, 1e-2, 0.75],
'reg_lambda':[1e-5, 1e-2, 0.45],
'subsample':[round(.3*i,1) for i in range(1,4)]}

#grid search initalization
xg = GridSearchCV(estimator=xg, param_grid=gs_grid, cv = 3, verbose=2, return_train_score=True)

#Fitting the random search model and looking for optimal parameters
xg.fit(Xtrain.iloc[0:6000000,:], ytrain[0:6000000]);

#catboost tuned model
#default catboost
catb = cat(random_state = 42, task_type='GPU', use_best_model=True, eval_metric='RMSE')
#for tuning parameters
gs_grid = {
'learning_rate':[round(.02*i-.01,2) for i in range(1,6)],
'l2_leaf_reg':[5*i+1 for i in range(10)]+[100],
'border_count':[i for i in range(5,100,15)]+[200],
'ctr_border_count':[i for i in range(5,100,15)]+[200],
'max_depth':[int(x) for x in np.linspace(10, 110, num = 11)],
'colsample_bytree':[round(.2*i,1) for i in range(1,5)],
'n_estimators':[int(x) for x in np.linspace(start = 200, stop = 2000
, num = 10)]
}

#grid search initalization
catb = GridSearchCV(estimator=catb, param_grid=gs_grid, cv = 3, verbose=2, return_train_score=True)
#Fitting the random search model and looking for optimal parameters
catb.fit(Xtrain.iloc[0:6000000,:], ytrain[0:6000000]);
#lightgbm tuned model
#default lightgbm
lgbmr = lgbm(random_state = 42, device='gpu', metric='rmse')
#for tuning parameters
gs_grid = {
'learning_rate':[round(.02*i-.01,2) for i in range(1,6)],
'n_estimators':[int(x) for x in np.linspace(start = 200, stop = 2000
, num = 10)],
'max_depth':[int(x) for x in np.linspace(10, 110, num = 11)],
'gamma':[0,.03,.1,.3],
'min_child_weight':[2*i-1 for i in range(1,11)],
'subsample':[round(.3*i,1) for i in range(1,4)],
#'colsample_bytree':[round(.2*i,1) for i in range(1,5)],
'reg_alpha':[1e-5, 1e-2, 0.75],
'reg_lambda':[1e-5, 1e-2, 0.45]
}
#grid search initalization
lgbmr = GridSearchCV(estimator=lgbmr, param_grid=gs_grid, cv = 3, verbos
e=2, return_train_score=True)
#Fitting the random search model and looking for optimal parameters
lgbmr.fit(Xtrain.iloc[0:6000000,:], ytrain[0:6000000]);
#exporting models for later use
warnings.filterwarnings('ignore')
dump_pickle(catb, open('cat.sav', 'wb'))
dump_pickle(xg, open('xg.sav', 'wb'))
dump_pickle(lgbmr, open('lgbmr.sav', 'wb'))
dump_pickle(rf, open('rf.sav', 'wb'))
#uploading models to colab from google drive
downloaded = drive.CreateFile({'id':"12wT37bvQ3TfUbF-R8s1vUN6H6b9tZRL"})
downloaded.GetContentFile('cat.sav')
downloaded = drive.CreateFile({'id':"1o8VX6bscgS343FUbdzFkMJKOqJIAVx4"})
downloaded.GetContentFile('lgbmr.sav')
downloaded =drive.CreateFile({'id':"1Z5jcm7dJUWsYx2Q95ztBsssOTP2sCWWw"})
downloaded.GetContentFile('rf.sav')
downloaded =drive.CreateFile({'id':"17fICWHmtnkwgzWbdSEzj0zTvBA5Yt90"})
downloaded.GetContentFile('xg.sav')
#importing models
warnings.filterwarnings('ignore')
m_cat = load_pickle(open('cat.sav', 'rb'))
m_xg = load_pickle(open('xg.sav', 'rb'))
m_lgbm = load_pickle(open('lgbmr.sav', 'rb'))
m_rf = load_pickle(open('rf.sav', 'rb'))
#get lgbm params
m_lgbm.params
#get rf params
m_rf.get_params()
71
#get xgboost params
m_xg.get_params()
#get catboost params
m_cat.get_params()
# plot feature importance rf
fig, ax = plt.subplots(1, 1, figsize=(22, 15))
pd.Series(m_rf.feature_importances_, index=X.columns).sort_values().plot
(kind='barh', ax=ax, title='Random forest feature importances');
# plot feature importance xgboost
fig, ax = plt.subplots(1, 1, figsize=(22, 15))
plot_importance(m_xg, ax=ax, title='XGboost feature importances', import
ance_type='gain', show_values=False)
plt.show()
# plot feature importance lgbm
fig, ax = plt.subplots(1, 1, figsize=(22, 15))
pd.Series(m_lgbm.feature_importance(importance_type='gain'), index=X.col
umns).sort_values().plot(kind='barh', ax=ax, title='LGBM feature importa
nces');
# plot feature importance catboost 
fig, ax = plt.subplots(1, 1, figsize=(22, 15))
pd.Series(m_cat.get_feature_importance(), index=X.columns).sort_values()
.plot(kind='barh', ax=ax, title='Catboost feature importances');
for col in set(df_test.columns).difference(set(['PRS_ACT_NBR', 'PRS_ACT_
QTE', 'PRS_PAI_MNT', 'PRS_REM_MNT', 'PRS_REM_TAU'])):
df_test[col]=df_test[col].astype(int)
df_test=reduce_mem_usage(df_test)
# Extract X and y
ytest = df_test['PRS_REM_MNT']
Xtest = df_test.drop('PRS_REM_MNT', axis = 1)
# plot rf metrics
print('mae = ' + str(round(mean_absolute_error(ytest, m_rf.predict(Xtest
)))) + '\n' + 'rmse = ' +\
str(round(np.sqrt(mean_squared_error(ytest, m_rf.predict(Xtest))))
) + '\n' + 'evc = ' +str(round(explained_variance_score(ytest, m_rf.pred
ict(Xtest)),2)))
# plot xg metrics
print('mae = ' + str(round(mean_absolute_error(ytest, m_xg.predict(Xtest
)))) + '\n' + 'rmse = ' +\
str(round(np.sqrt(mean_squared_error(ytest, m_xg.predict(Xtest))))
)+ '\n' + 'evc = ' +str(round(explained_variance_score(ytest, m_xg.predi
ct(Xtest)),2)))
72
# plot catboost metrics
print('mae = ' + str(round(mean_absolute_error(ytest, m_cat.predict(Xtes
t)))) + '\n' + 'rmse = ' +\
str(round(np.sqrt(mean_squared_error(ytest, m_cat.predict(Xtest)))
))+ '\n' + 'evc = ' +str(round(explained_variance_score(ytest, m_cat.pre
dict(Xtest)),2)))
# plot lgbm metrics
print('mae = ' + str(round(mean_absolute_error(ytest, m_lgbm.predict(Xte
st)))) + '\n' + 'rmse = ' +\
str(round(np.sqrt(mean_squared_error(ytest, m_lgbm.predict(Xtest))
)))+ '\n' + 'evc = ' +str(round(explained_variance_score(ytest, m_lgbm.p
redict(Xtest)),2))
