import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error , hamming_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss


dataframe = pd.read_csv("drive/app/Frogs_MFCCs.csv")
# dataframe = pd.read_csv("/media/ghost/Games And Images/INF552/hw4/Anuran Calls (MFCCs)/Frogs_MFCCs.csv")

dataframe.drop('RecordID',axis=1,inplace=True)
dataframe.replace('Bufonidae',int(1),inplace=True)
dataframe.replace('Dendrobatidae',int(2),inplace=True)
dataframe.replace('Hylidae',int(3),inplace=True)
dataframe.replace('Leptodactylidae',int(4),inplace=True)

dataframe.replace('Adenomera',int(10),inplace=True)
dataframe.replace('Ameerega',int(11),inplace=True)
dataframe.replace('Dendropsophus',int(12),inplace=True)
dataframe.replace('Hypsiboas',int(13),inplace=True)
dataframe.replace('Leptodactylus',int(14),inplace=True)
dataframe.replace('Osteocephalus',int(15),inplace=True)
dataframe.replace('Rhinella',int(16),inplace=True)
dataframe.replace('Scinax',int(17),inplace=True)

dataframe.replace('AdenomeraAndre',int(100),inplace=True)
dataframe.replace('AdenomeraHylaedactylus',int(101),inplace=True)
dataframe.replace('Ameeregatrivittata',int(102),inplace=True)
dataframe.replace('HylaMinuta',int(103),inplace=True)
dataframe.replace('HypsiboasCinerascens',int(104),inplace=True)
dataframe.replace('HypsiboasCordobae',int(105),inplace=True)
dataframe.replace('LeptodactylusFuscus',int(106),inplace=True)
dataframe.replace('OsteocephalusOophagus',int(107),inplace=True)
dataframe.replace('Rhinellagranulosa',int(108),inplace=True)
dataframe.replace('ScinaxRuber',int(109),inplace=True)


# dataframe.to_csv('/media/ghost/Games And Images/INF552/hw4/Frogs_MFCCs-changed.csv',index=False)
dataframe.to_csv('drive/app/Frogs_MFCCs-changed.csv',index=False)
# print(dataframe)

dataset=dataframe.values
# print(dataset.shape)
shuf_ds = shuffle(dataset, random_state=0)

X=shuf_ds[:,:22]
Y=shuf_ds[:,22:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

Y_train_family=Y_train[:,:1]
Y_test_family=Y_test[:,:1]

Y_train_genus= Y_train[:,1:2]
Y_test_genus=Y_test[:,1:2]

Y_train_species= Y_train[:,2:]
Y_test_species=Y_test[:,2:]

# print(Y_test_family.shape)


# hyperparameter setup
cv = KFold(10)

# tuned_parameters = [{'C': Cs}]
tuned_parameters = [{}]
Cs = np.linspace(0.01,5,5)
gamma_range = np.logspace(-9, 3, 3)
parameters = {'estimator__kernel':('linear', 'rbf'), 'estimator__C':Cs,'estimator__gamma':gamma_range}
# param_grid = dict(gamma=gamma_range, C=Cs)


# for genus

print("##### Gaussian kernel-SVM for Label:Genus... \n")

model=OneVsRestClassifier(SVC(kernel='rbf',probability=True,tol=0.1))

clf = GridSearchCV(model, param_grid=parameters, cv=cv, refit=True, n_jobs=5)
print('Fitting...')
clf.fit(X_train, Y_train_genus.ravel())

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Best SVM-penalty parameter is ", clf.best_params_)
print("Best Score with this parameter is", clf.best_score_)
sigma=1/(np.sqrt(2*clf.best_params_.get('estimator__gamma')))
print("\nMargin-Width: ",sigma)

print('Predicting...')
preds_gen = clf.predict(X_test)
# print(preds)
test_score_genus = accuracy_score(Y_test_genus, preds_gen)
print("Test Score:", test_score_genus)
test_error_genus = 1 - test_score_genus
hamming_genus=hamming_loss(Y_test_genus, preds_gen)
print('Hamming Loss:',hamming_genus)


Cs = np.linspace(0.01,10,9)
gamma_range = np.logspace(-9, 2, 7)
parameters = {'estimator__kernel':('linear', 'rbf'), 'estimator__C':Cs,'estimator__gamma':gamma_range}



# SVM for family label
print("##### Gaussian kernel-SVM for Label:Family... \n")

model=OneVsRestClassifier(SVC(kernel='rbf',probability=True,tol=0.1))

clf = GridSearchCV(model, param_grid=parameters, cv=cv, refit=True, n_jobs=5)
print('Fitting...')
clf.fit(X_train, Y_train_family.ravel())

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Best SVM-penalty parameter is ", clf.best_params_)
print("Best Score with this parameter is", clf.best_score_)
sigma=1/(np.sqrt(2*clf.best_params_.get('estimator__gamma')))
print("\nMargin-Width: ",sigma)

print('Predicting...')
preds = clf.predict(X_test)
# print(preds)
test_score_family = accuracy_score(Y_test_family, preds)
print("Test Score:", test_score_family)
test_error_family = 1 - test_score_family
hamming_family=hamming_loss(Y_test_family, preds)
print('Hamming Loss:',hamming_family)



Cs = np.linspace(0.01,9,11)
gamma_range = np.logspace(-9, 3, 13)
parameters = {'estimator__kernel':('linear', 'rbf'), 'estimator__C':Cs,'estimator__gamma':gamma_range}

# for species

print("##### Gaussian kernel-SVM for Label:Genus... \n")

model=OneVsRestClassifier(SVC(kernel='rbf',probability=True,tol=0.1))

clf = GridSearchCV(model, param_grid=parameters, cv=cv, refit=True, n_jobs=5)
print('Fitting...')
clf.fit(X_train, Y_train_species.ravel())

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Best SVM-penalty parameter is ", clf.best_params_)
print("Best Score with this parameter is", clf.best_score_)

sigma=1/(np.sqrt(2*clf.best_params_.get('estimator__gamma')))
print("\nMargin-Width: ",sigma)

print('Predicting...')
preds_sp = clf.predict(X_test)
# print(preds)
test_score_species = accuracy_score(Y_test_species, preds_sp)
print("Test Score:", test_score_species)
test_error_species = 1 - test_score_species
hamming_species=hamming_loss(Y_test_species, preds_sp)
print('Hamming Loss:',hamming_species)


correct_classification=0
misclassified=0
for i in range(0,len(Y_test_species)):
    if((preds[i]==Y_test_family[i]) and (preds_sp[i]==Y_test_species[i]) and (preds_gen[i]==Y_test_genus[i])):
        correct_classification+=1
    else:
        misclassified+=1

net_zero_one=misclassified/len(Y_test_genus)
arr=[]
arr.append(hamming_family)
arr.append(hamming_genus)
arr.append(hamming_species)
arr=np.array(arr)
net_hamm=np.mean(arr)
print("\nAverage Hamming Loss:",net_hamm)
print("\nAverage Zero-one Loss:",net_zero_one)