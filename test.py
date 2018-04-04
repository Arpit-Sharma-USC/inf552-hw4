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


# dataframe = pd.read_csv("drive/app/Frogs_MFCCs.csv")
dataframe = pd.read_csv("/media/ghost/Games And Images/INF552/hw4/Anuran Calls (MFCCs)/Frogs_MFCCs.csv")

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


dataframe.to_csv('/media/ghost/Games And Images/INF552/hw4/Frogs_MFCCs-changed.csv',index=False)
# dataframe.to_csv('drive/app/Frogs_MFCCs-changed.csv',index=False)
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
Y_test_genus=Y_train[:,1:2]

Y_train_species= Y_train[:,2:]
Y_test_species=Y_train[:,2:]

