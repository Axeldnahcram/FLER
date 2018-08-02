

import time
import nltk
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
import sys
import re
from sklearn.externals import joblib

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

train = pd.read_csv("C:/Users/user/Documents/Projet Statap/Statapp-master/RCV1.txt", delim_whitespace=True,
                    skipinitialspace=True, names=['Word', 'POStag', 'CHUNKtag', 'NEtag'])
train = train.dropna(axis=0, how='any')
train.reset_index(drop=True, inplace=True)
sparse = pd.get_dummies(train['NEtag'])

train['ORG'] = sparse['I-ORG'] + sparse['B-ORG']
train['LOC'] = sparse['I-LOC'] + sparse['B-LOC']
train['MISC'] = sparse['I-MISC'] + sparse['B-MISC']
train['PER'] = sparse['I-PER']
multi = []
for i in range(0, len(test)):
    if train['ORG'][i] == 1:
        multi.append(1)
    elif train['LOC'][i] == 1:
        multi.append(2)
    elif train['MISC'][i] == 1:
        multi.append(3)
    elif train['PER'][i] == 1:
        multi.append(4)
    else:
        multi.append(0)
train['multi'] = multi



# On
# importe
# la
# base
# d
# 'entrainement et on crée les indicatrices d'
# entités
# nommées



test = pd.read_csv("C:/Users/user/Documents/Projet Statap/Statapp-master/testa.txt", delim_whitespace=True,
                   skipinitialspace=True, names=['Word', 'POStag', 'CHUNKtag', 'NEtag'])
test = test.dropna(axis=0, how='any')
test.reset_index(drop=True, inplace=True)
sparse = pd.get_dummies(test['NEtag'])

test['ORG'] = sparse['I-ORG']
test['LOC'] = sparse['I-LOC']
test['MISC'] = sparse['I-MISC'] + sparse['B-MISC']
test['PER'] = sparse['I-PER']






# On
# fait
# la
# même
# chose
# sur
# notre
# base
# de
# test.
#
# \subsubsection * {Fonctions
# features}
#
# On
# a
# ensuite
# défini
# les
# différentes
# features
#
# \subsubsection * {Features
# orthographiques}



def capitalize(text):
    cap = []
    for x in text:
        if x[0].isupper():
            cap.append(1)
        else:
            cap.append(0)
    return cap


train['Cap'] = capitalize(train['Word'])


def fullcap(text):
    cap = []
    for x in text:
        if str(x).isupper():
            cap.append(1)
        else:
            cap.append(0)
    return cap


train['Cap2'] = fullcap(train['Word'])


def length(text):
    length = []
    for x in text:
        length.append(len(x))
    return length


train['leng'] = length(train['Word'])


def nocaps(text):
    nocaps = []
    for i in text:
        nocaps.append(str(i).lower())
    return nocaps


train['NoCaps'] = nocaps(train['Word'])
prefixe = ['anti', 'co', 'dis', 'il', 'im', 'in', 'inter', 'ir', 'mis', 'over', 'out', 'post', 'pre', 'pro', 'sub',
           'super', 'trans', 'under']
suffixe = ['dom', 'ship', 'hood', 'ian', 'er', 'er', 'ism', 'en', 'less', 'ish', 'ful', 'al', 'ly', 'en', 'ness',
           'ship', 'ity', 'ize', 'ly']


def presufixe(text):
    presuf = []
    for x in text:
        b = 0
        if len(x) > 5:
            for j in range(0, 6):
                if x[0:j] in prefixe:
                    b = 1

            for j in range(0, 5):
                if x[len(x) - j:len(x)] in suffixe:
                    b = 1
        presuf.append(b)
    return presuf


train['presuf'] = presufixe(train['NoCaps'])
Numb = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def numb(text):
    b = 0
    l = []
    for i in text:
        b = 0
        for j in Numb:
            if j in i:
                b = 1
        l.append(b)
    return l


train['Number'] = numb(train['Word'])






# On
# applique
# directement
# ces
# fonctions
# à
# notre
# base
# d
# 'entrainement.
#
# \subsubsection * {Gazetteer}


gazloc = nocaps(nltk.corpus.gazetteers.words(
    fileids=['countries.txt', 'uscities.txt', 'usstates.txt', 'usstateabbrev.txt', 'mexstates.txt', 'caprovinces.txt']))

gazper = nocaps(nltk.corpus.names.words(fileids=['male.txt', 'female.txt']))
gazmisc = nocaps(nltk.corpus.gazetteers.words(fileids=['nationalities.txt']))
L = [gazloc, gazper, gazmisc]


def gazetteer(text):
    resloc = []
    resper = []
    resmisc = []
    restot = [resloc, resper, resmisc]
    for i in range(0, 3):

        for word in text:
            if word in L[i]:
                restot[i].append(1)
            else:
                restot[i].append(0)
    return restot


gaztot = gazetteer(train['NoCaps'])
train['GAZMISC'] = gaztot[2]
train['GAZLOC'] = gaztot[0]
train['GAZPER'] = gaztot[1]



# \subsubsection * {Feature
# de
# fréquence
# et
# unigrammes}
#
# On
# définit
# dans
# un
# premier
# temps
# les
# listes
# de
# noms
# les
# plus
# fréquents, puis
# les
# préunigrammes
# et
# les
# post
# unigrammes
# les
# plus
# fréquents.



import collections

allname = []
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'NEtag') != "O":
        allname.append(getattr(row, 'NoCaps'))
counter = collections.Counter(allname)
frequentname = []
for i in allname:
    if counter[i] >= 6 and i not in frequentname:
        frequentname.append(i)
frequentname

allname = []
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'ORG') != 0:
        allname.append(getattr(row, 'NoCaps'))
counter = collections.Counter(allname)

frequentORG = []
for i in allname:
    if counter[i] >= 6 and i not in frequentORG:
        frequentORG.append(i)
frequentORG
freq = []

allname = []
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'LOC') != 0:
        allname.append(getattr(row, 'NoCaps'))
counter = collections.Counter(allname)

frequentLOC = []
for i in allname:
    if counter[i] >= 6 and i not in frequentLOC:
        frequentLOC.append(i)

allname = []
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'PER') != 0:
        allname.append(getattr(row, 'NoCaps'))
counter = collections.Counter(allname)

frequentPER = []
for i in allname:
    if counter[i] >= 6 and i not in frequentPER:
        frequentPER.append(i)

freq = []

allname = []
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'MISC') != 0:
        allname.append(getattr(row, 'NoCaps'))
counter = collections.Counter(allname)

frequentMISC = []
for i in allname:
    if counter[i] >= 6 and i not in frequentMISC:
        frequentMISC.append(i)

# In[41]:


alpha = ['ORG', 'MISC', 'LOC', 'PER']
frequentuni = [[], [], [], []]
frequentpostuni = [[], [], [], []]
for at in range(0, 4):
    uni = []
    for i in range(1, 202386):
        try:
            if train[alpha[at]][i] == 1:
                uni.append(train['NoCaps'][i - 1])
        except:
            pass
    counter2 = collections.Counter(uni)

    for i in uni:
        if counter2[i] >= 6 and i not in frequentuni[at]:
            frequentuni[at].append(i)
for at in range(0, 4):
    uni = []
    for i in range(0, 202386 - 1):
        try:
            if train[alpha[at]][i] == 1:
                uni.append(train['NoCaps'][i + 1])
        except:
            pass
    counter2 = collections.Counter(uni)

    for i in uni:
        if counter2[i] >= 6 and i not in frequentpostuni[at]:
            frequentpostuni[at].append(i)



# On
# enlève
# ensuite
# les
# éléments
# appartenant
# à
# plusieurs
# listes
# différentes.



frequentuni0 = [x for x in frequentuni[0] if x not in frequentuni[1]]
frequentuni00 = [x for x in frequentuni0 if x not in frequentuni[2]]
frequentuni000 = [x for x in frequentuni00 if x not in frequentuni[3]]
frequentuni1 = [x for x in frequentuni[1] if x not in frequentuni[0]]
frequentuni10 = [x for x in frequentuni1 if x not in frequentuni[2]]
frequentuni100 = [x for x in frequentuni10 if x not in frequentuni[3]]
frequentuni2 = [x for x in frequentuni[2] if x not in frequentuni[0]]
frequentuni20 = [x for x in frequentuni2 if x not in frequentuni[1]]
frequentuni200 = [x for x in frequentuni20 if x not in frequentuni[3]]
frequentuni3 = [x for x in frequentuni[3] if x not in frequentuni[0]]
frequentuni30 = [x for x in frequentuni3 if x not in frequentuni[1]]
frequentuni300 = [x for x in frequentuni30 if x not in frequentuni[2]]
frequentuni = [frequentuni000, frequentuni100, frequentuni200, frequentuni300]

frequentpuni0 = [x for x in frequentpostuni[0] if x not in frequentpostuni[1]]
frequentpuni00 = [x for x in frequentpuni0 if x not in frequentpostuni[2]]
frequentpuni000 = [x for x in frequentpuni00 if x not in frequentpostuni[3]]
frequentpuni1 = [x for x in frequentpostuni[1] if x not in frequentpostuni[0]]
frequentpuni10 = [x for x in frequentpuni1 if x not in frequentpostuni[2]]
frequentpuni100 = [x for x in frequentpuni10 if x not in frequentpostuni[3]]
frequentpuni2 = [x for x in frequentpostuni[2] if x not in frequentpostuni[0]]
frequentpuni20 = [x for x in frequentpuni2 if x not in frequentpostuni[1]]
frequentpuni200 = [x for x in frequentpuni20 if x not in frequentpostuni[3]]
frequentpuni3 = [x for x in frequentpostuni[3] if x not in frequentpostuni[0]]
frequentpuni30 = [x for x in frequentpuni3 if x not in frequentpostuni[1]]
frequentpuni300 = [x for x in frequentpuni30 if x not in frequentpostuni[2]]
frequentpostuni = [frequentpuni000, frequentpuni100, frequentpuni200, frequentpuni300]




# On
# crée
# les
# différentes
# fonctions
# correspondant
# aux
# features.




def frequence(data):
    freq = []
    for i in data:
        if i in frequentname:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def frequenceorg(data):
    freq = []
    for i in data:
        if i in frequentORG:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def frequenceloc(data):
    freq = []
    for i in data:
        if i in frequentLOC:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def frequenceper(data):
    freq = []
    for i in data:
        if i in frequentPER:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def frequencemisc(data):
    freq = []
    for i in data:
        if i in frequentMISC:
            freq.append(1)
        else:
            freq.append(0)
    return freq


# In[44]:


def preuniorg(data):
    freq = [0]
    for i in range(1, len(data)):
        if data[i - 1] in frequentuni[0]:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def preunimisc(data):
    freq = [0]
    for i in range(1, len(data)):
        if data[i - 1] in frequentuni[1]:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def preuniloc(data):
    freq = [0]
    for i in range(1, len(data)):
        if data[i - 1] in frequentuni[2]:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def preuniper(data):
    freq = [0]
    for i in range(1, len(data)):
        if data[i - 1] in frequentuni[3]:
            freq.append(1)
        else:
            freq.append(0)
    return freq


def postuniorg(data):
    freq = []
    for i in range(0, len(data) - 1):
        if data[i + 1] in frequentpostuni[0]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq


def postunimisc(data):
    freq = []
    for i in range(0, len(data) - 1):
        if data[i + 1] in frequentpostuni[1]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq


def postuniloc(data):
    freq = []
    for i in range(0, len(data) - 1):
        if data[i + 1] in frequentpostuni[2]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq


def postuniper(data):
    freq = []
    for i in range(0, len(data) - 1):
        if data[i + 1] in frequentpostuni[3]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq



# \subsubsection * {Finalisation
# de
# la
# base
# test}
#
# On
# applique
# les
# fonctions
# précédentes
# à
# notre
# base
# train.



train['Freq'] = frequence(train['NoCaps'])
train['Freqorg'] = frequenceorg(train['NoCaps'])
train['Freqloc'] = frequenceloc(train['NoCaps'])
train['Freqper'] = frequenceper(train['NoCaps'])
train['Freqmisc'] = frequencemisc(train['NoCaps'])
train['UNIORG'] = preuniorg(train['NoCaps'])
train['UNIPER'] = preuniper(train['NoCaps'])
train['UNILOC'] = preuniloc(train['NoCaps'])
train['UNIMISC'] = preunimisc(train['NoCaps'])
train['POSTUNIMISC'] = postunimisc(train['NoCaps'])
train['POSTUNIORG'] = postuniorg(train['NoCaps'])
train['POSTUNILOC'] = postuniloc(train['NoCaps'])
train['POSTUNIPER'] = postuniper(train['NoCaps'])



# \subsubsection * {Fonction
# preprocess}
#
# La
# fonction
# preprocess
# est
# la
# fonction
# la
# plus
# importante, elle
# prend
# en
# input
# un
# texte
# est
# retourne
# un
# dataframe
# contenant
# les
# bonnes
# features
# afin
# de
# pouvoir
# appliquer
# les
# différents
# algorithmes
# de
# machine
# learning.




def preprocess(texte):
    word = []
    b = re.findall(r"[\w']+|[.,!?;]", texte)
    data = pd.DataFrame()
    for x in b:
        word.append(x)
    data['Word'] = word
    data['NoCaps'] = nocaps(word)
    data['Cap'] = capitalize(word)
    data['presuf'] = presufixe(word)
    data['Cap2'] = fullcap(word)
    data['Number'] = numb(word)
    data['Freq'] = frequence(word)
    data['leng'] = length(word)
    data['GAZMISC'] = gazetteer(data['NoCaps'])[2]
    data['GAZPER'] = gazetteer(data['NoCaps'])[1]
    data['GAZLOC'] = gazetteer(data['NoCaps'])[0]
    data['Freqorg'] = frequenceorg(data['NoCaps'])
    data['Freqloc'] = frequenceloc(data['NoCaps'])
    data['Freqper'] = frequenceper(data['NoCaps'])
    data['Freqmisc'] = frequencemisc(data['NoCaps'])
    data['UNIORG'] = preuniorg(data['NoCaps'])
    data['UNIPER'] = preuniper(data['NoCaps'])
    data['UNILOC'] = preuniloc(data['NoCaps'])
    data['UNIMISC'] = preunimisc(data['NoCaps'])
    data['POSTUNIMISC'] = postunimisc(data['NoCaps'])
    data['POSTUNIORG'] = postuniorg(data['NoCaps'])
    data['POSTUNILOC'] = postuniloc(data['NoCaps'])
    data['POSTUNIPER'] = postuniper(data['NoCaps'])

    return data




# Nous
# avons
# également
# défini
# une
# fonction
# preprocess\_test
# qui
# renvoie
# un
# dataframe
# de
# notre
# base
# test.
#
# \subsubsection * {Algorithmes
# de
# machine
# learning}
#
# On
# applique
# ensuite
# les
# différents
# algorithmes
# utilisés.Nous
# n
# 'allons pas ici, par soucis de place, détailler l'
# intégralité
# des
# algorithmes, mais
# représenter
# une
# régression
# logistique, un
# SVM
# linéaire
# et
# à
# noyau
# gaussien, et
# un
# RandomForestClassifier.
#
# \subsubsection * {Régression
# logit}


Liste_feature = ['Cap', 'presuf', 'Freq', 'Freqorg', 'Freqloc', 'Freqper',
                 'Freqmisc', 'Number', 'UNIORG', 'UNIMISC', 'UNILOC', 'UNIPER',
                 'POSTUNIORG', 'POSTUNIMISC', 'POSTUNILOC', 'POSTUNIPER', 'GAZMISC',
                 'GAZPER', 'GAZLOC', 'leng']
yPER = ['PER']
logreg = LogisticRegression()
rfe = RFE(logreg, 2)
rfe = rfe.fit(train[Liste], train[yPER].values.ravel())
print(rfe.support_)
print(rfe.ranking_)
YPER = train['PER']
XPER = train[Liste_feature]
logit_modelPER = sm.Logit(YPER, XPER)
resultPER = logit_modelPER.fit()
print(resultPER.summary())
XPER_train, XPER_test, YPER_train, YPER_test = train_test_split(XPER, YPER,
    test_size = 0.3, random_state = 0)
logregPER = LogisticRegression()
logregPER.fit(XPER_train, YPER_train)
YPER_pred = logregPER.predict(XPER_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregPER.score(XPER_test, YPER_test)))
metrics.confusion_matrix(YPER_test, YPER_pred, labels=None, sample_weight=None)



# \subsubsection * {SVM
# linéaire
# et
# à
# noyau
# gaussien}
#
# Pour
# un
# SVM
# linéaire:



from sklearn.metrics import classification_report
from sklearn import svm

lin_clf = svm.LinearSVC()
lin_clf.fit(train[Liste_feature], train['multi'])

ypred = lin_clf.predict(preprocess_test(test)[Liste_feature])
report = classification_report(y_pred=ypred, y_true=test['multi'])
print(report)




# Pour
# un
# SVM
# gaussien



C = 1.0
gauss_svc = svm.SVC(kernel='rbf', C=C).fit(train[Liste_feature], train['multi'])

ypred = gauss_svc.predict(preprocess_test(test)[Liste_feature])
report = classification_report(y_pred=ypred, y_true=test['multi'])
print(report)




# \subsubsection * {RandomForestClassifier}


from sklearn.ensemble import RandomForestClassifier

random = RandomForestClassifier(n_estimators=20)
random.fit(train[Liste_feature], train['multi'])


# \subsubsection * {Cas
# du
# français}
#
# On
# a
# du
# définir
# une
# nouvelle
# feature
# Debut
# et
# incorporer
# les
# nouvelles
# gazetteers.\ \
#     Feature
# Debut:




def debut(data):
    freq = [0]
    for i in range(1, len(data)):
        if data[i - 1] == '.':
            freq.append(1)
        else:
            freq.append(0)
    return freq




# La
# nouvelle
# fonction
# gazetteer:



gazloc = pd.read_csv('C:/Users/user/Documents/Projet Statap/gazLOC.csv', names=['LOC'])
L = []
L = [x for x in gazloc['LOC'] if x not in L]

gazper = pd.read_csv('C:/Users/user/Documents/Projet Statap/gazPER.csv', names=['PER'])
L2 = []
L2 = [x for x in gazper['PER'] if x not in L2]

L3 = [gazloc, gazper]


def gazetteer(text):
    resloc = []
    resper = []
    restot = [resloc, resper]
    for i in range(0, 2):

        for word in text:
            if word in L[i]:
                restot[i].append(1)
            else:
                restot[i].append(0)
    return restot


gaztot = gazetteer(train['NoCaps'])
train['GAZLOC'] = gaztot[0]
train['GAZPER'] = gaztot[1]




# \subsubsection * {Executable
# Facile}
#
# Nous
# avons
# également
# créé
# un
# outil
# qui, pour
# un
# texte
# donné
# en
# input, renvoie
# les
# différentes
# entités
# nommées
# selon
# leur
# catégorie:




def total_multi(texte, modele):
    df = preprocess(texte)
    a = modele.predict(df[Liste_feature])
    orgas = []
    lieux = []
    per = []
    misc = []
    L = [orgas, lieux, per, misc]
    for i in range(0, len(a)):
        if a[i] == 1:
            orgas.append(df['Word'][i])
        elif a[i] == 2:
            lieux.append(df['Word'][i])
        elif a[i] == 3:
            misc.append(df['Word'][i])
        elif a[i] == 4:
            per.append(df['Word'][i])
    print('Les orgas sont')
    print(orgas)
    print('Les lieux sont')
    print(lieux)
    print('Les personnes sont')
    print(per)
    print('Les misc sont')
    print(misc)