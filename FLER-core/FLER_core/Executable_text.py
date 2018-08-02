
# coding: utf-8

# In[ ]:


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

print('Quel est votre path jusqu au download du github ?')
path = input("prompt")
os.chdir(path)

train=pd.read_csv("RCV1.txt", delim_whitespace=True, skipinitialspace=True, names= ['Word','POStag','CHUNKtag','NEtag'])
train=train.dropna(axis=0, how='any')
train.reset_index(drop=True, inplace=True)
sparse = pd.get_dummies(train['NEtag'])



train['ORG'] = sparse['I-ORG'] + sparse['B-ORG']
train['LOC'] = sparse['I-LOC'] + sparse['B-LOC']
train['MISC'] = sparse['I-MISC'] + sparse['B-MISC']
train['PER'] = sparse['I-PER']


# In[39]:


def capitalize(text):
    cap = []
    for x in text:
        if x[0].isupper():
            cap.append(1)
        else : 
            cap.append(0)
    return cap
train['Cap']=capitalize(train['Word'])
def fullcap(text):
    cap = []
    for x in text:
        if str(x).isupper():
            cap.append(1)
        else : 
            cap.append(0)
    return cap
train['Cap2']=fullcap(train['Word'])
def length(text):
    length = []
    for x in text:
        length.append(len(x))
    return length
train['leng']=length(train['Word'])
def nocaps(text):
    nocaps=[]
    for i in text :
        nocaps.append(str(i).lower())
    return nocaps
train['NoCaps']=nocaps(train['Word'])
prefixe=['anti','co','dis','il','im','in','inter','ir','mis','over','out','post','pre','pro','sub','super','trans','under']
suffixe=['dom','ship','hood','ian','er','er','ism','en','less','ish','ful','al','ly','en','ness','ship','ity','ize','ly']

def presufixe (text):
    presuf=[]
    for x in text :
        b=0
        if len(x)>5 :
            for j in range (0,6):
                if x[0:j] in prefixe :
                    b=1
                
            for j in range(0,5):
                if x[len(x)-j:len(x)] in suffixe :
                    b=1
        presuf.append(b)
    return presuf
train['presuf']=presufixe(train['NoCaps'])
gazloc =nocaps( nltk.corpus.gazetteers.words(fileids=['countries.txt','uscities.txt','usstates.txt','usstateabbrev.txt','mexstates.txt','caprovinces.txt']))

gazper = nocaps(nltk.corpus.names.words(fileids=['male.txt','female.txt']))
gazmisc = nocaps(nltk.corpus.gazetteers.words(fileids=['nationalities.txt']))
L=[gazloc,gazper,gazmisc]
def gazetteer( text  ):
    resloc=[]
    resper=[]
    resmisc=[]
    restot=[resloc, resper, resmisc]
    for i in range(0,3) :
        
        for word in text:
            if word in L[i] :
                restot[i].append(1)
            else :
                restot[i].append(0)
    return restot
gaztot=gazetteer(train['NoCaps'])
train['GAZMISC']=gaztot[2]
train['GAZLOC']=gaztot[0]
train['GAZPER']=gaztot[1]


# In[45]:


Numb=['0','1','2','3','4','5','6','7','8','9']
def numb(text):
    b=0
    l=[]
    for i in text :
        b=0
        for j in Numb:
            if j in i :
                b=1
        l.append(b)
    return l
train['Number']=numb(train['Word'])


# In[40]:


import collections

train=train.dropna(axis=0, how='any')

train['NoCaps']=nocaps(train['Word'])

allname=[]
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'NEtag')!="O":
        allname.append(getattr(row,'NoCaps'))
counter=collections.Counter(allname)
frequentname=[]
for i in allname:
    if counter[i]>=6 and i not in frequentname  :
        frequentname.append(i)
frequentname

allname=[]
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'ORG')!=0:
        allname.append(getattr(row,'NoCaps'))
counter=collections.Counter(allname)

frequentORG=[]
for i in allname:
    if counter[i]>=6 and i not in frequentORG  :
        frequentORG.append(i)
frequentORG     
freq=[]

allname=[]
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'LOC')!=0:
        allname.append(getattr(row,'NoCaps'))
counter=collections.Counter(allname)

frequentLOC=[]
for i in allname:
    if counter[i]>=6 and i not in frequentLOC  :
        frequentLOC.append(i)
        

allname=[]
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'PER')!=0:
        allname.append(getattr(row,'NoCaps'))
counter=collections.Counter(allname)

frequentPER=[]
for i in allname:
    if counter[i]>=6 and i not in frequentPER  :
        frequentPER.append(i)
        
freq=[]

allname=[]
for row in train.itertuples(index=True, name='Pandas'):
    if getattr(row, 'MISC')!=0:
        allname.append(getattr(row,'NoCaps'))
counter=collections.Counter(allname)

frequentMISC=[]
for i in allname:
    if counter[i]>=6 and i not in frequentMISC  :
        frequentMISC.append(i)


# In[41]:


alpha=['ORG','MISC','LOC','PER']
frequentuni=[[],[],[],[]]
frequentpostuni=[[],[],[],[]]
for at in range(0,4):
    uni=[]
    for i in range(1,202386):
        try :
            if train[alpha[at]][i]==1:
                uni.append(train['NoCaps'][i-1])
        except :
            pass
    counter2=collections.Counter(uni)

    for i in uni:
        if counter2[i]>=6 and i not in frequentuni[at]  :
            frequentuni[at].append(i)
for at in range(0,4):
    uni=[]
    for i in range(0,202386-1):
        try :
            if train[alpha[at]][i]==1:
                uni.append(train['NoCaps'][i+1])
        except :
            pass
    counter2=collections.Counter(uni)

    for i in uni:
        if counter2[i]>=6 and i not in frequentpostuni[at]  :
            frequentpostuni[at].append(i)


# In[42]:


def frequence(data):
    freq=[]
    for i in data :
        if i in frequentname :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequenceorg(data):
    freq=[]
    for i in data :
        if i in frequentORG :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequenceloc(data):
    freq=[]
    for i in data :
        if i in frequentLOC :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequenceper(data):
    freq=[]
    for i in data :
        if i in frequentPER :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequencemisc(data):
    freq=[]
    for i in data :
        if i in frequentMISC :
            freq.append(1)
        else :
            freq.append(0)
    return freq


# In[44]:


def preuniorg(data):
    freq=[0]
    for i in range(1,len(data)) :
        if data[i-1] in frequentuni[0]:
            freq.append(0)
        else : 
            freq.append(1)
    return freq
def preunimisc(data):
    freq=[0]
    for i in range(1,len(data)) :
        if data[i-1] in frequentuni[1]:
            freq.append(0)
        else : 
            freq.append(1)
    return freq
def preuniloc(data):
    freq=[0]
    for i in range(1,len(data)) :
        if data[i-1] in frequentuni[2]:
            freq.append(0)
        else : 
            freq.append(1)
    return freq
def preuniper(data):
    freq=[0]
    for i in range(1,len(data)) :
        if data[i-1] in frequentuni[3]:
            freq.append(0)
        else : 
            freq.append(1)
    return freq
def postuniorg(data):
    freq=[]
    for i in range(0,len(data)-1):
        if data[i+1] in frequentpostuni[0]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq
def postunimisc(data):
    freq=[]
    for i in range(0,len(data)-1):
        if data[i+1] in frequentpostuni[1]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq
def postuniloc(data):
    freq=[]
    for i in range(0,len(data)-1):
        if data[i+1] in frequentpostuni[2]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq
def postuniper(data):
    freq=[]
    for i in range(0,len(data)-1):
        if data[i+1] in frequentpostuni[3]:
            freq.append(1)
        else:
            freq.append(0)
    freq.append(0)
    return freq


# In[ ]:


train['Freq']=frequence(train['NoCaps'])
train['Freqorg']=frequenceorg(train['NoCaps'])
train['Freqloc']=frequenceloc(train['NoCaps'])
train['Freqper']=frequenceper(train['NoCaps'])
train['Freqmisc']=frequencemisc(train['NoCaps'])
train['UNIORG']=preuniorg(train['NoCaps'])
train['UNIPER']=preuniper(train['NoCaps'])
train['UNILOC']=preuniloc(train['NoCaps'])
train['UNIMISC']=preunimisc(train['NoCaps'])
train['POSTUNIMISC']=postunimisc(train['NoCaps'])
train['POSTUNIORG']=postuniorg(train['NoCaps'])
train['POSTUNILOC']=postuniloc(train['NoCaps'])
train['POSTUNIPER']=postuniper(train['NoCaps'])


# In[46]:


def preprocess2(texte):
    word = []
    b=re.findall(r"[\w']+|[.,!?;]",texte)
    data=pd.DataFrame()
    for x in b:
        word.append(x)
    data['Word'] = word
    data['NoCaps']=nocaps(word)
    data['Cap']=capitalize(word)
    data['presuf']=presufixe(word)
    data['Cap2']=fullcap(word)
    data['Number']=numb(word)
    data['Freq']=frequence(word)
    data['leng'] = length(word)
    data['GAZMISC'] = gazetteer( data['NoCaps'])[2]
    data['GAZPER'] = gazetteer(data['NoCaps'])[1]
    data['GAZLOC'] = gazetteer(data['NoCaps'] )[0]
    data['Freqorg']=frequenceorg(data['NoCaps'] )
    data['Freqloc']=frequenceloc(data['NoCaps'] )
    data['Freqper']=frequenceper(data['NoCaps'] )
    data['Freqmisc']=frequencemisc(data['NoCaps'] )
    data['UNIORG']=preuniorg(data['NoCaps'] )
    data['UNIPER']=preuniper(data['NoCaps'] )
    data['UNILOC']=preuniloc(data['NoCaps'] )
    data['UNIMISC']=preunimisc(data['NoCaps'] )
    data['POSTUNIMISC']=postunimisc(data['NoCaps'] )
    data['POSTUNIORG']=postuniorg(data['NoCaps'] )
    data['POSTUNILOC']=postuniloc(data['NoCaps'] )
    data['POSTUNIPER']=postuniper(data['NoCaps'] )

    return data

Liste_feature=['Cap', 'presuf', 'Freq', 'Freqorg', 'Freqloc', 'Freqper', 'Freqmisc',
       'Number', 'UNIORG', 'UNIMISC', 'UNILOC', 'UNIPER', 'POSTUNIORG',
       'POSTUNIMISC', 'POSTUNILOC', 'POSTUNIPER', 'GAZMISC', 'GAZPER',
       'GAZLOC','leng']

def total_multi (texte):
    df=preprocess2(texte)
    a=model.predict(df[Liste_feature])
    orgas=[]
    lieux=[]
    per=[]
    misc=[]
    L=[orgas,lieux,per,misc]
    for i in range (0,len(a)) :
        if a[i] ==1: 
            orgas.append(df['Word'][i])
        elif a[i]==2 :
            lieux.append(df['Word'][i])
        elif a[i]==3 :
            misc.append(df['Word'][i])
        elif a[i]==4:
            per.append(df['Word'][i])
    print('Les orgas sont')
    print(orgas)
    print('Les lieux sont')
    print(lieux)
    print('Les personnes sont')
    print(per)
    print('Les misc sont')
    print(misc)
    


# In[ ]:





# In[2]:
exit_system=0
while exit_system==0 :
    
    
    print('Quel modèle voulez vous utiliser  ? (utiliser le nom d un modèle sur le git sans le .pkl)')
    modele = input("prompt")
    model=joblib.load('Modele_test/'+modele+'.pkl')
    print('Quel est le path jusqu a votre fichier texte ? (ne pas oublier le .txt)')

    TEXT = input("prompt")


    # In[56]:

    with open(TEXT, 'r') as myfile:
        test=myfile.read().replace('\n', '').replace('"','').replace("'",'')



    total_multi(test)

