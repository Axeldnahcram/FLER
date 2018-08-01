import time
import nltk
import pandas as pd
import numpy as np
import collections

gazloc = nltk.corpus.gazetteers.words(fileids=['countries.txt','uscities.txt','usstates.txt','usstateabbrev.txt','mexstates.txt','caprovinces.txt'])
gazper = nltk.corpus.names.words(fileids=['male.txt','female.txt'])
gazmisc = nltk.corpus.gazetteers.words(fileids=['nationalities.txt'])

prefixe=['anti','co','dis','il','im','in','inter','ir','mis','over','out','post','pre','pro','sub','super','trans','under']
suffixe=['dom','ship','hood','ian','er','er','ism','en','less','ish','ful','al','ly','en','ness','ship','ity','ize','ly']
Numb=['0','1','2','3','4','5','6','7','8','9']
alpha=['ORG','MISC','LOC','PER']


def nub(x):
    if x==0:
        return 0
    for i in Numb :
        if i in x :
            return 1
    return 0

def numb(data):
    nube=[]
    for row in data.itertuples(index=True, name='Pandas'):
        nube.append(nub(getattr(row, "Word").lower()))
    return  nube


def presufixe (x):
    if x == 0 :
        return 0
    if len(x)>5 :
        for j in range (0,6):
            if x[0:j] in prefixe :
                return 1
        for j in range(0,5):
            if x[len(x)-j:len(x)] in suffixe :
                return 1
    return 0

def presufixe2(text):
    t=[]
    for i in text:
        t.append(presufixe(i))
    return t

def frequence(data):
    allname=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row, 'NEtag')!="O":
            allname.append(getattr(row,'Word').lower())
    counter=collections.Counter(allname)
    frequentname=[]
    for i in allname:
        if counter[i]>=6 and i not in frequentname  :
            frequentname.append(i)
    freq=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row,"Word").lower() in frequentname :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequenceorg(data):
    allname=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row, 'ORG')!=0:
            allname.append(getattr(row,'Word').lower())
    counter=collections.Counter(allname)

    frequentname=[]
    for i in allname:
        if counter[i]>=6 and i not in frequentname  :
            frequentname.append(i)
        
    freq=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row,"Word").lower() in frequentname :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequenceloc(data):
    allname=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row, 'LOC')!=0:
            allname.append(getattr(row,'Word').lower())
    counter=collections.Counter(allname)

    frequentname=[]
    for i in allname:
        if counter[i]>=6 and i not in frequentname  :
            frequentname.append(i)
        
    freq=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row,"Word").lower() in frequentname :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequenceper(data):
    allname=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row, 'PER')!=0:
            allname.append(getattr(row,'Word').lower())
    counter=collections.Counter(allname)

    frequentname=[]
    for i in allname:
        if counter[i]>=6 and i not in frequentname  :
            frequentname.append(i)
        
    freq=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row,"Word").lower() in frequentname :
            freq.append(1)
        else :
            freq.append(0)
    return freq

def frequencemisc(data):
    allname=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row, 'MISC')!=0:
            allname.append(getattr(row,'Word').lower())
    counter=collections.Counter(allname)

    frequentname=[]
    for i in allname:
        if counter[i]>=6 and i not in frequentname  :
            frequentname.append(i)
        
    freq=[]
    for row in data.itertuples(index=True, name='Pandas'):
        if getattr(row,"Word").lower() in frequentname :
            freq.append(1)
        else :
            freq.append(0)
    return freq


def NE(text):
    sparse = pd.get_dummies(text['NEtag'])
    text['ORG'] = sparse['I-ORG'] + sparse['B-ORG']
    text['LOC'] = sparse['I-LOC'] + sparse['B-LOC']
    text['MISC'] = sparse['I-MISC'] + sparse['B-MISC']
    text['PER'] = sparse['I-PER']
    return text

def BIOpostag(text):
    POStag = text['POStag'].tolist()
    BPOStag = POStag
    OPOStag = POStag
    m = len(text['Word'])-1
    for i in range(1,m+1):
        BPOStag[i]=text['POStag'][i-1]
    text['BPOStag'] = BPOStag
    for i in range(0,m):
        OPOStag[i]=text['POStag'][i+1]
    text['OPOStag'] = OPOStag
    return text

def capitalize(text):
    cap = []
    for x in text:
        if x[0].isupper():
            cap.append(1)
        else : 
            cap.append(0)
    return cap

def fullcap(text):
    cap = []
    for x in text:
        if x.isupper():
            cap.append(1)
        else : 
            cap.append(0)
    return cap

def length(text):
    length = []
    for x in text:
        length.append(len(x))
    return length

def gazetteer( text , ref ):
    res = []
    for word in text:
        if word in ref :
            res.append(1)
        else :
            res.append(0)
    return res

def preprocess(data):
    word = []
    for x in data['Word']:
        word.append(str(x))
    data['Word'] = word
    data = BIOpostag(data)
    data['Cap']=capitalize(word)
    data['presuf']=presufixe2(word)
    data['Cap2']=fullcap(word)
    data['Number']=numb(data)
    data['Freq']=frequence(data)
    data['leng'] = length(word)
    data['GAZMISC'] = gazetteer(word , gazmisc)
    data['GAZPER'] = gazetteer(word, gazper)
    data['GAZLOC'] = gazetteer(word , gazloc)
    data = NE(data)
    data['Freqorg']=frequenceorg(data)
    data['Freqloc']=frequenceloc(data)
    data['Freqper']=frequenceper(data)
    data['Freqmisc']=frequencemisc(data)
    for at in alpha:
        uni=[]
        for i,row in enumerate(data.itertuples(index=True, name='Pandas'),1):
            if getattr(row,at)==1 :
                uni.append(data.iloc[i-2,16])
        counter2=collections.Counter(uni)

        frequentuni=[]
        for i in uni:
            if counter2[i]>=6 and i not in frequentuni  :
                frequentuni.append(i)
        L=[0]
        for i in range (1,len(data)):
            if data.iloc[i-1,16] in frequentuni :
                L.append(1)
            else :
                L.append(0)
        data['UNI'+at]=L
    for at in alpha:
        postuni=[]
        for i,row in enumerate(data.itertuples(index=True, name='Pandas'),1):
            if getattr(row,at)==1 :
                postuni.append(data.iloc[i,16])
        counter2=collections.Counter(postuni)

        frequentuni=[]
        for i in postuni:
            if counter2[i]>=6 and i not in frequentuni  :
                frequentuni.append(i)
        L=[]
        for i in range (0,len(data)-1):
            if data.iloc[i+1,16] in frequentuni :
                L.append(1)
            else :
                L.append(0)
        L.append(0)
        data['POSTUNI'+at]=L

    return data

name = input("Path of the db? ")
x = pd.read_csv(name, delim_whitespace=True, skipinitialspace=True, names= ['Word','POStag','CHUNKtag','NEtag'])

y = preprocess(x)
y.to_csv('result.csv')