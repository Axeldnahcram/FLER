
# coding: utf-8

# In[136]:

import pandas as pd
import numpy as np


# In[137]:

path='/Users/antoinepradier/Desktop/STATAPP/donnees_FR/dependencies/'


# In[138]:

data=pd.read_csv(path+'flmf3_01000_01499ep.aa.conll', sep='\t', header=None, names=['1','word','none','type','p_type','PRO.1','sentid=flmf3_01000_01499ep-1000|g=m|n=p|s=card','10','suj','10.1','suj.1'])


# In[139]:

data.head()


# In[151]:

liste_fichiers=['flmf3_01000_01499ep.aa.conll','flmf3_03500_03999ep.aa.conll','flmf3_08000_08499ep.xd.cat.conll','flmf3_08500_08999ep.aa.conll','flmf3_09000_09499ep.aa.conll','flmf3_10000_10499ep.aa.conll','flmf3_11000_11499ep.aa.conll','flmf3_12000_12499ep.aa.conll','flmf3_12500_12999co.aa.conll','flmf7aa1ep.cat.conll','flmf7aa2ep.cat.conll','flmf7ab1co.aa.conll','flmf7ab2ep.conll','flmf7ad1co.aa.conll','flmf7ae1ep.cat.conll','flmf7af1ep.conll','flmf7af2ep.cat.conll','flmf7ag1exp.cat.conll','flmf7ag2ep.cat.conll','flmf7ah1ep.aa.conll','flmf7ah2ep.aa.conll','flmf7ai1exp.cat.conll','flmf7ai2ep.aa.cat.conll','flmf7aj1ep.indent.conll','flmf7aj2.xml.aa.cat.conll','flmf7ak1ep.indent.conll','flmf7ak2ep.xd.cat.conll','flmf7al1ep.cat.conll','flmf7al2ep.cat.conll','flmf7am1ep.xd.cat.conll','flmf7am2ep.xd.cat.conll','flmf7an1ep.conll','flmf7an2co.af.cat.conll','flmf7ao1ep.conll','flmf7ao2ep.conll','flmf7ap1ep.af.cat.conll','flmf7ap2ep.conll','flmf7aq1ep.conll','flmf7aq2ep.xd.cat.conll','flmf7ar2.ep.aa.cat.conll','flmf7as1ep.cat.conll','flmf7as2ep.af.cat.conll','flmf7atep.cat.conll','flmf300_13000ep.cat.conll']


# In[153]:

liste_df=[]
for f in liste_fichiers:
    data=pd.read_csv(path+f, sep='\t', header=None, names=['1','word','none','type','p_type','PRO.1','sentid=flmf3_01000_01499ep-1000|g=m|n=p|s=card','10','suj','10.1','suj.1'])
    del data['PRO.1']
    del data['none']
    del data['sentid=flmf3_01000_01499ep-1000|g=m|n=p|s=card']
    del data['10']
    del data['suj']
    del data['10.1']
    del data['suj.1']
    del data['1']
    liste_df.append(data)


# In[163]:

data=pd.concat(liste_df)


# In[171]:

path2='/Users/antoinepradier/Desktop/STATAPP/donnees_FR/francais.csv'


# In[172]:

data.to_csv(path_or_buf=path2)

