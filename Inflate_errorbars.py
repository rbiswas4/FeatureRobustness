
# coding: utf-8

# In[1]:

from __future__ import division
from snmachine import sndata, snfeatures, snclassifier, tsne_plot
import numpy as np
import matplotlib.pyplot as plt
import time, os, pywt,subprocess
from sklearn.decomposition import PCA
from astropy.table import Table,join,vstack
from astropy.io import fits
import sklearn.metrics 
from functools import partial
from multiprocessing import Pool
import sncosmo

from astropy.table import Column
# WARNING...
#Multinest uses a hardcoded character limit for the output file names. I believe it's a limit of 100 characters
#so avoid making this file path to lengthy if using nested sampling or multinest output file names will be truncated

#Change outdir to somewhere on your computer if you like
dataset='spcc'
outdir=os.path.join('output_%s_no_z' %dataset,'')
out_features=os.path.join(outdir,'features') #Where we save the extracted features to
out_class=os.path.join(outdir,'classifications') #Where we save the classification probabilities and ROC curves
out_int=os.path.join(outdir,'int') #Any intermediate files (such as multinest chains or GP fits)

subprocess.call(['mkdir',outdir])
subprocess.call(['mkdir',out_features])
subprocess.call(['mkdir',out_class])
subprocess.call(['mkdir',out_int])

read_from_file=False #True #We can use this flag to quickly rerun from saved features
run_name=os.path.join(out_features,'%s_all' %dataset)
rt=os.path.join('SPCC_SUBSET','')


# In[ ]:




# In[2]:

def make_new_data(filename, table, output, sntype):
    with open(filename) as f:
        data = f.read().split("\n")
    filters = data[5].split()
    survey = data[0].split()
    stuffRA = data[6].split()
    stuffDec = data[7].split()
    MWEBV = data[11].split()
    if sntype==1:
        typestring='SN Type = Ia , MODEL = mlcs2k2.SNchallenge'
    elif sntype==2:
        typestring='SN Type = II , MODEL = SDSS-017564'
    elif sntype==3:
        typestring='SN Type = Ic , MODEL = SDSS-014475'
    else:
        typestring='SOMETHING WENT HORRIBLY WRONG'
    table.meta = {survey[0][:-1]: survey[1], stuffRA[0][:-1]: stuffRA[1], stuffDec[0][:-1]: stuffDec[1],filters[0][:-1]: filters[1],
                 MWEBV[0][:-1]: MWEBV[1], 'SNTYPE': -9, 'SIM_COMMENT': typestring  }
    #table.rename_column('mjd', 'MJD')
    #table.rename_column('filter ', 'FLT')
    #table.rename_column('flux', 'FLUXCAL')
    #table.rename_column('flux_error', 'FLUXCALERR')
    sncosmo.write_lc(table, 'new_mocks/%s'%output,pedantic=True, format = 'snana')


# In[3]:

prototypes_Ia=[ 'DES_SN002542.DAT', 'DES_SN013866.DAT', 'DES_SN023940.DAT', 'DES_SN024734.DAT', 'DES_SN030701.DAT', 'DES_SN045040.DAT']
prototypes_II=[ 'DES_SN002457.DAT', 'DES_SN005519.DAT', 'DES_SN006381.DAT', 'DES_SN008569.DAT', 'DES_SN013360.DAT', 'DES_SN013481.DAT']
prototypes_Ibc=['DES_SN005399.DAT', 'DES_SN013863.DAT', 'DES_SN027266.DAT', 'DES_SN030183.DAT', 'DES_SN065493.DAT', 'DES_SN078241.DAT']
len(prototypes_II)


# In[4]:

prototypes_II


# In[5]:

def produce_mock_data_set(degrading_factor, realsperlc=100, prototypes_Ia=['DES_SN002542.DAT'], prototypes_II=['DES_SN002457.DAT'], prototypes_Ibc=['DES_SN005399.DAT']):
    #Data root
    dat=sndata.Dataset(rt)

    types=dat.get_types()
    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3
                             
    logfile=open('new_mocks/new_mocks.LIST', 'w')
    
    for prot in range(len(prototypes_II)):
        type_II = []
        for i in range(len(dat.data[prototypes_II[prot]]['flux'])):
            type_II.append(np.random.normal(dat.data[prototypes_II[prot]]['flux'][i], dat.data[prototypes_II[prot]]['flux_error'][i]*degrading_factor, realsperlc))
        type_II = np.array(type_II)
        filename_II = 'SPCC_SUBSET/'+prototypes_II[prot]
        test_table_II = dat.data[prototypes_II[prot]]
        test_table_II.rename_column('flux', 'FLUXCAL')
        test_table_II.rename_column('flux_error', 'FLUXCALERR')
        col_II = Table.Column(name='field',data=np.zeros(len(test_table_II)) )
        test_table_II.add_column(col_II, index = 2)
        for i in range(realsperlc):
            test_table_II.replace_column('FLUXCAL', type_II[:,i])
            test_table_II.replace_column('FLUXCALERR', test_table_II['FLUXCALERR']*degrading_factor)
            make_new_data(filename_II, test_table_II, 'II_%s_%s'%(i,prototypes_II[prot]), 2)
            logfile.write('II_'+str(i)+'_'+prototypes_II[prot]+'\n')

    for prot in range(len(prototypes_Ia)):
        type_Ia = []
        for i in range(len(dat.data[prototypes_Ia[prot]]['flux'])):
            type_Ia.append(np.random.normal(dat.data[prototypes_Ia[prot]]['flux'][i], dat.data[prototypes_Ia[prot]]['flux_error'][i]*degrading_factor, realsperlc))
        type_Ia = np.array(type_Ia)
        filename_Ia = 'SPCC_SUBSET/'+prototypes_Ia[prot]
        test_table_Ia = dat.data[prototypes_Ia[prot]]
        test_table_Ia.rename_column('flux', 'FLUXCAL')
        test_table_Ia.rename_column('flux_error', 'FLUXCALERR')
        col_Ia = Table.Column(name='field',data=np.zeros(len(test_table_Ia)) )
        test_table_Ia.add_column(col_Ia, index = 2)
        for i in range(realsperlc):
            test_table_Ia.replace_column('FLUXCAL', type_Ia[:,i])
            test_table_Ia.replace_column('FLUXCALERR', test_table_Ia['FLUXCALERR']*degrading_factor)
            make_new_data(filename_Ia, test_table_Ia, 'Ia_%s_%s'%(i,prototypes_Ia[prot]), 1)
            logfile.write('Ia_'+str(i)+'_'+prototypes_Ia[prot]+'\n')
        
    for prot in range(len(prototypes_Ibc)):
        type_Ibc = []
        for i in range(len(dat.data[prototypes_Ibc[prot]]['flux'])):
            type_Ibc.append(np.random.normal(dat.data[prototypes_Ibc[prot]]['flux'][i], dat.data[prototypes_Ibc[prot]]['flux_error'][i]*degrading_factor, realsperlc))
        type_Ibc = np.array(type_Ibc)
        filename_Ibc = 'SPCC_SUBSET/'+prototypes_Ibc[prot]
        test_table_Ibc = dat.data[prototypes_Ibc[prot]]
        test_table_Ibc.rename_column('flux', 'FLUXCAL')
        test_table_Ibc.rename_column('flux_error', 'FLUXCALERR')
        col_Ibc = Table.Column(name='field',data=np.zeros(len(test_table_Ibc)) )
        test_table_Ibc.add_column(col_Ibc, index = 3)
        for i in range(realsperlc):
            test_table_Ibc.replace_column('FLUXCAL', type_Ibc[:,i])
            test_table_Ibc.replace_column('FLUXCALERR', test_table_Ibc['FLUXCALERR']*degrading_factor)
            make_new_data(filename_Ibc, test_table_Ibc, 'Ibc_%s_%s'%(i,prototypes_Ibc[prot]), 3)
            logfile.write('Ibc_'+str(i)+'_'+prototypes_Ibc[prot]+'\n')

    logfile.close()
    


# In[6]:

#produce_mock_data_set(1.1, 111, prototypes_Ia, prototypes_II, prototypes_Ibc)


# In[ ]:




# In[ ]:




# In[7]:

def AUC_from_mock_data_set(classifiers=['nb','knn','svm','neural_network','boost_dt']):
    rt1=os.path.join('new_mocks','')
    dat1=sndata.Dataset(rt1)
    for obj in dat1.object_names:
        for i in range(len(dat1.data[obj])):
            dat1.data[obj]['filter'][i]=dat1.data[obj]['filter'][i][3:7]
    types=dat1.get_types()
    types['Type'][np.floor(types['Type']/10)==2]=2
    types['Type'][np.floor(types['Type']/10)==3]=3
 
    mod1Feats=snfeatures.ParametricFeatures('newling',sampler='leastsq')
    mod1_features=mod1Feats.extract_features(dat1,nprocesses=4,chain_directory=out_int)
    mod1_features.write('%s_newling.dat' %run_name, format='ascii')
    #Unfortunately, sometimes the fitting methods return NaN for some parameters for these models.
    for c in mod1_features.colnames[1:]:
        mod1_features[c][np.isnan(mod1_features[c])]=0
    mod1Feats.fit_sn
    dat1.set_model(mod1Feats.fit_sn,mod1_features)
    
    AUC=[]
    nprocesses=4
    return_classifier=False
    columns=[]
    training_set=0.7
    param_dict={}
    scale=True
    
    if isinstance(mod1_features,Table):
        #The features are in astropy table format and must be converted to a numpy array before passing to sklearn

        #We need to make sure we match the correct Y values to X values. The safest way to do this is to make types an
        #astropy table as well.

        if not isinstance(types,Table):
            types=Table(data=[mod1_features['Object'],types],names=['Object','Type'])
        feats=join(mod1_features,types,'Object')

        if len(columns)==0:
            columns=feats.columns[1:-1]

        #Split into training and validation sets
        if np.isscalar(training_set):
            objs=feats['Object']
            objs=np.random.permutation(objs)
            training_set=objs[:(int)(training_set*len(objs))]

        #Otherwise a training set has already been provided as a list of Object names and we can continue
        feats_train=feats[np.in1d(feats['Object'],training_set)]
        feats_test=feats[~np.in1d(feats['Object'],training_set)]

        X_train=np.array([feats_train[c] for c in columns]).T
        y_train=np.array(feats_train['Type'])
        X_test=np.array([feats_test[c] for c in columns]).T
        y_test=np.array(feats_test['Type'])

    else:
        #Otherwise the features are already in the form of a numpy array
        if np.isscalar(training_set):
            inds=np.random.permutation(range(len(features)))
            train_inds=inds[:(int)(len(inds)*training_set)]
            test_inds=inds[(int)(len(inds)*training_set):]

        else:
            #We assume the training set has been provided as indices
            train_inds=training_set
            test_inds=range(len(types))[~np.in1d(range(len(types)),training_set)]

        X_train=mod1_features[train_inds]
        y_train=types[train_inds]
        X_test=mod1_features[test_inds]
        y_test=types[test_inds]


    #Rescale the data (highly recommended)
    if scale:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(np.vstack((X_train, X_test)))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    probabilities={}
    classifier_objects={}

    if nprocesses>1 and not return_classifier:
        partial_func=partial(snclassifier.__call_classifier,X_train=X_train, y_train=y_train, X_test=X_test,
                             param_dict=param_dict,return_classifier=False)
        p=Pool(nprocesses)
        result=p.map(partial_func,classifiers)

        for i in range(len(result)):
            cls=classifiers[i]
            probabilities[cls]=result[i]

    else:
        for cls in classifiers:
            probs,clf=snclassifier.__call_classifier(cls, X_train, y_train, X_test, param_dict,return_classifier)
            probabilities[cls]=probs
            if return_classifier:
                classifier_objects[cls]=clf

    for i in range(len(classifiers)):
        cls=classifiers[i]
        probs=probabilities[cls]
        fpr, tpr, auc=snclassifier.roc(probs, y_test, true_class=1)
        AUC.append(auc)
    return AUC


# In[ ]:

classifiers=['nb','svm','boost_dt']
#replace model
#run with full size data set
#put classifier names into plot
auc_allclass={}

for cl in classifiers:
    auc_grid=[]
    for i in range(11):
        produce_mock_data_set(1.+2.*i/10., 22, prototypes_Ia, prototypes_II, prototypes_Ibc)
        auc_grid=np.append(auc_grid, AUC_from_mock_data_set([cl]));
    auc_allclass[cl]=auc_grid

np.save('allclassifiers.txt', auc_allclass)
