# coding: utf-8
# pip install numpy scikit-learn
from sklearn import preprocessing
import numpy as np
import re
import os
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import cross_val_score
PRIMELE_N_CUVINTE = 6000
def accuracy(y, p):
    return 100 * (y==p).astype('int').mean()

def files_in_folder(mypath):
    fisiere = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            fisiere.append(os.path.join(mypath, f))
    return sorted(fisiere)

def extrage_fisier_fara_extensie(cale_catre_fisier):
    nume_fisier = os.path.basename(cale_catre_fisier)
    nume_fisier_fara_extensie = nume_fisier.replace('.txt', '')
    return nume_fisier_fara_extensie

def citeste_texte_din_director(cale):
    date_text = []
    iduri_text = []
    for fis in files_in_folder(cale):
        id_fis = extrage_fisier_fara_extensie(fis)
        iduri_text.append(id_fis)
        with open(fis, 'r', encoding='utf-8') as fin:
            text = fin.read()

        # incercati cu si fara punctuatie sau cu lowercase
        text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        cuvinte_text = text_fara_punct.split()
        date_text.append(cuvinte_text)
    return (iduri_text, date_text)

### citim datele ###
dir_path = 'C:\\Users\\Alexandru Voinea\\Desktop\\dataPython\\dataPy\\trainData\\'
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

train_data_path = os.path.join(dir_path, 'trainExamples')
iduri_train, data = citeste_texte_din_director(train_data_path)


print(data[0][:10])
### citim datele ###


### numaram cuvintele din toate documentele ###
contor_cuvinte = defaultdict(int)
for doc in data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)

# extragem primele 1000 cele mai frecvente cuvinte din toate textele
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])


list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)
### numaram cuvintele din toate documentele ###


def get_bow(text, lista_de_cuvinte):
    '''
    returneaza BoW corespunzator unui text impartit in cuvinte
    in functie de lista de cuvinte selectate
    '''
    contor = dict()
    cuvinte = set(lista_de_cuvinte)
    for cuvant in cuvinte:
        contor[cuvant] = 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1
    return contor

def get_bow_pe_corpus(corpus, lista):
    '''
    returneaza BoW normalizat
    corespunzator pentru un intreg set de texte
    sub forma de matrice np.array
    '''
    bow = np.zeros((len(corpus), len(lista)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, lista)
        ''' 
            bow e dictionar.
            bow.values() e un obiect de tipul dict_values 
            care contine valorile dictionarului
            trebuie convertit in lista apoi in numpy.array
        '''
        v = np.array(list(bow_dict.values()))
        # incercati si alte tipuri de normalizari
        v = (v-np.mean(v))/np.std(v)

        bow[idx] = v
    return bow

data_bow = get_bow_pe_corpus(data, list_of_selected_words)
print ("Data bow are shape: ", data_bow.shape)

nr_exemple_train = 2600
nr_exemple_valid = 350
nr_exemple_test = len(data) - (nr_exemple_train + nr_exemple_valid)

indici_train = np.arange(0, nr_exemple_train)
indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))

print ("Histograma cu clasele din train: ", np.histogram(labels[indici_train])[0])
print ("Histograma cu clasele din validation: ", np.histogram(labels[indici_valid])[0])
print ("Histograma cu clasele din test: ", np.histogram(labels[indici_test])[0])
# clasele sunt balansate in cazul asta pentru train, valid si nr_exemple_test

# cu cat creste C, cu atat clasificatorul este mai predispus sa faca overfit
# https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

clasificator = svm.LinearSVC(penalty='l2',C = 0.01,max_iter=4000)
score_10_fold=cross_val_score(clasificator,data_bow[:2950],labels[:2950],scoring='accuracy',cv=10)

clasificator.fit(data_bow[indici_train, :], labels[indici_train])
predictii = clasificator.predict(data_bow[indici_valid, :])
print("score_10_fold=",score_10_fold)
print("Matrice confuzie:\n",confusion_matrix(labels[indici_valid],predictii))
begin=time.time()
# concatenam indicii de train si validare
# incercati diferite valori pentru C si testati pe datele de test
indici_train_valid = np.concatenate([indici_train, indici_valid])
clasificator = svm.LinearSVC(penalty='l2',C = 0.01,max_iter=4000)

clasificator.fit(data_bow[indici_train_valid, :], labels[indici_train_valid])
print(time.time()-begin)
predictii = clasificator.predict(data_bow[indici_test])

print ("Acuratete pe test cu C = 0.01: ", accuracy(predictii, labels[indici_test]))


def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("Id,Prediction\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(id_text + ',' + str(int(pred)) + '\n')
iduri_test,date_test=citeste_texte_din_director("C:\\Users\\Alexandru Voinea\\Desktop\\iaProiect\\testData-public\\testData-public") #folder kaggle date_test
print("Am citit ", len(iduri_test))
data_bow_test=get_bow_pe_corpus(date_test,list_of_selected_words)

clf = svm.LinearSVC(penalty='l2',C = 0.01,max_iter=4000)

clf.fit(data_bow,labels)
predictii_test=clf.predict(data_bow_test)

scrie_fisier_submission('subv22.csv',predictii_test,iduri_test);
'''
TODO pentru a face un submission pe kaggle:
    1) citesc datele de test si creez data_test_bow dupa modelul de mai sus
    2) antrenez un clasificator pe toate datele 
    3) generez predictii pe data_test_bow
    4) incerc cel putin doua clasificatoare (knn, naive bayes)
    5) fac matrice de confuzie pentru fiecare clasificator
    6) fac cross-validare: https://www.youtube.com/watch?v=sFO2ff-gTh0
    7) citesc cu atentie cerinta proiectului (!!! trebuie si un pdf cu detalii)
'''


