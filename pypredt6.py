def PyPredT6():
    import tkinter as tk

    class SampleApp(tk.Tk):
      def __init__(top):
        tk.Tk.__init__(top)
        top.geometry('550x300+500+300')
        top.title('PypredT6')
        top.configure(background='plum1')
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=0,column=3)
        #top.newline = tk.Label(top, text="", bd =5).grid(row=1,column=3)
        
        top.caption = tk.Label(top, text="Please insert the file names for executing PyPredT6", bd =5, bg='plum1').grid(row=2,column=1)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=3,column=1)
        top.label1 = tk.Label(top, text="Sample peptide file", bd =5, bg='plum1').grid(row=6,column=1)
        top.label2 = tk.Label(top, text="Sample nucleotide file", bd =5, bg='plum1').grid(row=8,column=1)
        top.label3 = tk.Label(top, text="Effector feature file", bd =5, bg='plum1').grid(row=10,column=1)
        top.label4 = tk.Label(top, text="Non-effector feature file", bd =5, bg='plum1').grid(row=12,column=1)
        
        top.entry1 = tk.Entry(top, bd =3, width=40)
        top.entry2 = tk.Entry(top, bd =3, width=40)
        top.entry3 = tk.Entry(top, bd =3, width=40)
        top.entry4 = tk.Entry(top, bd =3, width=40)
        
        top.button = tk.Button(top, text="Predict!", command=top.on_button, padx=2, pady=2, width=10, bg="bisque2")
        top.entry1.grid(row=6, column=2)
        top.entry2.grid(row=8, column=2)
        top.entry3.grid(row=10, column=2)
        top.entry4.grid(row=12, column=2)
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=13,column=1)
        top.button.grid(row=16, column=2)
    
        

      def on_button(top):
        x1=top.entry1.get()
        x2=top.entry2.get()
        x3=top.entry3.get()
        x4=top.entry4.get()
        top.destroy()
        voting(x1,x2,x3,x4)
        

    app = SampleApp()
    
    app.mainloop()
       
def voting(peptide_predict_file,nucleotide_predict_file,effector_train,noneffector_train):

##    #files for prediction
##    peptide_predict_file="H:/rishika/1_work_bacterial/newresult/newanalysis/latex/PyPredT6/sample1/protein.txt"
##    nucleotide_predict_file="H:/rishika/1_work_bacterial/newresult/newanalysis/latex/PyPredT6/sample1/gene.txt"
##    #files for training
##    effector_train="H:/rishika/1_work_bacterial/newresult/newanalysis/Feature_effector.csv"
##    noneffector_train="H:/rishika/1_work_bacterial/newresult/newanalysis/Feature_noneffector.csv"
    
    total = 0
  
    with open(peptide_predict_file) as f:
     for line in f:
        finded = line.find('>')
        
        if finded == 0:
            total =total+ 1

    print('Total number of sequences to be classified: ',total)
    
    import time
    start_time = time.time()
    import random
    import pandas
    import numpy as np
    import csv
    from sklearn import svm
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from random import shuffle
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    f=random.seed()
    from sklearn.metrics import accuracy_score
    import numpy as np
    np.random.seed(123)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from keras.models import Sequential
    from keras.layers import Dense
    from imblearn.over_sampling import SMOTE, ADASYN
    from collections import Counter
    import warnings
    warnings.filterwarnings("ignore")
    
    f=random.seed()

    #getting feature vector of sequence to be predicted
    featurevector=featureextraction(peptide_predict_file, nucleotide_predict_file, total)
    
    #getting training data
    dataframe = pandas.read_csv(effector_train, header=None, sep=',')
    dataset = dataframe.values
    eff = dataset[:,0:1000].astype(float)

    dataframe = pandas.read_csv(noneffector_train, header=None, sep=',')
    dataset = dataframe.values
    noneff = dataset[:,0:1000].astype(float)


    
    a1=eff.shape
    a2=noneff.shape
    X = np.ones((a1[0]+a2[0],a1[1]))
    Y = np.ones((a1[0]+a2[0],1))
    #combine1 = [[1 for x in range(a1[1]+1)] for y in range(a1[0]+a2[0])]
    
    
    for i in range(a1[0]):
        for j in range(a1[1]):
            X[i][j]=eff[i][j]
        Y[i,0]=0
        #print(i)    
    for i in range(a2[0]):
        for j in range(a2[1]):
            X[i+a1[0]][j]=noneff[i][j]
        Y[i+a1[0]][0]=1
        
        
    
    #print (X.shape, Y.shape)
    warnings.filterwarnings("ignore")
    print('Resampling the unbalanced data...')
    X_resampled, Y_resampled = SMOTE(kind='borderline1').fit_sample(X, Y)
    #print(sorted(Counter(Y_resampled).items()))
    
    #Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler().fit(X_resampled)
    X = scaler.transform(X_resampled)
    
    
    #Removing features with low variance
##    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
##    X_resampled=sel.fit_transform(X_resampled)
##    print(type(X_resampled))

    print("Training Classifiers...")
    #train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.15, random_state=f)
    y_t=y_train
    y_te=y_test
    y_train=np.ones((len(y_t),2))
    y_test=np.ones((len(y_te),2))
    for i in range(len(y_t)):
        if y_t[i]==0:
            y_train[i][1]=0
        if y_t[i]==1:
            y_train[i][0]=0
            
    for i in range(len(y_te)):
        if y_te[i]==0:
            y_test[i][1]=0
        if y_te[i]==1:
            y_test[i][0]=0    
    
    #ANN
    print("Training Artificial Neural Network...") 
    model = Sequential()
    model.add(Dense(523, activation='relu', input_shape=(522,)))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(700, activation='relu'))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    # Add an output layer 
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
    model.fit(X_train, y_train,epochs=100, batch_size=25, verbose=0)
    score = model.evaluate(X_test, y_test,verbose=0)
    #ANN = model.predict(X_test)
    ANN = model.predict(featurevector)
    
    #print('ANN',model.evaluate(X_test, y_test,verbose=0))

    y_train=[]
    y_test=[]
    y_train=y_t
    y_test=y_te
            
    #SVM
    print("Training Support Vector Machine...") 
    clf1 = svm.SVC(decision_function_shape='ovr', kernel='linear', max_iter=100)
    clf1.fit(X_train, y_train)
    y_pred=clf1.predict(X_test)
    SVM=clf1.predict(featurevector)
    #print('SVM',accuracy_score(y_test, y_pred))

    #KNN
    print("Training k-Nearest Neighbor ...") 
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train) 
    results=cross_val_score(neigh, X_train, y_train, cv=20)
    y_pred=neigh.predict(X_test)
    KNN=neigh.predict(featurevector)
    #print('KNN',accuracy_score(y_test, y_pred))

    #DecisionTree
    print("Training Decision Tree...") 
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    results=cross_val_score(clf, X_train, y_train, cv=20)
    y_pred=clf.predict(X_test)
    DT=clf.predict(featurevector)
    #print('DT',accuracy_score(y_test, y_pred))

    #RandomForest
    print("Training Random Forest...") 
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    results=cross_val_score(rf, X_train, y_train, cv=20)
    y_pred=rf.predict(X_test)
    RF=clf.predict(featurevector)
    #print('RF',accuracy_score(y_test, y_pred))
    
    vote_result = [[0 for x in range(2)] for y in range(len(SVM))]
    for i in range(len(ANN)):
          if round(ANN[i][0])==1.0:
              vote_result[i][0]=vote_result[i][0]+1
          if round(ANN[i][1])==1.0:
              vote_result[i][1]=vote_result[i][1]+1
          if SVM[i]==0:
              vote_result[i][0]=vote_result[i][0]+1
          if SVM[i]==1:
              vote_result[i][1]=vote_result[i][1]+1
          if KNN[i]==0:
              vote_result[i][0]=vote_result[i][0]+1
          if KNN[i]==1:
              vote_result[i][1]=vote_result[i][1]+1
          if DT[i]==0:
              vote_result[i][0]=vote_result[i][0]+1
          if DT[i]==1:
              vote_result[i][1]=vote_result[i][1]+1
          if RF[i]==0:
              vote_result[i][0]=vote_result[i][0]+1
          if RF[i]==1:
              vote_result[i][1]=vote_result[i][1]+1    
    #print(vote_result)
##    print(round(ANN[30][0]),round(ANN[30][1]))
##    print(SVM[30])
##    print(KNN[30])
##    print(DT[30])
    print('-----------------------Results-----------------------')
    for i in range(len(ANN)):
        if vote_result[i][0]>vote_result[i][1]:
            print('Sequence ',i+1,' is a probable Type 6 Effector')
##        elif vote_result[i][0]==vote_result[i][1]:
##            print('Cant say')
        else:    
            print('Sequence ',i+1,' is not a Type 6 Effector')
    end_time = time.time()
    print('Execution time',(end_time-start_time))

#-----------------------------------------------------------------------------

def featureextraction(peptide_file_name,nucleotide_file_name, total):
    import csv
    import requests
    import webbrowser
    from selenium import webdriver
    import re
    import time
    
##    peptide_file_name="G:/rishika/1_work_bacterial/newresult/non_protein.txt"
##    nucleotide_file_name="G:/rishika/1_work_bacterial/newresult/non_gene.txt"
    feature = [[0 for x in range(522)] for y in range(total)]
    #amino acid feature set
    aa=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    dipeptide=list()
    for i in range(len(aa)):
        for j in range(len(aa)):
            t=''
            t=aa[i]+aa[j]
            dipeptide.append(t)
            j=j+1
    #print(dipeptide)        
    id=open(peptide_file_name,"r")
    line=id.readline()
    line=id.readline()
    str=''
    count=0
    line_number=0
    print('Extracting features from amino-acid sequences...')
    while line:
        if '>' not in line:
            str=str+line
        if '>' in line:
            #print(str)
            #single amino acid count
            for i in range(len(aa)):
                feature[line_number][i]=round(str.count(aa[i])/len(str),4)

            #dipeptide count    
            for i in range(len(dipeptide)):
                feature[line_number][i+20]=round(str.count(dipeptide[i])/len(str),4)

            #physicochemical prroperties
            # Charged (DEKHR) 
            feature[line_number][420]=feature[line_number][3]+ feature[line_number][5]+feature[line_number][11]+feature[line_number][8]+feature[line_number][1] 

            #Aliphatic (ILV)
            feature[line_number][421]=feature[line_number][9]+ feature[line_number][10]+feature[line_number][19]

            # Aromatic (FHWY)
            feature[line_number][422]=feature[line_number][13]+feature[line_number][8]+ feature[line_number][17]+feature[line_number][18]

            # Polar (DERKQN)
            feature[line_number][423]=feature[line_number][3]+feature[line_number][5]+feature[line_number][1]+ feature[line_number][11]+feature[line_number][6]+feature[line_number][2]
            
            # Neutral (AGHPSTY)
            feature[line_number][424]=feature[line_number][0]+feature[line_number][7]+feature[line_number][8]+feature[line_number][14]+ feature[line_number][15]+feature[line_number][16]+feature[line_number][18]
           
            # Hydrophobic (CFILMVW)
            feature[line_number][425]=feature[line_number][4]+feature[line_number][13]+feature[line_number][9]+feature[line_number][10]+ feature[line_number][12]+feature[line_number][19]+feature[line_number][17]
       
            # + charged (KRH)
            feature[line_number][426]=feature[line_number][11]+feature[line_number][1]+feature[line_number][8]

            # - charged (DE)
            feature[line_number][427]=feature[line_number][3]+feature[line_number][5]

            # Tiny (ACDGST)
            feature[line_number][428]=feature[line_number][0]+feature[line_number][4]+feature[line_number][3]+feature[line_number][7]+feature[line_number][15]+feature[line_number][16]

            # Small (EHILKMNPQV)
            feature[line_number][429]=feature[line_number][5]+feature[line_number][8]+feature[line_number][9]+feature[line_number][10]+feature[line_number][11]+feature[line_number][12]+feature[line_number][2]+feature[line_number][14]+feature[line_number][6]+feature[line_number][19]

            # Large (FRWY)
            feature[line_number][430]=feature[line_number][13]+feature[line_number][1]+feature[line_number][17]+feature[line_number][18]
            str=''
            
            line_number=line_number+1
        line=id.readline()
    id.close()
    
    for i in range(len(aa)):
                feature[line_number][i]=str.count(aa[i])/len(str)
                
    for i in range(len(dipeptide)):
                #print(dipeptide[i],str.count(dipeptide[i]))
                feature[line_number][i+20]=str.count(dipeptide[i])/len(str)
                
    #physicochemical properties
    # Charged (DEKHR) 
    feature[line_number][0+420]=feature[line_number][3]+ feature[line_number][5]+feature[line_number][11]+feature[line_number][8]+feature[line_number][1] 

    #Aliphatic (ILV)
    feature[line_number][1+420]=feature[line_number][9]+ feature[line_number][10]+feature[line_number][19]

    # Aromatic (FHWY)
    feature[line_number][2+420]=feature[line_number][13]+feature[line_number][8]+ feature[line_number][17]+feature[line_number][18]

    # Polar (DERKQN)
    feature[line_number][3+420]=feature[line_number][3]+feature[line_number][5]+feature[line_number][1]+ feature[line_number][11]+feature[line_number][6]+feature[line_number][2]
            
    # Neutral (AGHPSTY)
    feature[line_number][4+420]=feature[line_number][0]+feature[line_number][7]+feature[line_number][8]+feature[line_number][14]+ feature[line_number][15]+feature[line_number][16]+feature[line_number][18]
           
    # Hydrophobic (CFILMVW)
    feature[line_number][5+420]=feature[line_number][4]+feature[line_number][13]+feature[line_number][9]+feature[line_number][10]+ feature[line_number][12]+feature[line_number][19]+feature[line_number][17]
       
    # + charged (KRH)
    feature[line_number][6+420]=feature[line_number][11]+feature[line_number][1]+feature[line_number][8]

    # - charged (DE)
    feature[line_number][7+420]=feature[line_number][3]+feature[line_number][5]

    # Tiny (ACDGST)
    feature[line_number][8+420]=feature[line_number][0]+feature[line_number][4]+feature[line_number][3]+feature[line_number][7]+feature[line_number][15]+feature[line_number][16]

    # Small (EHILKMNPQV)
    feature[line_number][9+420]=feature[line_number][5]+feature[line_number][8]+feature[line_number][9]+feature[line_number][10]+feature[line_number][11]+feature[line_number][12]+feature[line_number][2]+feature[line_number][14]+feature[line_number][6]+feature[line_number][19]

    # Large (FRWY)
    feature[line_number][10+420]=feature[line_number][13]+feature[line_number][1]+feature[line_number][17]+feature[line_number][18]

    print('Amino acid feature extraction done!')
    #----------------------------------------------------------------------------------------
    print('Extracting features from nucleotide sequences...')
    #nucleotide feature set
    nucleotide=['A','T','G','C']
    dinucleotide=list()
    trinucleotide=list()
    for i in range(len(nucleotide)):
        for j in range(len(nucleotide)):
            t=''
            t=nucleotide[i]+nucleotide[j]
            dinucleotide.append(t)
            
    #print(dinucleotide)
    
    for i in range(len(nucleotide)):
        for j in range(len(nucleotide)):
            for k in range(len(nucleotide)):
             t=''
             t=nucleotide[i]+nucleotide[j]+nucleotide[k]
             trinucleotide.append(t)
             
            
    id=open(nucleotide_file_name,"r")
    line=id.readline()
    line=id.readline()
    str=''
    count=0
    line_number=0
    while line:
        if '>' not in line:
            str=str+line
        if '>' in line:
            #print(str)
            #single nucleotide count
            for i in range(len(nucleotide)):
                feature[line_number][i+431]=round(str.count(nucleotide[i])/len(str),4)

            #dinucleotide count    
            for i in range(len(dinucleotide)):
                feature[line_number][i+4+431]=round(str.count(dinucleotide[i])/len(str),4)
            

            #trinucleotide count    
            for i in range(len(trinucleotide)):
                feature[line_number][i+4+16+431]=round(str.count(trinucleotide[i])/len(str),4)
            line_number=line_number+1
        line=id.readline()
    
    #single nucleotide count
    for i in range(len(nucleotide)):
             feature[line_number][i+431]=round(str.count(nucleotide[i])/len(str),4)


    #dinucleotide count    
    for i in range(len(dinucleotide)):
             feature[line_number][i+4+431]=round(str.count(dinucleotide[i])/len(str),4)
            

    #trinucleotide count    
    for i in range(len(trinucleotide)):
             feature[line_number][i+4+16+431]=round(str.count(trinucleotide[i])/len(str),4)
    id.close()
    print('Nucleotide feature extraction done!')
    #write in feature in a csv file
##    with open("featurefile_non.csv", 'w') as myfile:
##      wr = csv.writer(myfile)
##      wr.writerows(feature)
    #-----------------------------------------------------------------------------------
    #secondary structure
    #change file name to get rest of the sequences
    print("Extracting secondary structure and solvent accessibility features...")
    print("Please wait...")
    id=open(peptide_file_name,"r")
    line=id.readline()
    str=''
    str=str+line
    line=id.readline()
    count=0
    line_number=0
    while line:
        if '>' not in line:
            str=str+line
        if '>' in line:
            count=count+1
            
            #print(str)
            #print(count)
            structure=''
            solvent=''
##            fill(str)
            fillform(str)
            structure=secondarystructure()
            solvent=solventaccessibility()
            #print(len(structure))
            #print(len(solvent))
            feature[line_number][515]=round(structure.count('H')/len(structure),4)
            feature[line_number][516]=round(structure.count('E')/len(structure),4)
            feature[line_number][517]=round(structure.count('C')/len(structure),4)

            feature[line_number][518]=round(solvent.count('E')/len(solvent),4)
            feature[line_number][519]=round(solvent.count('e')/len(solvent),4)
            feature[line_number][520]=round(solvent.count('B')/len(solvent),4)
            feature[line_number][521]=round(solvent.count('b')/len(solvent),4)
            #print(feature[line_number][515],feature[line_number][516],feature[line_number][517])
            #print(feature[line_number][518],feature[line_number][519],feature[line_number][520],feature[line_number][521])
            line_number=line_number+1
##            with open("featurefile_non.csv", 'w') as myfile:
##                wr = csv.writer(myfile)
##                wr.writerows(feature)
            #print('writing done')
            str=''
            str=str+line
        line=id.readline()
        
    structure=''
    solvent=''
    fillform(str)
    structure=secondarystructure()
    solvent=solventaccessibility()
    #print(structure)
    feature[line_number][515]=round(structure.count('H')/len(structure),4)
    feature[line_number][516]=round(structure.count('E')/len(structure),4)
    feature[line_number][517]=round(structure.count('C')/len(structure),4)

    feature[line_number][518]=round(solvent.count('E')/len(solvent),4)
    feature[line_number][519]=round(solvent.count('e')/len(solvent),4)
    feature[line_number][520]=round(solvent.count('B')/len(solvent),4)
    feature[line_number][521]=round(solvent.count('b')/len(solvent),4)
                
    id.close()

    print('Extraction of secondary structure and solvent accessibility features done!')
    #print(len(feature))
    return feature

    
def fillform(sequence):
    import requests
    import webbrowser
    from selenium import webdriver
    import re
    import time
    url = 'http://distillf.ucd.ie/~distill/cgi-bin/distill/predict_porterpaleale'
    payload = {'input_text':sequence}
    r = requests.post(url, data=payload)
    data=r.text
    #print(data)
 
    
    #extract webaddress from data
    result_url=re.search("(?P<url>https?://[^\s]+)", data).group("url")
    
    result_url=result_url[:-1]
    #print(result_url)
    r1 = requests.get(result_url)
    data1=r1.text
    #print(data1)
    while "reload" in data1:
        #since the page itself reloads every 60 seconds
        time.sleep(62)
        r1 = requests.get(result_url)
        data1=r1.text
        #print(data1)
    #print('got')

    #writing data in file
    id=open("temp.txt","w")
    id.write(data1)
    id.close()

def secondarystructure():
    str=''
    with open("temp.txt","r") as id:
     for line in id:
         if "Query_length" in line:
             for line in id:
                 
                 if "Query served" in line:
                     break
                 if line is '\n':
                     count=0
                     continue
                 else:
                     if count%3==1:
                         if line is not '\n':
                             str=str+line
                     count=count+1
                 
                    
    id.close()
    str=str[:-1]
    return str

def solventaccessibility():
    str=''
    with open("temp.txt","r") as id:
     for line in id:
         if "Query_length" in line:
             for line in id:
                 
                 if "Query served" in line:
                     break
                 if line is '\n':
                     count=0
                     continue
                 else:
                     if count%3==2:
                         if line is not '\n':
                             str=str+line
                     count=count+1
                 
                    
    id.close()
    
    str=str[:-1]
    return str

