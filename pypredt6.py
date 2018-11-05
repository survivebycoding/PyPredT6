def PyPredT6():
    import tkinter as tk

    class SampleApp(tk.Tk):
      def __init__(top):
        tk.Tk.__init__(top)
        top.geometry('550x300+500+300')
        top.title('PyPredT6')
        top.configure(background='plum1')
        top.newline = tk.Label(top, text="", bd =5, bg='plum1').grid(row=0,column=3)
        
        
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
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
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
    from sklearn.ensemble import ExtraTreesClassifier
    import warnings
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    warnings.filterwarnings("ignore")
    
    f=random.seed()

    #getting feature vector of sequence to be predicted
    featurevector=featureextraction(peptide_predict_file, nucleotide_predict_file, total)
    print(len(featurevector))

 
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
    
    for i in range(a1[0]):
        for j in range(a1[1]):
            X[i][j]=eff[i][j]
        Y[i,0]=0
        #print(i)    
    for i in range(a2[0]):
        for j in range(a2[1]):
            X[i+a1[0]][j]=noneff[i][j]
        Y[i+a1[0]][0]=1
        
        
    
    warnings.filterwarnings("ignore")
    print('Resampling the unbalanced data...')
    X_resampled, Y_resampled = SMOTE(kind='borderline1').fit_sample(X, Y)
    
    #Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler().fit(X_resampled)
    X = scaler.transform(X_resampled)


  
    #Removing features with low variance

    model = ExtraTreesClassifier()
    model.fit(X_resampled, Y_resampled)
    X_resampled=model.fit_transform(X_resampled, Y_resampled)
    featurevector=model.transform(featurevector)
    newshape=X_resampled.shape
    

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
    model.add(Dense(newshape[1]+1, activation='relu', input_shape=(newshape[1],)))
    model.add(Dense(500, activation='relu'))
    #model.add(Dense(800, activation='relu'))
    #model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(90, activation='relu'))
    # Add an output layer 
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])
    model.fit(X_train, y_train,epochs=1000, batch_size=25, verbose=0)
    score = model.evaluate(X_test, y_test,verbose=0)
    ANN = model.predict(X_test)
    ANN = model.predict(featurevector)

    y_train=[]
    y_test=[]
    y_train=y_t
    y_test=y_te
            
    #SVM
    print("Training Support Vector Machine...") 
    clf1 = svm.SVC(decision_function_shape='ovr', kernel='linear', max_iter=1000)
    clf1.fit(X_train, y_train)
    y_pred=clf1.predict(X_test)
    results=cross_val_score(clf1, X_test, y_test, cv=10)
    SVM=clf1.predict(featurevector)

    #KNN
    print("Training k-Nearest Neighbor ...") 
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train) 
    results=cross_val_score(neigh, X_test, y_test, cv=10)
    y_pred=neigh.predict(X_test)
    KNN=neigh.predict(featurevector)

    #Naive Bayes
    print("Training Naive Bayes...") 
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    results=cross_val_score(clf, X_test, y_test, cv=10)
    y_pred=clf.predict(X_test)
    DT=clf.predict(featurevector)
     
    #RandomForest
    print("Training Random Forest...") 
    rf = RandomForestClassifier(random_state=0, min_samples_leaf=100)
    rf.fit(X_train, y_train)
    results=cross_val_score(rf, X_test, y_test, cv=10)
    y_pred=rf.predict(X_test)
    RF=clf.predict(featurevector)
    
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

    print('-----------------------Results-----------------------')
    for i in range(len(ANN)):
        if vote_result[i][0]>=vote_result[i][1]:
            print('Sequence ',i+1,' is a probable Type 6 Effector')
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
    import numpy as np
    
    feature = [[0 for x in range(872)] for y in range(total)]

    #amino acid feature set
    aa=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

    #conjoint triad initialization
    S1=['A','G','V']
    S2=['I','L','F','P']
    S3=['Y','M','T','S']
    S4=['H','N','Q','W']
    S5=['R','K']
    S6=['D','E']
    S7=['C']
    N=np.zeros((1,343))
    feature1=np.zeros((1,343))
    f=0
    for i in range(7):
        for j in range(7):
            for k in range(7):
                N[0,f]=(((i+1)*100)+((j+1)*10)+(k+1))
                f=f+1

    #dipeptide initialization            
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
            feature[line_number][0+420]=round((str.count('D')+ str.count('E')+str.count('K')+str.count('H')+str.count('R'))/len(str),4)

            #Aliphatic (ILV)
            feature[line_number][1+420]=round((str.count('I')+ str.count('L')+str.count('V'))/len(str),4)

            # Aromatic (FHWY)
            feature[line_number][2+420]=round((str.count('F')+str.count('H')+ str.count('W')+str.count('Y'))/len(str),4)

            # Polar (DERKQN)
            feature[line_number][3+420]=round((str.count('D')+str.count('E')+str.count('R')+ str.count('K')+str.count('Q')+str.count('N'))/len(str),4)
            
            # Neutral (AGHPSTY)
            feature[line_number][4+420]=round((str.count('A')+str.count('G')+str.count('H')+str.count('P')+ str.count('S')+str.count('T')+str.count('Y'))/len(str),4)
           
            # Hydrophobic (CFILMVW)
            feature[line_number][5+420]=round((str.count('C')+str.count('F')+str.count('I')+str.count('L')+ str.count('M')+str.count('V')+str.count('W'))/len(str),4)
       
            # + charged (KRH)
            feature[line_number][6+420]=round((str.count('K')+str.count('R')+str.count('H'))/len(str),4)

            # - charged (DE)
            feature[line_number][7+420]=round((str.count('D')+str.count('E'))/len(str),4)

            # Tiny (ACDGST)
            feature[line_number][8+420]=round((str.count('A')+str.count('C')+str.count('D')+str.count('G')+str.count('S')+str.count('T'))/len(str),4)

            # Small (EHILKMNPQV)
            feature[line_number][9+420]=round((str.count('E')+str.count('H')+str.count('I')+str.count('L')+str.count('K')+str.count('M')+str.count('N')+str.count('P')+str.count('Q')+str.count('V'))/len(str),4)

            # Large (FRWY)
            feature[line_number][10+420]=round((str.count('F')+str.count('R')+str.count('W')+str.count('Y'))/len(str),4)

            #Transmembrane amino acid
            feature[line_number][865]=round((str.count('I')+str.count('L')+str.count('V')+str.count('A'))/len(str),4)

            #dipole<1.0 (A, G, V, I, L, F, P)
            feature[line_number][866]=round((str.count('A')+str.count('G')+str.count('V')+str.count('I')+str.count('L')+str.count('F')+str.count('P'))/len(str),4)

            #1.0< dipole < 2.0 (Y, M, T, S)
            feature[line_number][867]=round((str.count('Y')+str.count('M')+str.count('T')+str.count('S'))/len(str),4)

            #2.0 < dipole < 3.0 (H, N, Q, W)
            feature[line_number][868]=round((str.count('H')+str.count('N')+str.count('Q')+str.count('W'))/len(str),4)

            #dipole > 3.0 (R, K)
            feature[line_number][869]=round((str.count('R')+str.count('K'))/len(str),4)

            #dipole > 3.0 with opposite orientation (D, E
            feature[line_number][870]=round((str.count('D')+str.count('E'))/len(str),4)


            for i in range(len(str)-2):
               p=0
               X=str[i]
               Y=str[i+1]
               Z=str[i+2]
               if X in S1:
                 p=1*100
               if X in S2:
                 p=2*100
               if X in S3:
                 p=3*100
               if X in S4:
                 p=4*100
               if X in S5:
                 p=5*100
               if X in S6:
                 p=6*100
               if X in S7:
                 p=7*100   
    
               if Y in S1:
                 p=p+1*10
               if Y in S2:
                 p=p+2*10
               if Y in S3:
                 p=p+3*10
               if Y in S4:
                 p=p+4*10
               if Y in S5:
                 p=p+5*10
               if Y in S6:
                 p=p+6*10
               if Y in S7:
                 p=p+7*10

               if Z in S1:
                 p=p+1
               if Z in S2:
                 p=p+2
               if Z in S3:
                 p=p+3
               if Z in S4:
                 p=p+4
               if Z in S5:
                 p=p+5
               if Z in S6:
                 p=p+6
               if Z in S7:
                 p=p+7
            
               for j in range(343):
                 if p==N[0,j]:
                   k=j
               feature1[0,k]=feature1[0,k]+1
            feature1[0,:]=feature1[0,:]/(len(str)-2)*100
            feature[line_number][522:865]=feature1[0,:]
            feature1=np.zeros((1,343))
            str=''
            
            line_number=line_number+1
        line=id.readline()
    id.close()
    
    for i in range(len(aa)):
                feature[line_number][i]=str.count(aa[i])/len(str)
                
    for i in range(len(dipeptide)):
                feature[line_number][i+20]=str.count(dipeptide[i])/len(str)
                
    #physicochemical properties
    # Charged (DEKHR) 
    # Charged (DEKHR) 
    feature[line_number][0+420]=round((str.count('D')+ str.count('E')+str.count('K')+str.count('H')+str.count('R')) /len(str),4)

    #Aliphatic (ILV)
    feature[line_number][1+420]=round((str.count('I')+ str.count('L')+str.count('V'))/len(str),4)

    # Aromatic (FHWY)
    feature[line_number][2+420]=round((str.count('F')+str.count('H')+ str.count('W')+str.count('Y'))/len(str),4)

    # Polar (DERKQN)
    feature[line_number][3+420]=round((str.count('D')+str.count('E')+str.count('R')+ str.count('K')+str.count('Q')+str.count('N'))/len(str),4)
            
    # Neutral (AGHPSTY)
    feature[line_number][4+420]=round((str.count('A')+str.count('G')+str.count('H')+str.count('P')+ str.count('S')+str.count('T')+str.count('Y'))/len(str),4)
           
    # Hydrophobic (CFILMVW)
    feature[line_number][5+420]=round((str.count('C')+str.count('F')+str.count('I')+str.count('L')+ str.count('M')+str.count('V')+str.count('W'))/len(str),4)
       
    # + charged (KRH)
    feature[line_number][6+420]=round((str.count('K')+str.count('R')+str.count('H'))/len(str),4)

    # - charged (DE)
    feature[line_number][7+420]=round((str.count('D')+str.count('E'))/len(str),4)

    # Tiny (ACDGST)
    feature[line_number][8+420]=round((str.count('A')+str.count('C')+str.count('D')+str.count('G')+str.count('S')+str.count('T'))/len(str),4)

    # Small (EHILKMNPQV)
    feature[line_number][9+420]=round((str.count('E')+str.count('H')+str.count('I')+str.count('L')+str.count('K')+str.count('M')+str.count('N')+str.count('P')+str.count('Q')+str.count('V'))/len(str),4)

    # Large (FRWY)
    feature[line_number][10+420]=round((str.count('F')+str.count('R')+str.count('W')+str.count('Y'))/len(str),4)

    #Transmembrane amino acid
    feature[line_number][865]=round((str.count('I')+str.count('L')+str.count('V')+str.count('A'))/len(str),4)

    #dipole<1.0 (A, G, V, I, L, F, P)
    feature[line_number][866]=round((str.count('A')+str.count('G')+str.count('V')+str.count('I')+str.count('L')+str.count('F')+str.count('P'))/len(str),4)

    #1.0< dipole < 2.0 (Y, M, T, S)
    feature[line_number][867]=round((str.count('Y')+str.count('M')+str.count('T')+str.count('S'))/len(str),4)

    #2.0 < dipole < 3.0 (H, N, Q, W)
    feature[line_number][868]=round((str.count('H')+str.count('N')+str.count('Q')+str.count('W'))/len(str),4)

    #dipole > 3.0 (R, K)
    feature[line_number][869]=round((str.count('R')+str.count('K'))/len(str),4)

    #dipole > 3.0 with opposite orientation (D, E
    feature[line_number][870]=round((str.count('D')+str.count('E'))/len(str),4)

    for i in range(len(str)-2):
               p=0
               X=str[i]
               Y=str[i+1]
               Z=str[i+2]
               if X in S1:
                 p=1*100
               if X in S2:
                 p=2*100
               if X in S3:
                 p=3*100
               if X in S4:
                 p=4*100
               if X in S5:
                 p=5*100
               if X in S6:
                 p=6*100
               if X in S7:
                 p=7*100   
    
               if Y in S1:
                 p=p+1*10
               if Y in S2:
                 p=p+2*10
               if Y in S3:
                 p=p+3*10
               if Y in S4:
                 p=p+4*10
               if Y in S5:
                 p=p+5*10
               if Y in S6:
                 p=p+6*10
               if Y in S7:
                 p=p+7*10

               if Z in S1:
                 p=p+1
               if Z in S2:
                 p=p+2
               if Z in S3:
                 p=p+3
               if Z in S4:
                 p=p+4
               if Z in S5:
                 p=p+5
               if Z in S6:
                 p=p+6
               if Z in S7:
                 p=p+7
            
               for j in range(343):
                 if p==N[0,j]:
                   k=j
               feature1[0,k]=feature1[0,k]+1
    feature1[0,:]=feature1[0,:]/(len(str)-2)*100
    feature[line_number][522:865]=feature1[0,:]

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
            feature[line_number][871]=round((str.count('G')+ str.count('C'))/len(str),4)
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
    feature[line_number][871]=round((str.count('G')+ str.count('C'))/len(str),4)         
    id.close()
    
    print('Nucleotide feature extraction done!')
    #-----------------------------------------------------------------------------------
    #secondary structure
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
            
            structure=''
            solvent=''

            feature[line_number][515]=round((str.count('E')+str.count('A')+str.count('L')+str.count('M')+str.count('Q')+str.count('K')+str.count('R')+str.count('H'))/len(str),4)
            feature[line_number][516]=round((str.count('V')+str.count('I')+str.count('Y')+str.count('C')+str.count('W')+str.count('F')+str.count('T'))/len(str),4)
            feature[line_number][517]=round((str.count('G')+str.count('N')+str.count('P')+str.count('S')+str.count('D'))/len(str),4)


            feature[line_number][518]=round((str.count('A')+str.count('L')+str.count('F')+str.count('C')+str.count('G')+str.count('I')+str.count('V')+str.count('W'))/len(str),4)
            feature[line_number][519]=round((str.count('R')+str.count('K')+str.count('Q')+str.count('E')+str.count('N')+str.count('D'))/len(str),4)
            feature[line_number][520]=round((str.count('M')+str.count('S')+str.count('P')+str.count('T')+str.count('H')+str.count('Y'))/len(str),4)
            feature[line_number][521]=round((str.count('M')+str.count('S')+str.count('P')+str.count('T')+str.count('H')+str.count('Y'))/len(str),4)
    
            line_number=line_number+1
            str=''
            str=str+line
        line=id.readline()
        
    structure=''
    solvent=''
    

    #secondary
    feature[line_number][515]=round((str.count('E')+str.count('A')+str.count('L')+str.count('M')+str.count('Q')+str.count('K')+str.count('R')+str.count('H'))/len(str),4)
    feature[line_number][516]=round((str.count('V')+str.count('I')+str.count('Y')+str.count('C')+str.count('W')+str.count('F')+str.count('T'))/len(str),4)
    feature[line_number][517]=round((str.count('G')+str.count('N')+str.count('P')+str.count('S')+str.count('D'))/len(str),4)

    #solvent
    feature[line_number][518]=round((str.count('Q')+str.count('E')+str.count('D'))/len(str),4)
    feature[line_number][519]=round((str.count('R')+str.count('K')+str.count('N')+str.count('G'))/len(str),4)
    feature[line_number][520]=round((str.count('A')+str.count('L')+str.count('F')+str.count('C')+str.count('I')+str.count('V'))/len(str),4)
    feature[line_number][521]=round((str.count('W')+str.count('M')+str.count('S')+str.count('P')+str.count('T')+str.count('H')+str.count('Y'))/len(str),4)
                
    id.close()

    print('Extraction of secondary structure and solvent accessibility features done!')
    with open("featurefile_noneff.csv", 'w') as myfile:
      wr = csv.writer(myfile)
      wr.writerows(feature)
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
 
    
    #extract webaddress from data
    result_url=re.search("(?P<url>https?://[^\s]+)", data).group("url")
    
    result_url=result_url[:-1]
    r1 = requests.get(result_url)
    data1=r1.text
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

