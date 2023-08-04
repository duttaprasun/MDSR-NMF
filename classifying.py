import xlwt
import numpy as np
import pandas as pd
from warnings import simplefilter
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef

def classify(sheet, i, fn, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print("\nx:", x.shape, "\ty:", y.shape, "\tx_train:", x_train.shape, "\tx_test:", x_test.shape, "\ty_train:", y_train.shape, "\ty_test:", y_test.shape)
    names = ["k Nearest Neighbors", "Neural Net - MLP", "Naive Bayes", "QDA"]
    classifiers = [KNeighborsClassifier(), MLPClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]
    nc = np.unique(y).size
    print("original number of classes: ", nc)
    if nc == 2 :
        avg = "binary"
    else :
        avg = "weighted"
    for name, clf in zip(names, classifiers) :
        print("----------", name, "----------", fn, "----------",i)
        j = 0
        sheet.write(i, j, name)
        j = 1
        sheet.write(i, j, fn)
        try :
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acs = accuracy_score(y_test, y_pred)
            f1s = f1_score(y_test, y_pred, average = avg)
            cks = cohen_kappa_score(y_test, y_pred)
            mc = matthews_corrcoef(y_test, y_pred)
            print("accuracy_score: ", acs)
            print("f1_score: ", f1s)
            print("cohen_kappa_score: ", cks)
            print("matthews_corrcoef: ", mc)
        except Exception as e :
            print("error/exception in fit(x): ", e)
        finally :
            j += 1
            sheet.write(i, j, round(acs, 6))
            j += 1
            sheet.write(i, j, round(f1s, 6))
            j += 1
            sheet.write(i, j, round(cks, 6))
            j += 1
            sheet.write(i, j, round(mc, 6))
        i += 1

def init(nme, d_original, d_class, dc, factor, red_dim):
    print(" nme:", nme, "\n", "dc:", dc, "\n", "red_dim:", red_dim, "\n", "factor:", factor)
    
    fn1 = nme + "/" + nme + "_" + "dn3mf2" + "_b_" + str(factor) + ".txt"
    f1 = open(fn1, "r")
    d1 = np.genfromtxt(f1, delimiter = ',')
    simplefilter(action = 'ignore', category = FutureWarning)
    simplefilter(action = 'ignore', category = UserWarning)
    book = xlwt.Workbook(encoding = "utf-8")
    sheet = book.add_sheet("classification_performance", cell_overwrite_ok = True)
    i = 0
    j = 2
    sheet.write(i, j, "accuracy_score")
    j += 1
    sheet.write(i, j, "f1_score")
    j += 1
    sheet.write(i, j, "cohen_kappa_score")
    j += 1
    sheet.write(i, j, "matthews_corrcoef")
    i = 1
    classify(sheet, i, "dn3mf2_b", d1, d_class)
    book.save(nme + "/" + nme + "_classify_performance_dn3mf2_" + str(dc) + "_" + str(red_dim) + "_" + str(factor) + ".xls")

def readData(choice):
    print("data choice", choice)
    if choice == 1 :
        nme = "GastrointestinalLesionsInRegularColonoscopy"
        fn = nme + "/" + "data.txt"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[1:, 2:701]
        d_class = d[1:, 701:]
    elif choice == 2 :
        nme = "OnlineNewsPopularity"
        fn = nme + "/" + nme + ".csv"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[1:, 1:60]
        d_class = d[1:, 61:]
    elif choice == 3 :
        nme = "ParkinsonsDiseaseClassification"
        fn = nme + "/" + "pd_speech_features.csv"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[2:, 1:754]
        d_class = d[2:, 754:]
    elif choice == 4 :
        nme = "StudentPerformance"
        fn = nme + "/" + "student_mat.csv"
        f = open(fn, "r")
        d = np.genfromtxt(f, delimiter = ',')
        d_original = d[1:, :32]
        d_class = d[1:, 33:]
    elif choice == 5 :
        nme = "MovieLens"
        fn = nme + "/" + "data_labled.csv"
        d = pd.read_csv(fn)
        d_original = d.iloc[:,3:-3].to_numpy()
        d_class = d["gender"].to_numpy()
    else :
        print("wrong cmd line i/p")
    if choice != 5 :
        f.close()
    print("original data shape:\t", d.shape)
    d_class = np.ravel(d_class)
    d_class = d_class.astype(int)
    print(d_original.shape, d_original[0], d_class.shape, d_class[0])
    return (nme, d_original, d_class)

def main(data_choice, f):
    data_choice = int(data_choice)
    nme, d_original, d_class = readData(data_choice)
    dr, dc = d_original.shape
    red_dim = int(f * dc)
    init(nme, d_original, d_class, dc, f, red_dim)

if __name__ == '__main__':
    data_choice = 5 # 1: GastrointestinalLesionsInRegularColonoscopy, 2: OnlineNewsPopularity, 3: ParkinsonsDiseaseClassification, 4: StudentPerformance, 5: MovieLens
    f = 0.25
    main(data_choice, f)