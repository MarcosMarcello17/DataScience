from numpy.core.fromnumeric import var
import pandas as pd
from scipy.sparse import data
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def prepararDataFrame(dataframe):
    retorno = dataframe.dropna()
    return retorno

#PARTE 1: ARBOLES DE DECISION PARA REGRESION

def arbolDecision(dataframe, target):
    X = dataframe.drop([target], axis = 1)
    y = dataframe[target]
    clf = tree.DecisionTreeRegressor(max_depth=3).fit(X, y)
    tree.plot_tree(clf, fontsize=5)
    plt.show()


#PARTE 2: TRANSFORMACION DEL DATASET

def transformarDataset(dataframe, target, valor, nuevonom):
    arr = dataframe[target].to_numpy()
    nval = []
    for i in arr:
        if (i < valor):
            nval.append(0)
        else:
            nval.append(1)
    dataframe.drop([target], axis = 1)
    #dataframe.loc[:,nuevonom] = nval
    dataframe.loc[:,target] = nval
    dataframe = dataframe.rename(columns = {target : nuevonom})
    return dataframe

def plotear(df, x, y):
    plt.scatter(df[x], df[y], c=df['PaisDesarrollado'])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.text(900, 5, "Yellow: 1, Violet: 0")
    plt.show()

def graficos(df):
    plotear(df, 'Income composition of resources', 'GDP')
    plotear(df, 'Schooling', 'Adult Mortality')
    plotear(df, 'Schooling', 'GDP')
    

#PARTE 3: MODELOS DE CLASIFICACION
#Bootstrap Aggregation

def BootstrapAggregation(dataframe, n):
    bootstrap = []
    for b in range(0, n):
        bootstrap.append(dataframe.sample(frac = 0.5, replace = True))
    return bootstrap

def bagging_score(b_sample, df, target):
    pred = []
    dfs = {}
    for b in range(0, len(b_sample)):
        df_test = df.drop(b_sample[b].index.values)
        X = b_sample[b].drop([target], axis=1)
        y = b_sample[b][target]
        X_test = df_test.drop([target], axis=1)
        clf = tree.DecisionTreeClassifier().fit(X,y)
        pred = clf.predict(X_test)
        for i in range(0, len(pred)):
            if X_test.index.values[i] in dfs:
                dfs[X_test.index.values[i]].append(pred[i])
            else:
                dfs[X_test.index.values[i]] = [pred[i]]
    score = count_hits(dfs, df, target)
    return score / len(dfs)

def count_hits(values, df, target):
    count = 0
    for key in values:
        values[key] = voting(values[key])
        if key in df[target].index.values and values[key] == df[target][key]:
            count += 1
    return count

def voting(lst):
    one = lst.count(1)
    zero = lst.count(0)
    if one > zero:
        return 1
    return 0

def bootstrap(dataframe):
    x = []
    y = []
    for i in range(0, 10):
        x.append(10 * (i + 1))
        samples = BootstrapAggregation(dataframe, (10 * (i+1)))
        y.append(bagging_score(samples, dataframe, 'PaisDesarrollado'))
    plt.scatter(x, y)
    plt.xlabel("Cantidad de muestras")
    plt.ylabel("Accuracy")
    plt.show()

def predicciones(b_sample, df, target, umbral = 0.5):
    pred = []
    dfs = {}
    for b in range(0, len(b_sample)):
        df_test = df.drop(b_sample[b].index.values)
        X = b_sample[b].drop([target], axis=1)
        y = b_sample[b][target]
        X_test = df_test.drop([target], axis=1)
        clf = tree.DecisionTreeClassifier().fit(X,y)
        pred = clf.predict_proba(X_test)
        for i in range(0, len(pred)):
            if X_test.index.values[i] in dfs:
                if(pred[i][1] > umbral):
                    dfs[X_test.index.values[i]].append(1)
                else:
                    dfs[X_test.index.values[i]].append(0)
            else:
                if(pred[i][1] > umbral):
                    dfs[X_test.index.values[i]] = [1]
                else:
                    dfs[X_test.index.values[i]] = [0]
    predRet = []
    for key in dfs:
        predRet.append(voting(dfs[key]))
    return predRet

def matriz_confusion(dataframe, target):
    y = dataframe[target]
    sample = BootstrapAggregation(dataframe, 100)
    y_pred = predicciones(sample, dataframe, target)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print("­­­­­­­­­­­­­­­­­­­­­Matriz de confusion con umbral neutro")
    print(pd.DataFrame({"1": [fn, tp], "0": [tn, fp]}))
    print("­­­­­­­­­­­­­­­­­­­­­")
    vp_graf = []
    fp_graf = []
    y_graf = []
    for i in np.arange(0, 1, 0.1):
        y_graf.append(i)
        y_pred = predicciones(sample, dataframe, target, i)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        vp_graf.append(tp)
        fp_graf.append(fp)
    plt.scatter(y_graf, vp_graf)
    plt.xlabel("Umbral")
    plt.ylabel("Valor de VP")
    plt.show()
    plt.scatter(y_graf, fp_graf)
    plt.xlabel("Umbral")
    plt.ylabel("Valor de FP")
    plt.show()
    
#Adaboost y GradientBoosting

def ClassifierBoosting(dataframe, var_objetivo, clf):
    X = dataframe.drop([var_objetivo], axis = 1)
    y = dataframe[var_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    print("Score:", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(pd.DataFrame({"1": [fn, tp], "0": [tn, fp]}))
    print("Clasificadores: ")
    print(clf.decision_function(X_test))
    d_tree = tree.DecisionTreeClassifier(max_depth=1, random_state=209652396).fit(X_train, y_train)
    plt.figure(figsize=(6,6))
    tree.plot_tree(d_tree, fontsize = 10)
    plt.show()

def AdaBoost(dataframe, var_objetivo):
    clf = AdaBoostClassifier(n_estimators=10, learning_rate=1, random_state=0)
    print("\nAdaboost:")
    ClassifierBoosting(dataframe, var_objetivo, clf)


def GradientBoosting(dataframe, var_objetivo):
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1, max_depth=1, random_state = 0)
    print("\nGradient:")
    ClassifierBoosting(dataframe, var_objetivo, clf)

#En ambos casos, existen casos de overfitting. Habria que reducir el valor de n_estimators asi el modelo no se pega a los datos de entrenamiento


df = pd.read_csv('Life_Expectancy.csv')
dfLimpio = prepararDataFrame(df)
#Parte 1
arbolDecision(dfLimpio, 'Life expectancy')
#Parte 2
df2 = transformarDataset(dfLimpio, 'Life expectancy', 72, 'PaisDesarrollado')
graficos(df2)
#Parte 3
bootstrap(df2)
matriz_confusion(df2, 'PaisDesarrollado')
AdaBoost(df2, 'PaisDesarrollado')
GradientBoosting(df2, 'PaisDesarrollado')
