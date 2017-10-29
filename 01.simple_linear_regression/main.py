# -*- coding: utf-8 -*-
import pandas as panda
#import numpy as numpy
from sklearn import linear_model
#import matplotlib.pyplot as plt
from sklearn.externals import joblib

def main():
    # CSV読み込み(DataFrame型)
    data = panda.read_csv("./data.csv", sep=",")
    
    # LinearRegression
    clf = linear_model.LinearRegression()
    
    # 説明変数=x1  [[1], [2], [3]]
    x = data.loc[:, ['x1']].as_matrix()
    #x = data['x1'].as_matrix()
    
    # 目的変数=x2 [1, 2, 3]
    y = data['x2'].as_matrix()
    
    # 単回帰
    clf.fit(x, y)
    
    # 回帰係数と切片の抽出
    a = clf.coef_
    b = clf.intercept_
    
    # 表示
    print("回帰係数:", a);
    print("切片:", b);
    print("決定係数:", clf.score(x,y));
    
    # 予測モデルをダンプ
    joblib.dump(clf, 'clf.learn')
    
if __name__ == "__main__":
    main()

