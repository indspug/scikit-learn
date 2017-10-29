# -*- coding: utf-8 -*-
import pandas as panda
from sklearn import linear_model
from sklearn.externals import joblib

def main():
    # CSV読み込み(DataFrame型)
    data = panda.read_csv("./data.csv", sep=",")
    
    # LinearRegression
    clf = linear_model.LinearRegression(
        fit_intercept=True,     # Trueで切片を求める
        normalize=False,        # Trueで説明変数を事前に正規化する
        copy_X=True,            # Trueでメモリ内でデータを複製してから実行
        n_jobs=1                # CPUで計算する際のJOB数
    )
    
    # 説明変数=x1  [[1,2], [3,4], [5,6]]
    x = data.loc[:, ['x1', 'x2']].as_matrix()
    #x = data['x1'].as_matrix()
    
    # 目的変数=x3 [1, 2, 3]
    y = data['x3'].as_matrix()
    
    # 単回帰
    clf.fit(x, y)
    
    # 回帰係数と切片の抽出
    a = clf.coef_
    b = clf.intercept_
    
    # 表示
    print("X:", x);
    print("Y:", y);
    print("回帰係数:", a);
    print("切片:", b);
    print("決定係数:", clf.score(x,y));
    
    # 予測モデルをダンプ
    joblib.dump(clf, 'clf.learn')
    
if __name__ == "__main__":
    main()

