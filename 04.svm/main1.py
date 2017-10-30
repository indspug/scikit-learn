# -*- coding: utf-8 -*-
#import pandas as panda
#import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib

##################################################
# main
##################################################
def main():
    
    # Scikit-learnが用意してくれているテスト用データをロード
    digits = datasets.load_digits()
    
    # データを取得
    data = digits.data[:-1]
    
    # 教師データを取得
    target = digits.target[:-1]
    
    # 学習データとテストデータに分割
    train_data, test_data, train_target, test_target = \
        model_selection.train_test_split( \
            data, target, test_size=0.2, random_state=0)

    # サポートベクターマシンで教師あり学習
    #clf = svm.SVC(gamma=0.001, C=100.)
    clf = svm.SVC(C=1.0, gamma='auto')
    clf.fit(train_data, train_target)
    
    # 作成した分類器でテストデータを分類
    predicated = clf.predict(test_data)
    
    # 結果を表示
    print('data.shape:', data.shape)
    print('target.shape:', target.shape)
    print('##################################################')
    print('Classification report for classifier %s:\n%s\n' % \
            (clf, metrics.classification_report(test_target, predicated)) )
    print('##################################################')
    print('Accuracy Score %s\n' % (metrics.accuracy_score(test_target, predicated)) )
    print('##################################################')
    print('Confusion matrix:\n%s' % \
            metrics.confusion_matrix(test_target, predicated) )
    
    # 予測モデルをダンプ
    joblib.dump(clf, 'clf.learn')
    
if __name__ == "__main__":
    main()

