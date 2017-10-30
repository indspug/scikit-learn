# -*- coding: utf-8 -*-
from sklearn import datasets
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as classifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib

##################################################
# main
##################################################
def main():
    
    # SVMのハイパーパラメータ
    C = 100.0      # 小さいほど誤分類を許容、大きいほど許容しない
    GAMMA = 0.01   # 小さいほど単純な境界、大きいほど複雑な境界
    
    # Scikit-learnが用意してくれているデータをロード
    iris = datasets.load_iris()
    
    # データを取得
    data = iris.data[:-1]
    
    # 教師データを取得
    target = iris.target[:-1]
    
    # 学習データとテストデータに分割
    train_data, test_data, train_target, test_target = \
        model_selection.train_test_split( \
            data, target, test_size=0.2, random_state=0)
    
    # ランダムフォレストで分類
    clf = classifier()
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
    joblib.dump(clf, 'clf.learn_classifier')
    
if __name__ == "__main__":
    main()

