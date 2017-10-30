# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import svm
#from sklearn import grid_search
#from sklearn.grid_search import GridSearchCV
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib

##################################################
# main
##################################################
def main():
    
    # SVMのハイパーパラメータ
    parameters = [
        {   'C'         : [1., 10., 100., 1000.],
            'kernel'    : ['linear']
        },
        {   'C'         : [1., 10., 100., 1000.],
            'kernel'    : ['poly'],
            'degree'    : [2, 3, 4, 5],
            'gamma'     : [1., 0.1, 0.01, 0.001, 0.0001]
        },
        {   'C'         : [1., 10., 100., 1000.],
            'kernel'    : ['rbf'],
            'gamma'     : [1., 0.1, 0.01, 0.001, 0.0001]
        }
    ]
    #C = 1000.0      # 小さいほど誤分類を許容、大きいほど許容しない
    #GAMMA = 0.001   # 小さいほど単純な境界、大きいほど複雑な境界
    
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
    clf = model_selection.GridSearchCV(svm.SVC(), parameters, scoring='accuracy')
    #clf = grid_search.GridSearchCV(svm.SVC(), parameters)
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
    print('##################################################')
    print('Grid Scores:\n')
    print('mean  var            params')
    for i in range(len(clf.cv_results_['params'])):
        mean_score = clf.cv_results_['mean_test_score'][i]
        std = clf.cv_results_['std_test_score'][i]
        params = clf.cv_results_['params'][i]
        print('%0.3f (+/-%0.03f) for %r' % \
                (mean_score, std*2, params) )
    print('##################################################')
    print('Best Parameters:%s\n' % clf.best_params_)
    
    # 予測モデルをダンプ
    joblib.dump(clf, 'clf.learn')
    
if __name__ == "__main__":
    main()

