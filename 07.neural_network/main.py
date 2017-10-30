# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.externals import joblib

##################################################
# main
##################################################
def main():
    
    # ニューラルネットワークのパラメータ
    parameters = [
        {   'solver'                : ['sgd', 'adam'],
            'activation'            : ['logistic', 'relu'],
            'hidden_layer_sizes'    : [(20,), (40,), (80,), (160,), (320,), (640,)],
            'max_iter'              : [200, 400],
            'early_stopping'        : [False],
        },
        {   'solver'                : ['sgd', 'adam'],
            'activation'            : ['logistic', 'relu'],
            'hidden_layer_sizes'    : [(20,10), (40,20),],
            'max_iter'              : [400, 800],
            'early_stopping'        : [False],
        },
        #{   'solver'                : ['sgd', 'adam'],
        #    'activation'            : ['logistic', 'relu'],
        #    'hidden_layer_sizes'    : [(10,10,10)],
        #    'max_iter'              : [800],
        #    'early_stopping'        : [False],
        #},
    ]
    
    # Scikit-learnが用意してくれているデータをロード
    #iris = datasets.load_iris()
    iris = datasets.load_digits()
    
    # データを取得
    data = iris.data[:-1]
    
    # 教師データを取得
    target = iris.target[:-1]
    
    # 学習データとテストデータに分割
    train_data, test_data, train_target, test_target = \
        model_selection.train_test_split( \
            data, target, test_size=0.2, random_state=0)

    # ニューラルネットワークで教師あり学習
    clf = model_selection.GridSearchCV( \
                MLPClassifier(), parameters, scoring='accuracy')
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
    print('##################################################')
    print('Best Estimator:%s' % clf.best_estimator_)
    
    # 予測モデルをダンプ
    joblib.dump(clf, 'clf.learn')
    
if __name__ == "__main__":
    main()

