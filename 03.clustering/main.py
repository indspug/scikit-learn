# -*- coding: utf-8 -*-
import pandas as panda
import numpy as np
from sklearn.cluster import KMeans

##################################################
# main
##################################################
def main():
    
    # CSV読み込み(DataFrame型)
    data = panda.read_csv("./data.csv", sep=",")
    
    # DataFrame型 -> numpy配列
    data = np.array(
        [
            data['x1'].tolist(), 
            data['x2'].tolist(), 
            data['x3'].tolist()
        ],
        np.int32
    )
    data = data.T
    
    # k-means法でクラスタ分析
    result = KMeans(
        n_clusters=3,           # クラスタの個数
        init='k-means++',       # 初期化の方法(random, ndarrayで指定等)]
        n_init=10,              # 初期重心を選ぶ処理の実行回数
        max_iter=300,           # 繰返し回数の最大値
        tol = 0.0001,           # 収束判定の誤差値
        precompute_distances='auto',     # 距離を事前に計算するか
        verbose=0,              # 分析結果の表示(1で表示)
        random_state=None,      # 乱数のシード
        copy_x=True,            # メモリでデータを複製してから計算
        n_jobs=1                # 並列処理で初期化する際の多重度
    ).fit_predict(data)

    # クラスタ番号を表示
    print('data:', data)
    print('result:', result)
    
if __name__ == "__main__":
    main()

