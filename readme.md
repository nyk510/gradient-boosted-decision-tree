# Gradient Boosted Decision Tree

木構造のブースティングモデルによる予測器、Gradient Boosted Decision Tree の実装。

## 必要なライブラリ

* matplotlib
* numpy
* seaborn(sample.pyのグラフ表示)

## 数値実験

### MNIST の手書きデータ分類問題

* training data
  * `MNIST Original`の手書き文字データ
  * 出力は`{0,1,2,...,9}`の１０クラス
  * そのままだと時間がかかりすぎるので、二値分類（３と８の分類）で`datasize=2000`になおして実行

* Gradient Boosted Tree のparameters
  * 目的関数：交差エントロピー
  * 活性化関数：ロジスティクスシグモイド関数

### 結果

```
2016-06-23 01:20:01,501	__main__	This is MNIST Original dataset
2016-06-23 01:20:01,502	__main__	target: 3,8
2016-06-23 01:20:01,803	__main__	training datasize: 2000
2016-06-23 01:20:01,803	__main__	test datasize: 11966
2016-06-23 01:52:45,349	__main__	accuracy:0.9745946849406653
```

分類精度97.5%を達成

### 二値分類問題

* training data:
  * 各クラスを、[1,1] [-1.,-1]を中心としたガウス分布からのサンプリング
  * 図中で青と緑で表示
* GBDTのパラメータ
  * 目的関数: 交差エントロピー
  * 活性化関数.: シグモイド関数 $\sigma(x)$ ( $\sigma(x):=\frac{1}{1+\exp[-x]}$ )

#### 結果

![](experiment_figures/binary_classification.png)


### 連続変数に対する回帰問題

`sample.py`の`regression_sample`

* training data
  * sin(x)+ノイズ（正規分布）
* GBDTのパラメータ
  * 目的関数： 二乗ロス関数
  * 活性化関数: 恒等写像
  * `max_depth=8`（毎回最大でどのぐらいの深さまで木を作るか）
  * `num_iter=20`（ブースティングの繰り返し回数）
  * `gamma=.5`（木を一段階深くするのに対するペナルティ）

#### 結果

![](experiment_figures/regression.png)
