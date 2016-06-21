# Gradient Boosted Decision Tree

木構造のブースティングモデルによる予測器、Gradient Boosted Decision Tree の実装。

## 必要なライブラリ

* matplotlib
* numpy
* seaborn(sample.pyのグラフ表示)

## 回帰の結果の例

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
