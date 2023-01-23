# Optimizer比較

いくつかのOptimizerについて精度・学習時間などの観点で実験した結果についてを示す.   

## 概要
検証用Datasetについてはkaggleからダウンロードした学習データを利用.  
StratifiedKFoldによって5foldに分割し、そのうちの1foldを使用している.  
分割したファイルは sample_train.csv, sample_valid.csv としてリポジトリに配置している.  
    
D-Adaptationにはschedulerについては通常通りのものを利用との記載が見受けられたため、  
timm のスケジューラを利用して 3epoch の warmup を実施後に通常の lr となるように設定.  
  
オーグメンテーションなどの設定はコード記載通り.  

## 結果

|  Optimizer  |  Accuracy  |  Training Time | Initial LR | warmup LR |
| ---- | ---- | ---- | ---- | ---- |
|  Adam  | 0.764 |  981 sec.  | 1e-4 | 1e-7 |
|  Adam with SAM  | 0.772 |  1605 sec.  | 1e-4 | 1e-7 |
|  DAdaptAdam  | 0.625 |  1252 sec.  | 1.0 | 1e-3 |
  
<img src="images/epoch_acc_curve.png" alt="epoch_accuracy_curve" width="800px" />
  
## 所見
現状のところ、Adam with SAMが指標面では優秀.  
ただし学習時間がおおよそ倍程度に伸びることから常用に関しては難がある.  
  
DAdaptationについては設定(または利用方法)が間違えているのか、指標があまり伸びなかった.  
lrの調整が不要になるとしても精度が伸びないのであれば選択肢から外れる.  
また学習epoch数が不足しているとしても学習時間の面で不利となるため活用が厳しいか.  

## 使用方法

### 1. ファイルの用意
下記に記載した引用元(kaggle)より画像データをダウンロード.  
train.py と同一のディレクトリに data/dog-breed-identification を展開する.  
具体的には下記のような構成となることを想定.

```bash
showcase-optimizer/data/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg
showcase-optimizer/data/dog-breed-identification/train/00a338a92e4e7bf543340dc849230e75.jpg
showcase-optimizer/data/dog-breed-identification/train/00a366d4b4a9bbb6c8a63126697b7656.jpg

...(省略)...

showcase-optimizer/data/dog-breed-identification/train/fff43b07992508bc822f33d8ffd902ae.jpg
showcase-optimizer/README.md
showcase-optimizer/sample_train.csv
showcase-optimizer/sample_valid.csv
showcase-optimizer/train.py
```

### 2. 学習実行
train.py のから試行したい optimizer を編集・選択(初期値ではAdamとしている).
```python
class CFG:
    #
    # ...
    #
    optimizer = 'Adam'
    # optimizer = 'SAM_Adam'
    # optimizer = 'DAdaptAdam'
```

### 3. 実行と出力
下記コマンドで実行する.  
```bash
python train.py >> log.txt
```

## 実験環境
実験環境としては以下の通り.  
|   |  |
| ---- | ---- |
| CPU | AMD Ryzen 3700X |
| RAM | 64GB |
| GPU | NVIDIA RTX 4070 Ti | 
| Framework | PyTorch 1.13.1 |
| batch_size | 64 |
| epoch | 50 |
| amp | True (fp16) |

## 引用
* Dataset  
kaggleの[dog-breed-identification](https://www.kaggle.com/c/dog-breed-identification)より引用.  
  
* SAM  
論文は[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)  
実装については[このリポジトリ](https://github.com/davda54/sam/)から引用. 
  
* D-Adaptation(DAdaptAdam)  
[D-Adaptation](https://github.com/facebookresearch/dadaptation)の実装を利用. 
