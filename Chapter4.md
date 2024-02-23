## Chapter 4: 자동화된 엔드투엔드 머신러닝 솔루션

머신러닝 문제를 정의하고 알맞은 데이터를 갖추고 있다면 쉽게 ML 솔루션을 도출할 수 있다.
- 전처리된 데이터셋을 준비
- 목적에 맞는 API 정의 (Classification, Regression, ... etc)
- 일부 튜닝할 파라미터 명시
### Ex. 이미지 분류
```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import autokeras as ak
clf = ak.ImageClassifier(max_trials=2)
clf.fit(x_train, y_train, epochs=3)
```

타 ML Framework와 마찬가지로 `.predict()`를 사용하여 예측 수행
```python
clf.predict(x_test)
```

최상의 파이프라인을 `.export_model()` 케라스로 저장
```python
clf.export_model()
```

외에도 Tensorflow, Keras와 공유하는 메서드들이 많음

튜닝 옵션들: 기존의 ML, DL에서 설정하는 파라미터들을 동일하게 지정하여 튜닝이 가능
```python
clf = ak.ImageClassifier(
    max_trials=2,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    objective="val_accuracy",
)

clf.fit(
    x_train,
    y_train,
    validation_split=0.15,
    epochs=3,
    verbose=2,
)
```

### Ex. 뉴스그룹 텍스트 분류
```python
clf = ak.TextClassifier(
    max_trials=2, overwrite=True
)

clf.fit(doc_train, label_train, verbose=2)
```
- callback option 개념이 존재 -> 10회의 연속된 epochs 동안 개선되지않으면 training 중단
- max_epochs = 1000

### Ex. 정형 데이터 분류
```python
clf = ak.StructuredDataClassifier(
    column_names=[
        "sex",
        "age",
        "n_siblings_spouses",
        "parch",
        "fare",
        "class",
        "deck",
        "embark_town",
        "alone",
    ],
    column_types={"sex": "categorical", "fare": "numerical"},
    max_trials=10,
)

clf.fit(
    train_file_path,
    "survived",
    verbose=2,
)
```
- csv 파일 그대로 데이터셋으로 지정가능
- not numeric 데이터들은 casting type를 직접 명시해주어야함. -> 결국 데이터 전처리는 사람이 개입해야함
- Classifier가 아닌 Regression 모델들도 있음
  - StructuredDataRegressor
  - ImageRegressor
  - TextRegressor
  - etc..

### Ex. 다중 레이블 이미지 분류
1. 다중 레이블 이미지 분류용 합성 데이터셋 생성
```python
from sklearn.datasets import make_multilabel_classification

X, Y = make_multilabel_classification(
    n_samples=100,         # 100장의 데이터 생성
    n_features=64,         # 64개의 피처를 가짐
    n_classes=3,           # 최대 3개의 범주에 포함
    n_labels=2,            # 평균 2개의 범주를 가진 분포
    allow_unlabeled=False,
    random_state=1,
)
X = X.reshape((100, 8, 8))
```
2. API 사용한 traning
```python
clf = ak.ImageClassifier(
    max_trials=10, multi_label=True, overwrite=True
) 
clf.fit(x_train, y_train, epochs=3, verbose=2) 
```
- 내부적으로는 one-hot encoding을 사용하지 않고 multi-hot encoding 사용
- `multi_label=True` 옵션을 주어야함