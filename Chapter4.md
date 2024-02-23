## Chapter 4: 자동화된 엔드투엔드 머신러닝 솔루션 생성

import autokeras as ak

# Initialize the image classifier.
clf = ak.ImageClassifier(max_trials=2)  # It tries two different models.

# Feed the image classifier with training data
# 20% of the data is used as validation data by default for tuning
# the process may run for a bit long time, please try to use GPU
clf.fit(x_train, y_train, epochs=3)  # each model is trained for three epochs
