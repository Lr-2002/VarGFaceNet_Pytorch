import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_lfw_people
import sys
sys.path.append('../')
from util import load_np, save_csv


X_train, X_test, y_train, y_test = load_np()
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))

# 训练模型
model.fit(X_train, y_train)
# 进行预测
y_pred = model.predict(X_test)

# 提取y_test中实际存在的标签
report = classification_report(y_test, y_pred, output_dict=True)
save_csv(report, 'svm.csv')
# 更新 classification_report 的 target_names
# 输出分类报告和混淆矩阵
# print(classification_report(y_test, y_pred, target_names=lfw_people.target_names))
print(confusion_matrix(y_test, y_pred))
