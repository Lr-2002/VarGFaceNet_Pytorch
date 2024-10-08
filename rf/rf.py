import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys
sys.path.append('../')
from util import load_np, save_csv

X_train, X_test, y_train, y_test = load_np()

# 创建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)
# 进行预测
y_pred = model.predict(X_test)

# 输出分类报告和混淆矩阵
csv = classification_report(y_test, y_pred, output_dict=True)
save_csv(csv, 'rf.csv')
print(confusion_matrix(y_test, y_pred))
