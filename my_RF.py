import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve,average_precision_score
#from my_statistics import bootstrap_auc
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier
import numpy as np

def rf(X=None,y=None,mode='test',X_train=None, X_test=None, y_train=None, y_test=None,pattern = 0,fund = 30):#model_name是web用的

    # 创建RF随机森林分类器
    if mode =='test':
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dt = RandomForestClassifier()

    # 创建Bagging分类器，构建集成模型
    clf = BaggingClassifier(dt, n_estimators= fund, max_samples=0.8, random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 使用测试集进行预测
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # 计算ROC曲线的假正率、真正率和阈值
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线的精确率、召回率和阈值
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    # 计算平均准确率(MAP)
    average_precision = average_precision_score(y_test, y_pred_proba)

    if pattern == 0:#不用返回结果保存为
        # 使用MCC评估模型性能
        y_pred = clf.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        return mcc#默认阈值
    else:
        # 获取最佳youden's指数对应的SEN和SPE
        youden = tpr - fpr
        best_index = np.argmax(youden)  # 找到最佳youden指数对应的索引
        best_youden = youden[best_index]
        best_thresholds_roc = thresholds_roc[best_index]  # best_thresholds_roc是兼顾tpr fpr因此youden指数找最佳

        # 找最佳tpr fpr 以计算表格的最佳SEN SPE
        best_tpr = tpr[best_index]
        best_SEN = best_tpr
        best_fpr = fpr[best_index]
        best_SPE = 1 - best_fpr

        # 获取最佳F1数值，对应的precision 和recall
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_index2 = np.argmax(f1_scores)
        best_f1 = f1_scores[best_index2]  # 获得最佳f1
        best_precision = precision[best_index2]  #
        best_recall = recall[best_index2]  #

        best_thresholds_pr = thresholds_pr[best_index2]

        performance = {
            'precision':precision, 'recall':recall, 'fpr':fpr, 'tpr':tpr, 'average_precision':average_precision,'best_youden':best_youden, 'best_SEN':best_SEN, 'best_SPE':best_SPE, 'best_precision':best_precision, 'best_recall':best_recall, 'best_f1':best_f1, 'best_thresholds_pr':best_thresholds_pr,
                'best_thresholds_roc':best_thresholds_roc
        } #收集pattern=0和1情况的所有性能指标，不包括2的ROC_AUC均值和95%CI,因为太久

        set_threshold = best_thresholds_pr
        # 使用MCC评估模型性能
        y_pred = (y_pred_proba > set_threshold).astype(int)
        best_mcc = matthews_corrcoef(y_test, y_pred)

        return performance,best_mcc#返回precision 的作用是验证是否在不同的py文件下结果仍然一致

#返回thresholds_roc,thresholds_pr是为了程序化模型需要的参数

def gbdt(X=None,y=None,mode='test',X_train=None, X_test=None, y_train=None, y_test=None,pattern=0,fund = 30):
    if mode == 'test':
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 创建GBDT分类器
    dt = GradientBoostingClassifier()

    # 创建Bagging分类器，构建集成模型
    clf = BaggingClassifier(dt, n_estimators= fund, max_samples=0.8, random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 使用测试集进行预测
    y_pred_proba = clf.predict_proba(X_test)[:, 1]


    # 计算ROC曲线的假正率、真正率和阈值
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线的精确率、召回率和阈值
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    # 计算平均准确率(MAP)
    average_precision = average_precision_score(y_test, y_pred_proba)

    if pattern ==0:
        # 使用MCC评估模型性能
        y_pred = clf.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        return mcc  # 默认阈值

def xgboost(X=None,y=None,mode='test',X_train=None, X_test=None, y_train=None, y_test=None,pattern=0,fund = 30):
    if mode == 'test':
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 构建XGBoost分类器
    clf = XGBClassifier(n_estimators= fund, max_samples=0.8)
    # 训练分类器
    clf.fit(X_train, y_train)

    # 计算预测概率值
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # 计算ROC曲线的假正率、真正率和阈值
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线的精确率、召回率和阈值
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    # 计算平均准确率(MAP)
    average_precision = average_precision_score(y_test, y_pred_proba)

    if pattern ==0:
        # 使用MCC评估模型性能
        y_pred = clf.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        return mcc  # 默认阈值


