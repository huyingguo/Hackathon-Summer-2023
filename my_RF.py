import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve,average_precision_score
#from my_statistics import bootstrap_auc
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
import numpy as np

def rf(X=None,y=None,mode='test',X_train=None, X_test=None, y_train=None, y_test=None,pattern = 0):#model_name是web用的
    if mode=='test':
        # 生成分类数据集
        X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

    # 将数据集分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建RF随机森林分类器
    dt = RandomForestClassifier()

    # 创建Bagging分类器，构建集成模型
    clf = BaggingClassifier(dt, n_estimators=50, max_samples=0.8, random_state=42)

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

def gbdt(X=None,y=None,mode='test',m_plt=True,X_train=None, X_test=None, y_train=None, y_test=None,pattern=0):
    if mode=='test':
        # 生成分类数据集
        X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

    # 将数据集分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建GBDT分类器
    dt = GradientBoostingClassifier()

    # 创建Bagging分类器，构建集成模型
    clf = BaggingClassifier(dt, n_estimators=50, max_samples=0.8, random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 使用测试集进行预测
    y_pred_proba = clf.predict_proba(X_test)[:, 1]


    # 计算ROC曲线的假正率、真正率和阈值
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)
    if m_plt:
        # 绘制ROC曲线图
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show(block=True)

    # 计算PR曲线的精确率、召回率和阈值
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    # 计算平均准确率(MAP)
    average_precision = average_precision_score(y_test, y_pred_proba)
    if m_plt:
        # 绘制PR曲线图
        plt.plot(recall, precision, label='TPR curve (area = %0.2f)' % average_precision)
        plt.plot([0, 1], [1, 0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall example')
        plt.legend(loc="lower right")
        plt.show(block=True)

    if pattern ==0:
        return [precision,recall,fpr,tpr,average_precision]
    elif pattern == 1:
        #获取最佳youden's指数对应的SEN和SPE
        youden = tpr - fpr
        best_index = np.argmax(youden)#找到最佳youden指数对应的索引
        best_youden = youden[best_index]
        best_thresholds_roc = thresholds_roc[best_index]#best_thresholds_roc是兼顾tpr fpr因此youden指数找最佳

        #找最佳tpr fpr 以计算表格的最佳SEN SPE
        best_tpr = tpr[best_index]
        best_SEN = best_tpr
        best_fpr = fpr[best_index]
        best_SPE = 1 - best_fpr


        #获取最佳F1数值，对应的precision 和recall
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_index2 = np.argmax(f1_scores)
        best_f1 = f1_scores[best_index2]#获得最佳f1
        best_precision = precision[best_index2]#
        best_recall = recall[best_index2]#

        best_thresholds_pr = thresholds_pr[best_index2]
        return [best_youden,best_SEN,best_SPE,best_precision,best_recall,best_f1,best_thresholds_pr,best_thresholds_roc]#best_thresholds_roc放在最后，方便调用
    elif pattern == 2:#专门返回95%CI，怕意外或时间太长
        mean_roc_auc, range95ci_roc = bootstrap_auc(clf, X_train, X_test, y_train, y_test, nsamples=1000,seed_n=0)  # 获得ROC_AUC的95%CI
        # print('Rf的ROC_AUC的95%CI:','(',"{:.3f}".format(range95ci[0]),',',"{:.3f}".format(range95ci[1]),')')
        return mean_roc_auc, range95ci_roc#前者是一个值，后者是个列表


