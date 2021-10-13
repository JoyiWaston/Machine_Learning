from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

def knn_iris_demo():
    """
    用KNN算法对鸢尾花分类
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=23)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法一：直接比对真实值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None


def knn_iris_gscv():
    """
    用KNN算法对鸢尾花分类,添加网格搜索和交叉验证
    :return:
    """
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=23)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()

    # 加入网格搜索和交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    estimator = GridSearchCV(estimator=estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法一：直接比对真实值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 最佳参数：best_params
    print("最佳参数：\n", estimator.best_params_)
    # 最佳结果：best_score
    print("最佳结果：\n", estimator.best_score_)
    # 最佳估计器：best_estimator
    print("最佳估计器：\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results
    print("交叉验证结果：\n", estimator.cv_results_)

    return None


def nb_new():
    """
    用朴素贝叶斯算法对20个新闻文本分类
    :return:
    """

    # 1.获取数据
    news = fetch_20newsgroups(data_home="./ins", subset="all")

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3.特征工程:文本特征抽取: tf - idf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.朴素贝叶斯预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5.模型调优评估
    # 方法一：直接比对真实值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None

def decision_tree_iris():
    """
    用决策树对鸢尾花数据进行分类
    :return:
    """
    # 1.获取数据集
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=23)

    # 3.决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4.模型评估
    # 方法一：直接比对真实值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：\n", y_test == y_predict)

    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 5.可视化决策树
    dot_data = export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)

    return None



if __name__ == "__main__":
    # 代码一：用KNN算法对鸢尾花分类
    # knn_iris_demo()
    # 代码二：用KNN算法对鸢尾花分类,添加网格搜索和交叉验证
    # knn_iris_gscv()
    # nb_new()
    decision_tree_iris()
