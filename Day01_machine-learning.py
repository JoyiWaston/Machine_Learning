from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import pandas
import jieba
# load_* 加载小规模数据集
# fetch_* 获取大规模数据集

# sklearn.model_selection_train_test_split(arrays,*option)


def datasets_demo():
    """
    sklearn数据集的使用
    :return:
    """
    # 获取鸢尾花数据集
    iris = load_iris()
    print("鸢尾花数据集返回值:\n", iris)
    # 返回值是一个继承自字典的bunch
    print("鸢尾花的特征值:\n", iris["data"])
    print("鸢尾花的目标值:\n", iris.target)
    print("鸢尾花特征的名字:\n", iris.feature_names)
    print("鸢尾花目标值的名字:\n", iris.target_names)
    print("鸢尾花的描述:\n", iris.DESCR)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值:\n", x_train, x_train.shape)
    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '苏州', 'temperature': 30},
            {'city': '兰州', 'temperature': 20},
            {'city': '杭州', 'temperature': 25}]
    # 1.实例化一个转换器类
    '''sparse=true为稀疏矩阵,节省内存'''
    transfer = DictVectorizer(sparse=False)

    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名:\n", transfer.get_feature_names())
    return None


def count_demo():
    """
    文本特征抽取：CountVectorizer
    :return:
    """
    data = ["Life is short, i like python. And, i like c++ too!",
            "Life is too long, i dislike python, but i still like c++!"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is", "too"])

    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("特征名：\n", transfer.get_feature_names())
    print("data_new\n", data_new.toarray())
    return None


def count_demo_zn():
    """
    中文文本特征抽取：CountVectorizer
    :return:
    """
    data = ["生命 如此 短暂，我 想 我 会 爱上 python,但是 我 还 爱 cplusplus。",
            "生命 如此 漫长，我 想 不会 喜欢 python，但是 我 依然 爱 cplusplus！"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer()

    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("特征名：\n", transfer.get_feature_names())
    print("data_new\n", data_new)
    return None


def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    text = " ".join(list(jieba.cut(text)))
    return text


def count_demo_zn2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    # 1.将中文文本进行分词
    data = ["秋日，收获的歌，静秋，蓝天一般的灿烂。轻轻道一声：秋日安好!远方的你亦安好如初!",
            "此刻，我仿佛听见窗外的风呢喃着心事，九月天的浪漫，正踏过我的指尖",
            "一幅生命的静美图，在灯火阑珊的夜，温暖而璀璨地铺展开来。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1.实例化一个转换器类
    transfer = CountVectorizer()

    # 2.调用fit_transform
    data_final = transfer.fit_transform(data)
    print("特征名：\n", transfer.get_feature_names())
    print("data_final\n", data_final.toarray())
    return None


def tfidf_demo():
    """
    通过Tfidf方法文本特征抽取
    :return:
    """
    data = ["秋日，收获的歌，静秋，蓝天一般的灿烂。轻轻道一声：秋日安好!远方的你亦安好如初!",
            "此刻，我仿佛听见窗外的风呢喃着心事，九月天的浪漫，正踏过我的指尖",
            "一幅生命的静美图，在灯火阑珊的夜，温暖而璀璨地铺展开来。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1.实例化一个转换器类
    transfer = TfidfVectorizer()

    # 2.调用fit_transform
    data_final = transfer.fit_transform(data)
    print("特征名：\n", transfer.get_feature_names())
    print("data_final\n", data_final.toarray())
    return None


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1.获取数据
    data = pandas.read_csv("./ins/dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2.实例化一个转换器类
    transfer = MinMaxScaler(feature_range=[4, 5])

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


def standard_demo():
    """
    标准化
    :return:
    """
    # 1.获取数据
    data = pandas.read_csv("./ins/dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2.实例化一个转换器类
    transfer = StandardScaler()

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1.获取数据
    data = pandas.read_csv("./ins/f.csv")
    data = data.iloc[:, 1:-2]
    print("data:\n", data)

    # 2.实例化一个转换器类
    transfer = VarianceThreshold(threshold=5)
    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)

    # 计算某两个变量之间的相关系数
    result1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("pe_ratio与pb_ratio的相关系数：\n", result1)
    result2 = pearsonr(data["revenue"], data["total_expense"])
    print("revenue与total_expense的相关系数：\n", result2)

    return None


def pca_demo():
    """
    PCA降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1.实例化一个转换器类
    transfer = PCA(n_components=3)
    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)

    print("data_new:\n", data_new)
    return None


if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()
    # 代码2：字典特征抽取
    # dict_demo()
    # 代码3：文本特征提取
    # count_demo()
    # 代码4：中文文本特征提取
    # count_demo_zn()
    # 代码5：（自动分词）中文文本特征提取
    # count_demo_zn2()
    # 代码6：通过tTfidf方法文本特征抽取
    # tfidf_demo()
    # 代码7：归一化
    # minmax_demo()
    # 代码8：标准化
    # standard_demo()
    # 代码9：低方差特征过滤
    # variance_demo()
    # 代码10：PCA降维
    pca_demo()
