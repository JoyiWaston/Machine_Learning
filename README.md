# sklearn数据集使用
    # x数据集的特征值
    # y数据集的标签值
    # test_size测试集的大小，一般为float
    # randon_state随机数种子，不同的种子会造成不同的随机抽样结果。相同种子采样结果相同。
    # return训练集特征值，测试集特征值，训练集目标值，测试集目标值


# Feature engineering特征工程
    # pandas 数据清洗，数据处理
    # 特征抽取/特征提取 机器学习算法->统计方法->数学公式
    # 文本 -> 数值；类型 -> 数值；

### 特征提取
    # sklearn.feature_extraction

### 字典特征提取
    # sklearn.feature_extraction.DicVectorizer(sparse=True)
#### vector向量；矢量
    # 矩阵 matrix 一维数组
    # 向量 vector 二维数组
#### 父类
    # 转换器类
#### 返回sparse矩阵
    # 稀疏矩阵
    # 将非零值按位置表示出来
    # 节省内存 - 提高加载效率
#### 应用场景
    # 1）pclass，sex数据集中类别特征较多    将数据集特征->字典特征    DictVectorizer转换
    # 2）本身拿到的数据类型就是字典类型
#### 文本特征提取
    # 一篇英语短文应该把单词作为特征
    # 句子，短语，单词，字母
    # 特征:特征词
    # 方法一：CountVectorizer   统计每个样本特征词出现的频次  stop_words停用词
    # 在某一个类别的文章中，出现次数多，但其他类别中很少：关键词

    # 方法二：TfidefVectorizer Tf-idf方法：评估一个词在对数据集的重要程度
    # Tf 词频 idf 逆向文档频率
    # sklearn.feature_extraction.text.TfidfVectorizer(Stop_words=None,...)

### 特征预处理
    # 归一化  X'=(x-min)/(max-min)     x''=x'*(xm-mi)+mi
    # 如果出现异常值：最大值or最小值; 归一化由最值得来，鲁棒性较差，只适合传统精确小数据场景
    # sklearn.preprocessing,MinMaxScaler(feature_range+(0,1)...)

    # 标准化  X'=（x-均值)/σ  标准差描述各数据偏离平均数的距离（离均差）的平均数
    # 如果出现异常值，本身数据体量大，少量异常点对平均值影响不大，方差改变较小,较稳定，适大数据背景
    # sklearn.preprocessing.StandardScaler()
### 特征降维
    # 降维：在某些限定条件下，降低随机变量（特征）个数，得到一组"不相干"主变量
    # 两种方法：特征选择 主成分分析（一种特征提取方法）
#### 特征选择
    # 数据中包含冗余或相关变量（或称特征，属性，指标），旨在从原有特征中找出主要特征
    # Filter(过滤式)：主要探究特征本身特点，特征与特征和目标值之间关联
        -方差选择法：低方差特征过滤
            * sklearn.feature_selection.VarianceThreshold(threshold = 0.0)
            
        -相关系数：特征与特征之间的相关程度
            * -1 <= r <= +1
            * 特征与特征之间相关性很高：
                1）选取其中一个
                2）加权求和
                3）主成分分析
            * 主成分分析
                1.sklearn.decomposition.PCA(n_componts=None)
                2.将数据分解为较低维数空间
                3.n_compents:
                    小数：表示保留百分之多少的信息
                    整数：减少到多少特征
                4.PCA.fit_transform(X)      X:numpy array格式数据[n_samples,n_features]
                5.返回值：转换后指定维数的array
    # Embedded（嵌入式）：算法自动选择特征（特征和目标值之间的关联）
        -决策树：信息熵，信息增益   
        -正则化：l1,l2          
        -深度学习：卷积等

#### eg.皮尔逊相关系数
![img.png](img.png)

#案例一：探究用户对物品类别的喜好细分
    #用户            #物品类别
    user_id         aisle
    # 需要将user_id和aisle放在同一张表中
    # 找到user_id和aisle的关系——交叉表和透视表
    # 特征冗余——PCA降维

# 分类算法
    # 目标值：类别
    # 1.sklearn转换器和预估器
    # 2.KNN算法
    # 3.模型选择和调优
    # 4.朴素贝叶斯算法
    # 5.决策树
    # 6.随机森林

### 1.sklearn转换器和预估器
    # 转换器（特征工程的父类）       实例化 调用fit
    # 估计器estimator
        1.实例化一个estimator
        2.estimator.fit(x_train,y_train)   fit计算    ——调用完毕，模型生成
        3.模型评估：
            1）直接比对真实值和预测值   y_predict = estimator.predict(x_test)
                                    y_test == y_predict
            2）计算准确率    accruacy = estimator.score(x_test,y_test)
### 2.KNN算法(K近邻算法)
    # 根据与已知类别的邻居的距离推断自己的类别
    # 定义： 如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）
            的样本中的大多数属于某一个类别，那么该样本也属于这个类别
    # 如果k = 1，容易受到异常点影响；若k值过大且样本不均衡，可能分类错误
    # 如何确定谁是邻居？
        （欧式）距离公式：如a（a1,a2,a3）,b（b1,b2,b3）
                        [(a1-b1)^2+(a2-b2)^2+(a3-b3)^2]^(-1)
        （曼哈顿）距离公式：|a1-b1|+|a2-b2|+|a3-b3|
        （明可夫斯基）距离公式：
    # 无量纲化处理，标准化
    # sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
    # 案例：鸢尾花种类预测
        1.获取数据
        2.数据集划分
        3.特征工程 标准化
        4.KNN预估器流程
        5.模型评估
    # 优点：简单，易于理解，易于实现，无需训练
    # 缺点：懒惰算法，对测试样本分类的计算量大，内存开销大且必须指定k值，选择不当损失精度
    # 使用场景：小数据场景
### 3.模型选择和调优
    # 1.交叉验证：将拿到的训练数据，分为训练和验证集
    # 2.超参数搜索-网格搜索
    # sklearn.model_selection.GridSearchCV(estimator,param_grid=None,cv=None)
# 案例：预测facebook签到位置
    # file descriptions
      - train.csv,test.csv
        * row_id:id of the check-in event
        * x y:coordinates
        * accuracy:location accuracy
        * time:timestamp
        * place_id:id of the business,this is the target you are predicting
    # 流程分析：
        1）获取数据
        2）数据处理
        目的：
            特征值
            目标值
            a.time -> 年月日时分秒
            b.过滤签到次数少的地点
        3）特征工程：标准化
        4）KNN算法预估流程
        5）模型选择与调优
        6）模型评估

### 4.朴素贝叶斯算法
    # P(A,B) ; P(A|B) ; P(A,B) = P(A)P(B)
    # 贝叶斯公式 P(C|W) = P(W|C)P(C) / P(W)  
    # 注意：w为给定文档的特征值（频数统计，预测文档给定），c为文档类别
    # 朴素：假设特征与特征之间相互独立
    # 朴素贝叶斯算法：朴素+贝叶斯
    # 应用场景：文本分类，情感分析，单词作为特征
    # 思考：若概率求得为零，合理？
        - 所以引入拉普拉斯平滑系数，为防止计算出的分类概率为零
        - P(F1|C) = （Ni+α）/(N+αm)   α一般为1，m为训练文档中统计出的特征词的个数
    # sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
# 案例：20类新闻分类
    1）获取数据
    2）划分数据集
    3）特征工程
        文本特征抽取
    4）朴素贝叶斯预估器流程
    5）模型调优评估
### 5.决策树
    # 如何高效的进行决策？
    # 特征的先后顺序 
    # 信息论基础：
        - 信息：（香农布朗）消除随机不定性的东西
        - 信息熵
    # 原理：
        - 信息熵，信息增益等（信息论）
        - H的专业术语称为信息熵，单位：比特 H（X）= -（从i到n求和）P（Xi）logbP（Xi）
    # 决策树划分依据之一   ——————   信息增益 最大的准则
        - 特征A对训练集D的信息增益g(D,A),定义为集合D的信息熵H（D）与特征A给定条件下D的信息条件熵H（D|A)之差
            * 公式为：g(D,A) = H(D)-H(D|A)
        - 信息增益比：最大的准则
        - 分类树：基尼系数 最小的准则 在sklearn中可以选择划分的默认原则
        - 优势：划分更加细致
    # class sklearn.tree.DecisionTreeClassifier(criterion='gini',max_depth=None,random_state=None)
        - 决策树分类器
        - criterion：默认是gini系数，也可以选择信息增益的熵entropy
        - max_depth：树的深度大小
        - random_state：随机数种子
    # 决策树可视化：sklearn.tree.export_graphviz()能到处DOT格式
        - tree.export_graphviz(estimator,out_file="tree.dot",feature_names['','']
    # 优点：可视化 - 可解释力强
    # 缺点：容易产生过拟合
    # 改进：
        - 减枝cart算法（决策树API已经实现，随机森林参数调优介绍）
        - 随机森林
# 案例：泰坦尼克号生还者预测
    # 流程分析：特征值和目标值
        - 1）获取数据
        - 2）数据处理
            * 缺失值处理
            * 特征值 -> 字典类型
            * 
        - 3）准备特征值 目标值
        - 4）划分数据集
        - 5）特征工程：字典特征抽取
        - 6）决策树预估器流程
        - 7）模型评估
### 6.集成化学习方法：随机森林
    # 集成学习方法：通过建立几个模型组合来解决单一预测问题
        - 原理：生成多个分类器/模型，各自独立地学习和做出预测，这些预测最后结合成组合预测，因此优于任何一个单分类做出的预测
    # 随机森林：一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定
    # 训练集随机 ：bootstrap 随机有放回抽样  N个样本随机有放回抽样N个
    # 特征随机 ：从M个特征中随机抽取m个特征  M >> m       降维
    # class sklearn.ensemble.RandomForestClassifier(n_estimator=10,criterion='gini',
        max_depth=None,bootstrap=True,random_state=None,min_samples_split=2)    
        - 随机森林分类器
        - n_estimator：integer,optional(default=10) 森林里的树木数量
        - criteria：string，可选分割特征的测量方法
        - max_depth最大深度
        - max_features每个决策树的最大特征数量
        - bootstrap: boolean,optional(default = True)是否在构建树时使用放回抽样
        - min_samples_split:节点划分最少样本数
        - min_samples_leaf:叶子节点的最少样本数
    # 总结：
        - 在所有算法中有极好的准确率
        - 处理具有高维度特征的输入样本，而且不需要降维
# 回归和聚类算法
### 线性回归
    # 回归问题：目标值 - 连续性数据
    # 应用：房价预测 销售额预测 金融预测
    # 定义：利用回归方程对一个或多个自变量（特征值）和应变量（目标值）之间关系进行建模
    # 数据挖掘
    # 单自变量单元回归，多自变量多元回归
    # 广义线性模型
        非线性关系？
    # 损失函数：最小二乘法    通过求导优化模型总损失
        - 正规方程 - 直接求解
        - 拓展： 
            1) y = ax^2+bx+c    y'=2ax + b    x = -b/2a
            2) a*b=1  -> b=1/a=a^-1         A*B=E                   B=A^-1
                                                    [[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]]
        - 梯度下降 - 不断试错改进
    # sklearn.linear_model.LinearRegression(fit_intercept = True)
        - 通过正规方程优化
        - fit_intercept是否计算偏置
        - LinearRegression.coef_ 回归系数
        - LinearRegression.intercept:偏置
    # sklearn.linear_model.SGDRegressor(loss="squared_loss",fit_intercept=True,learning_rate='invscaling',eta0=0.01)
        - SGDRegressor类实现了随机梯度下降，它支持不同的loss函数和正则化惩罚项来拟合线性回归模型
        - loss:损失类型         “squared_loss”:普通最小二乘法
        - fit_intercept:是否计算偏置
        - learning_rate：spring,optional
            * 学习率填充（步长）
            * 'constant':eta = eta0
            * 'optimal':eta = 1.0 / (alpha * (t+10))[default]
            * 'invscaling':eta = eta0 / pow(t,power_t)          power_t=0.25存在于父类中
            * 对于一个常数值的学习率来书，可以使用learning_rate=‘constant’,并使用eta0来指定学习率
            * SGDRegressor.coef_        回归系数
            * SGDRegressor.intercept_   偏置

# 案例：波士顿房价预测
    # 流程
        1）获取数据集
        2）划分数据集
        3）特征工程：无量纲化-标准化     
        4）预估器流程     fit（） ---> 模型  coef_  intercept_    
        5）模型评估
    # 回归性能评估    均方误差：sklearn.metrics.mean_squared_error(y_true,y_pred)
    
### 欠拟合与过拟合
    # 训练集表现好，测试集表现不好————过拟合
    # 特征过少欠拟合，特征过多过拟合
    # 欠拟合：
        - 原因：学习到的数据特征过少
        - 解决办法：增加数据特征数量
    # 过拟合：
        - 原因：原始特征过多，存在嘈杂特征，模型过于复杂，因为模型尝试去兼顾各个测试数据点
        - 解决办法：正则化
            * L1正则化：
                1）作用：可以使其中一些参数的值直接为0，删除这个特征的影响
                2）LASSO回归  损失函数 + λ惩罚项（|w|累加 = 删除）
            * L2正则化：损失函数 + λ惩罚项（w^2累加 = 削弱）
                1）作用：可以使其中的一些参数变得很小，接近于零，削弱某个特征的影响
                2）优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象
                3）Ridge回归  损失函数 + λ惩罚项（w^2累加 = 削弱）
                4）加入L2正则化后的损失函数：J（w） = 1/2m (求和从1到m）(Hw(Xi)-Yi)^2+λ（求和从1到n）Wj^2
                                                                            惩罚项：λ（求和从1到n）Wj^2
### 岭回归
    # 建立回归方程时加L2正则化限制，从而解决过拟合的效果
    # sklearn.linear_model.Ridge(alpha=1.0,fit_intercept=True,solver="auto",normalize=Faise)
        - alpha：正则化力度，也叫λ，0<λ<1 or 1<λ<10   惩罚项系数
        - solver：会根据数据自动选择优化方法
            * sag：如果数据集，特征都比较大，选择该随机梯度下降优化
        - normalize：数据是否进行标准化
        - Ridge.coef_
        _ Ridge.intercept_
    # sklearn.linear_model.RidgeCV(_BaseRidgeCV,RegressorMixin)
        - 具有L2正则化的线性回归，可以进行交叉验证
        - coef_回归系数
    # 正则化力度与权重系数成反比例关系
### 分类算法：逻辑回归_二分类
    # 应用：两个类别之间的判断
    # 原理：h（w）= w1x1+w2x2+...+b  逻辑回归的输入是线性回归的输出
    # 激活函数  sigmoid函数   g（θ^T x）= 1/(1+e^(-θ^T x）
    # 总结：线性回归的输出 映射到sigmoid函数上 作为逻辑回归的输入 二分类
    # 假设函数/线性模型
    # 损失函数：
        - 逻辑回归的真实值/预测值：是否属于某个类别
        - 对数似然损失
        - 分开类别
        - 综合完整损失函数
    # 优化损失：梯度下降
    # sklearn.linear_model.LogisticRegression(solver='liblinear',penalty='l2',C=1.0)
        - （有sag方法）solver优化求解方法（默认开源的liblinear实现，内部使用坐标轴下降法来迭代优化损失函数）
        - penalty正则化种类
        - C正则化力度
        - 默认将类别数量少的当作正例
    # SGDClassifer（loss="log",penalty="")
        - 普通随机梯度下降，可通过设置使用平均随机梯度下降法ASGD，可使average=True
![img_2.png](img_2.png)
![img_1.png](img_1.png)
# 案例：癌症分类预测-良/恶性乳腺肿瘤预测
    # 恶性肿瘤为正例
    # 流程分析
        1）获取数据：读取加上names
        2）数据处理：处理缺失值
        3）数据集划分
        4）特征工程：无量纲化——标准化
        5）逻辑回归预估器
        6）模型评估
    # 真的患癌症的，能够被检测出来的概率：召回率（对正样本的区分能力）
    # 分类的评估方法
        - 精确率和召回率
            * 混淆矩阵：预测结果和正确标记之间存在四种组合，构成混淆矩阵（多分类）
                1）TP FN FP TN       True Possitive False Negative
            * 精确率（Precision）：预测结果为正例样本中真实为正例的比例（了解）
            * 召回率（Recall）：真是为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力）
            * F1-score反映了模型稳健性   F1=2TP/(2TP+FN+NP)=2PR/(P+R)
        - sklearn.metrics.classification_report(y_true,y_pred,labels=[],target_names=None)
            * y_true:真实目标值
            * y_pred:估计器预测目标值
            * labels:指定类别对应数字
            * target_name:目标类别名字
            * return:每个类别精确率和召回率
        - 如何衡量样本不均衡下的评估？
            * 
        - ROC曲线和AUC指标
            * TPR=TP/(TP+FN) 所有真实类别为1的样本中，预测类别为1的正例
            * FPR=FP/(FP+TN) 所有真实类别为0的样本中，预测类别为1的正例
            * AUC指标：
            * AUC的概率意义是随机取一对正负样本，正样本得分大于负样本的概率
            * AUC的最小值为0.5，最大值为1，取值越高越好
            * AUC=1，完美分类器，采用这个预测模型时，不管设定什么阈值都能得出完美预测。绝大多数预测的场合，不存在完美分类器。
            * 0.5<AUC<1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
            * 最终AUC的范围在[0.5, 1]之间，并且越接近1越好
        - sklearn.metrics.roc_auc_score(y_true, y_score)
            * 计算ROC曲线面积，即AUC值
            * y_true:每个样本的真实类别，必须为0(反例),1(正例)标记
            * y_score:每个样本预测的概率值
        - 总结：
            * AUC只能用来评价二分类
            * AUC非常适合评价样本不平衡中的分类器性能
![img_3.png](img_3.png)
### 模型保存与加载
    # from sklearn.externals import joblib
        - 保存：joblib.dump(rf, 'test.pkl')
        - 加载：estimator = joblib.load('test.pkl')
### 无监督学习   eg: k-means算法
    # 没有目标值-无监督学习
    # K-means聚类步骤
        1、随机设置K个特征空间内的点作为初始的聚类中心
        2、对于其他每个点计算到K个中心的距离，未知的点选择最近的一个聚类中心点作为标记类别
        3、接着对着标记的聚类中心之后，重新计算出每个聚类的新中心点（平均值）
        4、如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程
    # sklearn.cluster.KMeans(n_clusters=8,init=‘k-means++’)
        * k-means聚类
        * n_clusters:开始的聚类中心数量
        * init:初始化方法，默认为'k-means ++’
        * labels_:默认标记的类型，可以和真实值比较（不是值比较）
![img_4.png](img_4.png) 
# 案例：K—means对Instacart Market用户聚类
    k = 3
### 流程分析
        * 降维后的数据
            1）预估器流程
            2）看结果
            3）模型评估
    
### Kmeans性能评估指标
    # 注：对于每个点i 为已聚类数据中的样本 ，b_i 为i 到其它族群的所有样本的距离最小值，
      a_i 为i 到本身簇的距离平均值。最终计算出所有的样本点的轮廓系数平均值
    # 分析过程（我们以一个蓝1点为例）
        1、计算出蓝1离本身族群所有点的距离的平均值a_i
        2、蓝1到其它两个族群的距离计算出平均值红平均，绿平均，取最小的那个距离作为b_i
        3、根据公式：极端值考虑：如果b_i >>a_i: 那么公式结果趋近于1；如果a_i>>>b_i: 那么公式结果趋近于-1
    # 结论：如果b_i>>a_i:趋近于1效果越好， b_i<<a_i:趋近于-1，效果不好。
           轮廓系数的值是介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优。
    # sklearn.metrics.silhouette_score(X, labels)
        1.计算所有样本的平均轮廓系数
        2.X：特征值
        3.labels：被聚类标记的目标值
    # K-means总结：
        特点分析：采用迭代式算法，直观易懂并且非常实用
        缺点：容易收敛到局部最优解(多次聚类)
        注意：聚类一般做在分类之前
        
![img_5.png](img_5.png)
![img_6.png](img_6.png)
