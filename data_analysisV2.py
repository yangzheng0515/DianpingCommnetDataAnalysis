# /home/zeno/anaconda3/bin/pip install jieba
# /home/zeno/anaconda3/bin/pip install findspark
# pyspark --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/home/zeno/anaconda3/bin/python

import findspark
findspark.init()
from pyspark import *

sc = SparkContext()

# originData = sc.textFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rd_test_data.txt")
originData = sc.textFile("file:///home/zeno/桌面/DaZhong_Comment_DataAnalysis/data/rate_document/part-00000")
originDistinctData = originData.distinct()

rateDocument = originDistinctData.map(lambda line : line.split('\t')).filter(lambda line : len(line) == 2)

# 1为好评，0为差评
lableDocument = rateDocument.map(lambda line : [1, line[1]] if int(line[0]) > 4 else [0, line[1]])

negLableDocument = lableDocument.filter(lambda line : int(line[0]) == 0)
goodLableDocument = lableDocument.filter(lambda line : int(line[0]) == 1)

posLableDocument = sc.parallelize(goodLableDocument.take(negLableDocument.count())).repartition(1)
allLableDocument = negLableDocument.union(posLableDocument)
allLableDocument.repartition(1)
lable = allLableDocument.map(lambda s : s[0])
document = allLableDocument.map(lambda s: s[1])

import jieba
words = document.map(lambda w:"/".join(jieba.cut_for_search(w))).map(lambda line: line.split("/"))

from pyspark.mllib.feature import HashingTF, IDF
hashingTF = HashingTF()
tf = hashingTF.transform(words)
tf.cache()

idfModel = IDF().fit(tf)
tfidf = idfModel.transform(tf)


from pyspark.mllib.regression import LabeledPoint
zipped = lable.zip(tfidf)
data = zipped.map(lambda line:LabeledPoint(line[0],line[1]))
training, test = data.randomSplit([0.6, 0.4], seed = 0)

from pyspark.mllib.classification import NaiveBayes
NBmodel = NaiveBayes.train(training, 1.0)
predictionAndLabel = test.map(lambda p : (NBmodel.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda x: 1.0 if x[0] == x[1] else 0.0).count() / test.count()
# 0.6707555665973106

# yourDocument=input("输入待分类的评论：")
yourDocument = """那道黄金饺主食太肯爹了，每个饺子比小馄炖还小，炸过的，吃起来软塔塔的，里面就点萝卜丝，小小的12个，58元，大家千万别上当啊，菜谱里没有的，点菜时服务员竭力推荐的，千万别上当！??"""
yourwords = "/".join(jieba.cut_for_search(yourDocument)).split("/")
yourtf = hashingTF.transform(yourwords)
yourtfidf = idfModel.transform(yourtf) 
print('NaiveBayes Model Predict:', NBmodel.predict(yourtfidf))

# ------------------------------------------------------------------------------------------------------------------ #

"""
- [基于 Spark 的文本情感分析，以《疯狂动物城》为例](https://developer.huawei.com/ict/forum/thread-49583.html)
- [pyspark依赖部署](https://luzhijun.github.io/2017/12/10/pyspark%E4%BE%9D%E8%B5%96%E9%83%A8%E7%BD%B2/)
- [如何将PySpark导入Python](https://blog.csdn.net/sinat_26599509/article/details/51895999)


```
>>> from pyspark.mllib.feature import HashingTF, IDF
```
RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility
造成这个问题产生的原因很简单，就是各种库之间的版本不匹配，只需要把numpy的版本降到1.14.5就可以了
sudo pip uninstall numpy
sudo pip install numpy==1.14.5


```
>>> tf = hashingTF.transform(words)

'PipelinedRDD' object has no attribute '_jdf'
```
{"pyspark.mllib": "pyspark.rdd.RDD", "pyspark.ml": "pyspark.sql.DataFrame"}
https://stackoverflow.com/questions/39643185/pipelinedrdd-object-has-no-attribute-jdf


LabeledPoint
[spark机器学习笔记：（四）用Spark Python构建分类模型（上）](https://blog.csdn.net/u013719780/article/details/51784452)

"""
