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

fiveRateDocument = rateDocument.filter(lambda line : int(line[0]) == 5)
oneRateDocument = rateDocument.filter(lambda line : int(line[0]) == 1)
twoRateDocument = rateDocument.filter(lambda line : int(line[0]) == 2)
threeRateDocument = rateDocument.filter(lambda line : int(line[0]) == 3)

negRateDocument = oneRateDocument.union(twoRateDocument).union(threeRateDocument)
negRateDocument.repartition(1)

posRateDocument = sc.parallelize(fiveRateDocument.take(negRateDocument.count())).repartition(1)
allRateDocument0 = negRateDocument.union(posRateDocument)
allRateDocument.repartition(1)
# rate = allRateDocument.map(lambda s : ReduceRate(s[0]))
rate = allRateDocument.map(lambda s : s[0])
document = allRateDocument.map(lambda s: s[1])

import jieba
words = document.map(lambda w:"/".join(jieba.cut_for_search(w))).map(lambda line: line.split("/"))

from pyspark.mllib.feature import HashingTF, IDF
hashingTF = HashingTF()
tf = hashingTF.transform(words)
tf.cache()

idfModel = IDF().fit(tf)
tfidf = idfModel.transform(tf)


from pyspark.mllib.regression import LabeledPoint
zipped = rate.zip(tfidf)
data = zipped.map(lambda line:LabeledPoint(line[0],line[1]))
training, test = data.randomSplit([0.6, 0.4], seed = 0)

from pyspark.mllib.classification import NaiveBayes
NBmodel = NaiveBayes.train(training, 1.0)
predictionAndLabel = test.map(lambda p : (NBmodel.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda x: 1.0 if x[0] == x[1] else 0.0).count() / test.count()



# yourDocument=input("输入待分类的评论：")
yourDocument = """无论是环境，出品，服务，都棒棒哒????????性价比高吖"""
yourwords = "/".join(jieba.cut_for_search(yourDocument)).split("/")
yourtf = hashingTF.transform(yourwords)
yourtfidf = idfModel.transform(yourtf) 
print('NaiveBayes Model Predict:', NBmodel.predict(yourtfidf))



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
