package cn.yz0515

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{HashingTF, IDF, LabeledPoint}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._

/**
  * 统计数据基本信息
  * 本文得到，总共有83753条数据，五分的数据有43168条，4 分、3 分、2 分、1 分的数据分别有27086条，10184条，1438条，1876条。
  * 打五分的毫无疑问是好评；考虑到不同人对于评分的不同偏好，对于打四分的数据，本文无法得知它是好评还是坏评；对于打三分及三分以下的是坏评。
  * 下面就可以将带有评分数据转化成为好评数据和坏评数据，为了提高计算效率，本文将其重新分区。
  * 参考：
  *  - [基于 Spark 的文本情感分析，以《疯狂动物城》为例](https://developer.huawei.com/ict/forum/thread-49583.html)
  *  - [Spark入门：特征抽取： TF-IDF — spark.ml](http://dblab.xmu.edu.cn/blog/1261-2/)
  *  - [Spark MLlib实现的中文文本分类–Naive Bayes](http://lxw1234.com/archives/2016/01/605.htm)
  */
object Step02_AtisticalData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Step02_AtisticalDataV2").master("local[*]").getOrCreate()
    //val conf = new SparkConf().setAppName("Step02_AtisticalData").setMaster("local[*]")
    //val sc = new SparkContext(conf)
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext

    //val rateDocument = sc.textFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rate_document")

    /* 测试，数据集只取了1000行 */
    val rateDocument = sc.textFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rd_test_data.txt")

    rateDocument.persist()

    /*rateDocument.take(10).foreach(println(_))
    println(rateDocument.count())  // 83753*/

    // 五分的数据
    val fiveRateDocument = rateDocument.filter(line =>{
      try {
        line.split("\t")(0).toInt == 5
      } catch {
        case e: NumberFormatException => false
      }
    })

    /*fiveRateDocument.take(10).foreach(println(_))
    println(fiveRateDocument.count()) // 43168*/

    /**
      * 打印1到5分数据的数量
      */
    /*for (n <- 1 to 5) {
      val NRateDocument = rateDocument.filter(_.split("\t")(0).toInt == n)
      println(n + "分的数据有" + NRateDocument.count + "条")
    }*/

    /**
      * 合并负样本数据
      */
    val oneRateDocument = rateDocument.filter(_.split("\t")(0).toInt == 1)
    val twoRateDocument = rateDocument.filter(_.split("\t")(0).toInt == 2)
    val threeRateDocument = rateDocument.filter(_.split("\t")(0).toInt == 3)

    // 负样本数据
    val negativeRateDocument = oneRateDocument.union(twoRateDocument).union(threeRateDocument).cache()
    // println(negativeRateDocument.count()) // 13498

    /**
      * 生成训练数据集
      * 好评和坏评分别有 43168 条和 13498 条，属于非平衡样本的机器模型训练。这里只取部分好评数据，好评和坏评的数量一样，这样训练的正负样本就是均衡的。最后把正负样本放在一起，并把分类标签和文本分开，形成训练数据集。
      */
    val posRateDocument = sc.parallelize(fiveRateDocument.take(negativeRateDocument.count.toInt))

    // 训练集
    val allRateDocument = posRateDocument.union(negativeRateDocument)
    allRateDocument.persist()
    // allRateDocument.repartition(1) // why
    //val rate: RDD[String] = allRateDocument.map(_.split("\t")(0))
    //val document: RDD[String] = allRateDocument.map(_.split("\t")(1))

    /**
      * 分词
      */
    // val words: RDD[Array[AnyRef]] = document.map(AnaylyzerTools.anaylyzerWords(_).toArray())
    val wordRDD = allRateDocument.map(rd => {
      val rate = rd.split("\t")(0).toInt
      val document = rd.split("\t")(1)
      val words = AnaylyzerTools.anaylyzerWords(document).toArray
      Row(rate, document, words)
    })
    wordRDD.cache()

    /**
      * RDD转换成DataFrame
      */
    val schema = StructType(Array(
      StructField("rate", IntegerType, true),
      StructField("document", StringType, true),
      StructField("words", ArrayType(StringType), false)
    ))

    val wordDF = sqlContext.createDataFrame(wordRDD, schema).cache()
    //wordDF.show()

    /**
      * 训练词频矩阵
      */
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
    // 用HashingTF的transform（）方法把句子哈希成特征向量
    val featurizedTF = hashingTF.transform(wordDF).cache()

    // featurizedTF.select("words", "rawFeatures").show(20, false)
    /*
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|words                                                     |rawFeatures
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[这个, 价位, ，, 这个, 口味, 就, 那样, ，, 说不上, 特别, 好吃] | (262144,[12231,17078,37436,54992,67849,85364,112276,153025,154679],[1.0,1.0,1.0,1.0,1.0,2.0,1.0,2.0,1.0]) |

|[吃完, 会, 感觉, 很, 舒服, 的, 地方, ，, 这个, 季节, 最爱, 炖汤, 和, 滋补, 的, 菜式, ，, 朋友, 来, 广州, 首推, 的, 地方, 。]|(262144,[5224,12859,13599,26081,30776,37083,45248,52547,85364,101376,132859,137444,140763,145803,153025,167900,177554,186651,217856,238030],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0]) |

...

以这行为例，262144代表哈希表的桶数，[12231,17078,37436,54992,67849,85364,112276,153025,154679]代表着单词的哈希值，[1.0,1.0,1.0,1.0,1.0,2.0,1.0,2.0,1.0]为对应单词的出现次数
*/

    /**
      * 计算 TF-IDF 矩阵
      */
    /* 调用IDF方法来重新构造特征向量的规模，生成的idf是一个Estimator，在特征向量上应用它的fit（）方法，会产生一个IDFModel。 */
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedTF)
    /* 同时，调用IDFModel的transform方法，可以得到每一个单词对应的TF-IDF 度量值。 */
    val rescaledData = idfModel.transform(featurizedTF)
    //rescaledData.select("words", "rawFeatures", "features").take(3).foreach(println)
    //rescaledData.select("words", "features").show(20, false)
/*
|[这个, 价位, ，, 这个, 口味, 就, 那样, ，, 说不上, 特别, 好吃] | (262144,[12231,17078,37436,54992,67849,85364,112276,153025,154679],[2.1893967487159727,1.1084840371472637,5.539300835990577,1.2352357427864076,2.3204250111223765,0.2958972069507029,4.286537867495209,4.6012447676523935,4.286537867495209]) |
 [2.1893967487159727,1.1084840371472637,...]代表单词的TF-IDF值
 */



    /**
      * 生成训练集和测试集
      */
//    var trainDataRdd = rescaledData.select($"category",$"features").map {
//      case Row(label: String, features: Vector) =>
//        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
//    }

    /* 将上面的数据转换成Bayes算法需要的格式 */
    import spark.implicits._
    val trainDataRdd = rescaledData.select("rate", "features").map { case Row(label: Int, features: Vector) =>
      LabeledPoint(label.toInt, Vectors.dense(features.toArray))
    }

    val Array(trainingData, testData) = trainDataRdd.randomSplit(Array(0.6, 0.4), seed = 1234L)

    trainingData.take(3).foreach(println)


    /**
      * 训练贝叶斯分类模型
      */
    val NBmodel = new NaiveBayes().fit(trainingData)

    //对测试数据集使用训练模型进行分类预测
    val predictions = NBmodel.transform(testData)
    // predictions.show(10)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    sc.stop()
  }

}
