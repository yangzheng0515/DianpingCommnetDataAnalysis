package cn.yz0515

import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapred.TextInputFormat
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 第一步： 数据预处理和抽取
  * 从comment_data.csv数据集中抽取评分和评价内容数据作为一个新的数据集保存到rate_document
  */
object Step01_ExtractDateDocument {
  /**
    * spark读取gbk文件
    * @param sc
    * @param path
    * @return
    */
  def transfer(sc:SparkContext, path:String):RDD[String] = {
    // 将value的字节码按照GBK的方式读取变成字符串，运行之后能够正常显示
    sc.hadoopFile(path, classOf[TextInputFormat], classOf[LongWritable], classOf[Text], 1)
      .map(p => new String(p._2.getBytes, 0, p._2.getLength, "GBK"))
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Step01_ExtractDateDocument").setMaster("local[4]")
    val sc = new SparkContext(conf)

    // textFile 默认只支持Utf-8格式，通过封装后的方法读取GBK文件,并讲每一行数据以字符串格式返回(RDD[String])
    val lines = transfer(sc, "file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/comment_data.csv")

    val rate_document = lines.map(line => {
      val rating = line.split(",")(2)
      val content_review = line.split(",")(10)
      rating + "\t" + content_review
    })

    // 过滤掉没有评论内容或者没有评分的数据
    val new_rate_document = rate_document.filter(line => {
      line.split("\t").length == 2 && !"Rating".equals(line.split("\t")(0))
    })

    // new_rate_document.take(10).foreach(println(_))

    // 保存改数据
    new_rate_document.saveAsTextFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rate_document")

    sc.stop()


    val rate_document = transfer(sc, "file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/comment_data.csv")
      .map(line => {
      val rating = line.split(",")(2)
      val content_review = line.split(",")(10)
      rating + "\t" + content_review
    })


  }
}
