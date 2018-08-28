package cn.yz0515;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class Step01_ExtractDateDocumentJava {

    /**
     * spark读取gbk文件
     * @param jsc
     * @param path
     * @return
     */
    public static JavaRDD<String> transfer(JavaSparkContext jsc, String path) {
        return jsc.hadoopFile(path, TextInputFormat.class, LongWritable.class, Text.class, 1)
                .map(new Function<Tuple2<LongWritable,Text>, String>() {
                    @Override
                    public String call(Tuple2<LongWritable, Text> line) throws Exception {
                        return new String(line._2.getBytes(), 0, line._2.getLength(), "GBK");
                    }
                });
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("Step01_ExtractDateDocumentJava")
                .master("local[*]")
                .getOrCreate();
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        JavaRDD<String> lines = transfer(jsc, "file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/comment_data.csv");

        JavaRDD<String> rate_document = lines.map(new Function<String, String>() {
            @Override
            public String call(String line) throws Exception {
                String rate = line.split(",")[2];
                String document = line.split(",")[10];
                return rate + "\t" + document;
            }
        });

        rate_document = rate_document.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String line) throws Exception {
                return line.split("\t").length == 2 && ! "Rating".equals(line.split("\t")[0]);
            }
        });

        // 只取前1000行进行分析
        JavaRDD<String> less_rate_document = jsc.parallelize(rate_document.take(1000));
        less_rate_document.saveAsTextFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rate_document_java");
    }
}