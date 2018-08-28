package cn.yz0515;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
//import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.VectorUDT;
//import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import scala.Serializable;

import java.util.ArrayList;

/**
 * 参考：
 *  - [关于spark的mllib学习总结（Java版）](http://www.mamicode.com/info-detail-1774414.html)
 *  - [jcseg](https://code.google.com/archive/p/jcseg/)
 */
public class Step02_AtisticalDataJava {
    public static void main(String[] args) {
        //记录开始时间
        long startTime = System.currentTimeMillis();

        SparkSession spark = SparkSession
                .builder()
                .appName("Step02_AtisticalDataJava")
                .master("local[*]")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        /*JavaRDD<String> rateDocument = spark.read().
                textFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rd_test_data.txt").javaRDD();*/
        JavaRDD<String> rateDocument = spark.read().
                textFile("file:///home/zeno/Desktop/DaZhong_Comment_DataAnalysis/data/rate_document").javaRDD();

        /*rateDocument.foreach(new VoidFunction<String>() {
            @Override
            public void call(String line) throws Exception {
                System.out.println(line);
            }
        });*/

        JavaRDD<String> fiveRateDocument = rateDocument.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                try {
                    return Integer.parseInt(s.split("\t")[0]) == 5;
                } catch (Exception e) {
                    return false;
                }
            }
        });

        /*fiveRateDocument.foreach(new VoidFunction<String>() {
            @Override
            public void call(String s) throws Exception {
                System.out.println(s);
            }
        });*/

        JavaRDD<String> negativeRateDocument = rateDocument.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                try {
                    int rate = Integer.parseInt(s.split("\t")[0]);
                    return rate == 1 || rate == 2 || rate == 3;
                } catch (Exception e) {
                    return false;
                }
            }
        });

        /*negativeRateDocument.foreach(new VoidFunction<String>() {
            @Override
            public void call(String s) throws Exception {
                System.out.println(s);
            }
        });*/

        JavaRDD<String> posRateDocument = jsc.parallelize(fiveRateDocument.take((int)negativeRateDocument.count()));

        JavaRDD<String> allRateDocument = posRateDocument.union(negativeRateDocument);

        JavaRDD<Object> wordRDD = allRateDocument.map(new Function<String, Object>() {
            @Override
            public Object call(String s) throws Exception {
                int rata = Integer.parseInt(s.split("\t")[0]);
                int label = rata > 4 ? 1 : 0;   // 好评：1， 差评：0

                String document = s.split("\t")[1];
                ArrayList<String> words = AnaylyzerTools.anaylyzerWords(document);

                RowRateDocument row = new RowRateDocument(label, document, words);

                return row;
            }
        });

         Dataset<Row> wordDF = spark.createDataFrame(wordRDD, RowRateDocument.class);

        // wordDF.select("label", "document", "words").show();

        /**
         * 训练词频矩阵
         */
        HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures");
        Dataset<Row> featurizedTF = hashingTF.transform(wordDF).cache();

        //featurizedTF.select("words", "rawFeatures").show(20, false);

        /**
         * 计算 TF-IDF 矩阵
         */
        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedTF);

        Dataset<Row> rescaledData = idfModel.transform(featurizedTF);

        /**
         * 生成训练集和测试集
         */
        /*Dataset<Row> select = rescaledData.select("label", "features");


        JavaRDD<Object> map1 = select.javaRDD().map(new Function<Row, Object>() {
            @Override
            public Object call(Row row) throws Exception {
                return null;
            }
        });*/

        /*JavaRDD<LabeledPoint> trainDataRdd = rescaledData
                .select("label", "features")
                .javaRDD()
                .map(new Function<Row, LabeledPoint>() {
            @Override
            public LabeledPoint call(Row row) throws Exception {
                int label = Integer.parseInt(row.getAs("label").toString());
                double[] features = row.getAs("features");

                return new LabeledPoint(label, Vectors.dense(features));
            }
        });*/

        JavaRDD<Row> trainDataRdd = rescaledData
                .select("label", "features")
                .javaRDD();

        StructType schema = new StructType(new StructField[] {
            new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        // 转DataFrame
        Dataset<Row> trainData = spark.createDataFrame(trainDataRdd, schema);

        //trainData.select("label", "features").show();

        Dataset<Row>[] randomSplit = trainData.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> trainingData = randomSplit[0];
        Dataset<Row> testData = randomSplit[1];


        NaiveBayesModel NBModel = new NaiveBayes().fit(trainingData);

        Dataset<Row> predictions = NBModel.transform(testData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);

        System.out.println("Test set accuracy = " + accuracy);

        spark.close();

        //记录结束时间
        long endTime=System.currentTimeMillis();
        float excTime=(float)(endTime-startTime)/1000;
        System.out.println("执行时间："+ excTime + "s");

        /* Test set accuracy = 0.7812093023255814
         * 执行时间：3183.829s
         */
    }

    /**
     * RowRateDocument
     */
    public static class RowRateDocument implements Serializable {
        int label;
        String document;
        ArrayList<String> words;

        public RowRateDocument() {

        }

        public RowRateDocument(int label, String document, ArrayList<String> words) {
            this.label = label;
            this.document = document;
            this.words = words;
        }

        public int getLabel() {
            return label;
        }

        public void setLabel(int label) {
            this.label = label;
        }

        public String getDocument() {
            return document;
        }

        public void setDocument(String document) {
            this.document = document;
        }

        public ArrayList<String> getWords() {
            return words;
        }

        public void setWords(ArrayList<String> words) {
            this.words = words;
        }

        @Override
        public String toString() {
            return "RowRateDocument{" +
                    "label=" + label +
                    ", document='" + document + '\'' +
                    ", words=" + words +
                    '}';
        }
    }
}
