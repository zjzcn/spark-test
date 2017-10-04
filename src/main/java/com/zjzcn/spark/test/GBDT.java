package com.zjzcn.spark.test;


import com.alibaba.fastjson.JSON;
import com.google.common.collect.Lists;
import com.xiaoleilu.hutool.util.StrUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.*;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.col;

public class GBDT {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("GBDT")
                .master("local")
                .getOrCreate();

        Dataset<Row> rawData = spark
                .read()
                .option("delimiter", "\t")
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/pred_data_200.txt");

        Dataset<Row> data = rawData.filter(col("viewed_idx").lt(50));
        data.show(5);
        data.printSchema();

        Dataset<Row> trainData = data.filter(col("day_no").gt(80))
                .filter(col("day_no").leq(145));
        trainData.show(5);

        Dataset<Row> testData = data.filter(col("day_no").gt(145));
        testData.show(5);

        List<StringIndexerModel> stringIndexerStages = Arrays.stream(data.columns())
                .filter(col -> isCat(col))
                .map(col -> new StringIndexer()
                        .setInputCol(col)
                        .setOutputCol(catNewCol(col))
                        .fit(data)
                ).collect(Collectors.toList());

        List<String> featureCols = Arrays.stream(data.columns())
                .filter(col -> !Lists.newArrayList("log_id", "user_id", "restaurant_id", "is_click", "is_buy").contains(col))
                .map(col -> catNewCol(col))
                .collect(Collectors.toList());

        System.out.println(featureCols);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols.toArray(new String[featureCols.size()]))
                .setOutputCol("feature");

        GBTClassifier gbt = new GBTClassifier()
                .setLabelCol("is_click")
                .setFeaturesCol("feature")
                .setMaxBins(40)
                .setMaxIter(10);

        List<PipelineStage> stages = new ArrayList<>();
        stages.addAll(stringIndexerStages);
        stages.add(assembler);
        stages.add(gbt);

        Pipeline pipeline = new Pipeline()
                .setStages(stages.toArray(new PipelineStage[stages.size()]));

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(gbt.maxIter(), new int[] {10})
                .addGrid(gbt.maxDepth(), new int[] {10})
                .build();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("is_click")
                .setRawPredictionCol("prediction")
                .setMetricName("areaUnderROC");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(2);

        CrossValidatorModel cvModel = cv.fit(trainData);
//        PipelineModel pipelineModel = pipeline.fit(trainData);

        double[] featureImportances = ((GBTClassificationModel) ((PipelineModel)cvModel.bestModel()).stages()[stages.size() - 1]).featureImportances().toArray();

        Dataset<Row> predictions = cvModel.transform(testData);
        long is_click = predictions.filter(col("is_click").equalTo(1)).count();
        System.out.println("is_click count:" + is_click);

        long prediction = predictions.filter(col("prediction").equalTo(1)).count();
        System.out.println("prediction count:" + prediction);

        predictions.select(col("rawPrediction"), col("probability"), col("prediction"), col("is_click")).show();

        predictions.show(5);

        double auc = evaluator.evaluate(predictions);

        Map<String, Double> featureImportanceMap = new HashMap<>();
        for (int i = 0; i < featureCols.size(); i++) {
            featureImportanceMap.put(featureCols.get(i), featureImportances[i]);
        }

        printResult(trainData, testData, 10, 10, 2, auc, featureImportanceMap);

        spark.stop();
    }

    private static boolean isCat(String col) {
        return col.endsWith("_cat");
    }

    private static String catNewCol(String col) {
        return isCat(col) ? col + "_idx" : col;
    }
    
    private static void printResult(Dataset<Row> trainData, Dataset<Row> testData, int maxIter, int maxDepth, int numFolds, double auc, Map<String, Double> featureImportancesMap) {
        String output = "\n=====================================================================\n" +
                "TrainData count: {}\n" +
                "TestData count: {}\n" +
                "=====================================================================\n" +
                "Param maxIter = {}\n" +
                "Param maxDepth = {}\n" +
                "Param numFolds = {}\n" +
                "=====================================================================\n" +
                "TestData AUC = {}\n" +
                "=====================================================================\n" +
                "GBT features importances:\n " +
                "{}\n" +
                "=====================================================================\n";

        Map<String, Double> sortedMap = sortByValue(featureImportancesMap);
        output = StrUtil.format(output, trainData.count(), testData.count(), maxIter, maxDepth, numFolds, auc, JSON.toJSONString(sortedMap, true));
        System.out.println(output);
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
        Map<K, V> result = new LinkedHashMap<>();
        map.entrySet()
                .stream()
                .sorted(Map.Entry.<K, V>comparingByValue().reversed())
                .forEach(e -> result.put(e.getKey(), e.getValue()));

        return result;
    }
}
