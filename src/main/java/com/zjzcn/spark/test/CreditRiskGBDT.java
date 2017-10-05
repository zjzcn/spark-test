package com.zjzcn.spark.test;

import com.alibaba.fastjson.JSON;
import com.xiaoleilu.hutool.util.StrUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class CreditRiskGBDT {

    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("CreditRiskGBDT")
                .master("local")
                .getOrCreate();

        Dataset<Row> rawData = spark
                .read()
                .option("delimiter", ",")
                .option("inferSchema", "true")
                .csv("src/main/resources/german.txt");

        String[] cols = new String[] {
                "creditability",
                "balance",
                "duration",
                "history",
                "purpose",
                "amount",
                "savings",
                "employment",
                "instPercent",
                "sexMarried",
                "guarantors",
                "residenceDuration",
                "assets",
                "age",
                "concCredit",
                "apartment",
                "credits",
                "occupation",
                "dependents",
                "hasPhone",
                "foreign"
        };


        String[] rawCols = rawData.columns();
        for(int i=0; i<rawCols.length; i++) {
            rawData = rawData.withColumnRenamed(rawCols[i], cols[i]);
        }

        int splitSeed = 5043;
        Dataset<Row>[] split = rawData.randomSplit(new double[]{0.7, 0.3}, splitSeed);
        Dataset<Row> trainData = split[0];
        Dataset<Row> testData = split[1];

        String[] featureCols = Arrays.copyOfRange(cols, 1, cols.length);

        System.out.println(featureCols);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        GBTClassifier gbt = new GBTClassifier()
                .setLabelCol("creditability")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, gbt});

        ParamMap[] paramGrid = new ParamGridBuilder()
//                .addGrid(gbt.maxBins(), new int[]{25, 31})
                .addGrid(gbt.maxIter(), new int[] {15})
                .addGrid(gbt.maxDepth(), new int[] {5})
                .build();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("creditability");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        CrossValidatorModel cvModel = cv.fit(trainData);

        double[] featureImportances = ((GBTClassificationModel) ((PipelineModel)cvModel.bestModel()).stages()[1]).featureImportances().toArray();

        Dataset<Row> predictions = cvModel.transform(testData);

        predictions.show(5);

        double trainAuc = evaluator.evaluate(cvModel.transform(trainData));
        double testAuc = evaluator.evaluate(predictions);

        Map<String, Double> featureImportanceMap = new HashMap<>();
        for (int i = 0; i < featureCols.length; i++) {
            featureImportanceMap.put(featureCols[i], featureImportances[i]);
        }

        printResult(trainData, testData, 10, 10, 2, trainAuc, testAuc, featureImportanceMap);

        spark.stop();
    }


    private static void printResult(Dataset<Row> trainData, Dataset<Row> testData, int maxIter, int maxDepth, int numFolds, double auc, double testAuc, Map<String, Double> featureImportancesMap) {
        String output = "\n=====================================================================\n" +
                "TrainData count: {}\n" +
                "TestData count: {}\n" +
                "=====================================================================\n" +
                "Param maxIter = {}\n" +
                "Param maxDepth = {}\n" +
                "Param numFolds = {}\n" +
                "=====================================================================\n" +
                "TrainData AUC = {}\n" +
                "TestData AUC = {}\n" +
                "=====================================================================\n" +
                "GBT features importances:\n " +
                "{}\n" +
                "=====================================================================\n";

        Map<String, Double> sortedMap = sortByValue(featureImportancesMap);
        output = StrUtil.format(output, trainData.count(), testData.count(), maxIter, maxDepth, numFolds, auc, testAuc, JSON.toJSONString(sortedMap, true));
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
