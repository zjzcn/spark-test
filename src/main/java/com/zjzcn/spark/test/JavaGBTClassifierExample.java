package com.zjzcn.spark.test;


import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

// $example on$
// $example off$

public class JavaGBTClassifierExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaGradientBoostedTreeClassifierExample")
                .master("local")
                .getOrCreate();

        // $example on$
        // Load and parse the data file, converting it to a DataFrame.
        Dataset<Row> data = spark
                .read()
                .format("libsvm")
                .load("src/main/resources/sample_libsvm_data.txt");

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a GBT model.
        GBTClassifier gbt = new GBTClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                .setMaxIter(10);

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        // Chain indexers and GBT in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {labelIndexer, featureIndexer, gbt, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5);

        // Select (prediction, true label) and compute test error.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        GBTClassificationModel gbtModel = (GBTClassificationModel)(model.stages()[2]);
        System.out.println("Learned classification GBT model:\n" + gbtModel.toDebugString());
        // $example off$
        try {
            gbtModel.save("model");
        } catch (IOException e) {
            e.printStackTrace();
        }
        spark.stop();
    }
}
