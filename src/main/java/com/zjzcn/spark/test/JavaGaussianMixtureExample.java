package com.zjzcn.spark.test;

import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off$
import org.apache.spark.sql.SparkSession;


public class JavaGaussianMixtureExample {

    public static void main(String[] args) {

        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaGaussianMixtureExample")
                .master("local")
                .getOrCreate();

        // $example on$
        // Loads data
        Dataset<Row> dataset = spark.read().format("libsvm").load("src/main/resources/sample_kmeans_data.txt");

        // Trains a GaussianMixture model
        GaussianMixture gmm = new GaussianMixture()
                .setK(2);
        GaussianMixtureModel model = gmm.fit(dataset);

        // Output the parameters of the mixture model
        for (int i = 0; i < model.getK(); i++) {
            System.out.printf("Gaussian %d:\nweight=%f\nmu=%s\nsigma=\n%s\n\n",
                    i, model.weights()[i], model.gaussians()[i].mean(), model.gaussians()[i].cov());
        }
        // $example off$

        spark.stop();
    }
}