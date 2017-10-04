package com.zjzcn.spark.test;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GridInfoTest {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("GridInfoTest")
                .master("local")
                .config("spark.some.config.option", "some-value")
                .getOrCreate();

        Dataset<Row> df = spark.read().csv("src/main/resources/grid_info.csv");

        df.show();

        df.printSchema();

        df.select("_c0").show();

        df.createOrReplaceTempView("grid_info");

        // 网格所有面积和（单位：平方公里）
        spark.sql("select sum(_c12) from grid_info").show();
    }
}
