:load ./keelParser.scala

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import scala.collection.mutable.ListBuffer

sc.setLogLevel("ERROR")

val converter = new KeelParser(sc, "ECBDL14_mbd/ecbdl14.header")
val train = sc.textFile("ECBDL14_mbd/ecbdl14tra.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist
val test = sc.textFile("ECBDL14_mbd/ecbdl14tst.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist

val observations = train.map(_.features)
val summary:MultivariateStatisticalSummary = Statistics.colStats(observations)

summary.mean(0)
summary.variance(0)
summary.max(0)
summary.min(0)

object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("Hello, world!")
    println("\n\nFIN\n")
  }
}

HelloWorld.main(Array())

System.exit(0)
