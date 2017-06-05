:load ./keelParser.scala

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import scala.collection.mutable.ListBuffer

//Switch LogLevel to avoid info messages
sc.setLogLevel("ERROR")

//Load data in persist object (not wait to run)
val converter = new KeelParser(sc, "ECBDL14_mbd/ecbdl14.header")
val train = sc.textFile("ECBDL14_mbd/ecbdl14tra.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist
val test = sc.textFile("ECBDL14_mbd/ecbdl14tst.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist

//model params
val numClasses = converter.getNumClassFromHeader()
val categoricalFeaturesInfo = Map[Int,Int]()
val impurity = "entropy"
val maxDepth = 10
val maxBins = 100

//train and predict using DecisionTree
val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val predictions = test.map{ point=>
	val prediction = model.predict(point.features)
	(prediction, point.label)
}.persist

//metrics
val metrics = new MulticlassMetrics(predictions)
val cm = metrics.confusionMatrix

var pos = cm.apply(0, 0)/(cm.apply(0, 0)+ cm.apply(0, 1))
var neg = cm.apply(1, 1)/(cm.apply(1, 0)+ cm.apply(1, 1))
var tprTnr = pos * neg


//***************
println("\nHello, world!")
println("\nFIN\n")


System.exit(0)
