:load ./keelParser.scala

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation._
import scala.collection.mutable.ListBuffer

//Switch LogLevel to avoid info messages
sc.setLogLevel("ERROR")

//Load data in persist object (not wait to run)
val converter = new KeelParser(sc, "ECBDL14_mbd/ecbdl14.header")
val train = sc.textFile("ECBDL14_mbd/ecbdl14tra.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist
val test = sc.textFile("ECBDL14_mbd/ecbdl14tst.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist

//RandomForest params
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 10
val featureSubsetStrategy = "auto"		//algorithm choice
val impurity = "entropy"	//"gini"
val maxDepth = 12			//4
val maxBins = 32

//train and test RandomForest model
val model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, 
	impurity, maxDepth, maxBins) 
val predsAndLabels = test.map { point =>
		val prediction = model.predict(point.features)
		(prediction, point.label)
	}.persist

//metrics
val metrics = new MulticlassMetrics(predsAndLabels)
val cm = metrics.confusionMatrix

var pos = cm.apply(0, 0)/(cm.apply(0, 0)+ cm.apply(0, 1))
var neg = cm.apply(1, 1)/(cm.apply(1, 0)+ cm.apply(1, 1))
var tprTnr = pos * neg


//***************
println("\nHello, world!")
println("\nFIN\n")


System.exit(0)