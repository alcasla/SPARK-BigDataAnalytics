:load ./keelParser.scala

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation._
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.feature.MDLPDiscretizer	//discretization
import org.apache.spark.mllib.feature._				//feature selection
import srio.org.apache.spark.mllib.sampling._			//Sara Del Río, ROS and RUS sampling methods
import org.apache.spark.mllib.tree.DecisionTree 		//decision tree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

//Switch LogLevel to avoid info messages
sc.setLogLevel("ERROR")

//Load data in persist object (not wait to run)
val converter = new KeelParser(sc, "ECBDL14_mbd/ecbdl14.header")
val train = sc.textFile("ECBDL14_mbd/ecbdl14tra.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist
val test = sc.textFile("ECBDL14_mbd/ecbdl14tst.data", 10).map(line=>converter.parserToLabeledPoint(line)).persist

//discretizer preparation
val categoricalFeat: Option[Seq[Int]] = None
val nBins= 15
val maxByPart= 10000
println("*** Discretizationmethod: Fayyad discretizer(MDLP)")
println("*** Number of bins: " + nBins)
val discretizer= MDLPDiscretizer.train(train, categoricalFeat, nBins, maxByPart)

//train and test discretization
val discrete = train.map(i=> LabeledPoint(i.label, discretizer.transform(i.features))).cache()
val trainNRows = discrete.count()
//discrete.first()		//print a discretised instance values - train

val discreteTest= test.map(i=> LabeledPoint(i.label, discretizer.transform(i.features))).cache()
val testNRows = discreteTest.count()
//discreteTest.first()		//print a discretised instance values - test

//rank feature importances
val criterion = new InfoThCriterionFactory("mrmr")
val nToSelect = 160
val nPartitions = 6
println("*** FS criterion: " + criterion.getCriterion.toString)
println("*** Number of features to select: " + nToSelect)
println("*** Number of partitions: " + nPartitions)
val featureSelector = new InfoThSelector(criterion, nToSelect, nPartitions).fit(discrete)

//reduce features for train and test
val reduced = discrete.map(i => LabeledPoint(i.label, featureSelector.transform(i.features))).cache()
reduced.count()
reduced.first()

val reducedTest = discreteTest.map(i => LabeledPoint(i.label, featureSelector.transform(i.features))).cache()
reducedTest.count()
reducedTest.first()

//RUS apply on train and test
val reducedRUS = runRUS.apply(reduced, 1.0, 0.0)
val trainRUSins = reducedRUS.count()
val reduceTestdRUS = runRUS.apply(reducedTest, 1.0, 0.0)
val testRUSins = reduceTestdRUS.count()


//DecisionTree params
val numClasses = converter.getNumClassFromHeader()		//num classes from featureSelector
val categoricalFeaturesInfo = Map[Int,Int]()
val impurity = "entropy"
val maxDepth = 10
val maxBins = 100

//train and predict using DecisionTree
val model = DecisionTree.trainClassifier(reducedRUS, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
val predictions = reduceTestdRUS.map{ point=>
	val prediction = model.predict(point.features)
	(prediction, point.label)
}.persist

//metrics
val metrics = new MulticlassMetrics(predictions)
val precision = metrics.precision
val cm = metrics.confusionMatrix

val binaryMetrics = new BinaryClassificationMetrics(predictions)
val AUC = binaryMetrics.areaUnderROC


//***************
println("\nHello, world!")
println("\nFIN\n")


System.exit(0)