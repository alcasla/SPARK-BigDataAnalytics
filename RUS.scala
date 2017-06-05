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
import org.apache.spark.mllib.feature._			//feature selection
import srio.org.apache.spark.mllib.sampling._		//Sara Del Río, ROS and RUS sampling methods

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
val nToSelect = 10
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
val reducedRos = runRUS.apply(reduced, 1.0, 0.0)
val trainRUS = reducedRos.count()
val reduceTestdRos = runRUS.apply(reduced, 1.0, 0.0)
val testRUS = reducedRos.count()


//***************
println("\nHello, world!")
println("\nFIN\n")


System.exit(0)