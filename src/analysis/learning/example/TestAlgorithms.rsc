module analysis::learning::example::TestAlgorithms

import analysis::learning::DataPoints;
import analysis::learning::KMeans;
import analysis::learning::KNearestNeighbors;
import analysis::learning::DecisionTrees;
import analysis::learning::BootstrapAggregation;
import analysis::learning::RandomForest;

import util::Math;
import IO;
import Set;

void testAlgorithms(set[Point] trainers, set[Point] tests) {
  println("Selected <size(trainers)> data points for training, and <size(tests)> are left for testing");
  
  println("Training kMeans");
  clusters = kmeans(simplePartition([*trainers], 5));
  println("Testing kMeans");
  correct = (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := respond(clusters, t));
  println("Accuracy of kMeans is <correct> out of <size(tests)> = <percent(correct, size(tests))>%");
  
  println("Testing kNN directly (it requires no explicit training other than collecting the data)");
  correct = (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := respond(trainers, t, 5));
  println("Accuracy of kNN is <correct> out of <size(tests)> = <percent(correct, size(tests))>%");
  
  println("Training Decision Tree");
  tree = buildDecTree(trainers, maxDepth=5, minSize=10);
  println("Testing the Decision Tree");
  correct = (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := respond(tree, t));
  println("Accuracy of DecisionTree is <correct> out of <size(tests)> = <percent(correct, size(tests))>%");
  
  println("Training Bagged Decision Trees");
  trees = buildDecTrees(trainers, maxDepth=5, minSize=10, sampleRatio=0.5, treeCount=20);
  println("Testing the Bagged Decision Trees");
  correct = (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := respond(trees, t));
  println("Accuracy of Bagged Decision Trees is <correct> out of <size(tests)> = <percent(correct, size(tests))>%");
  
  println("Training a Random Forest");
  forest = buildRandomForest(trainers, maxDepth=5, minSize=10, sampleRatio=0.5, treeCount=20, featureCount=dim(trainers) / 2);
  println("Testing the Random Forest");
  correct = (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := respond(forest, t));
  println("Accuracy of the Random Forest is <correct> out of <size(tests)> = <percent(correct, size(tests))>%");
  
}
