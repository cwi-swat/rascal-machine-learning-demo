module analysis::learning::example::TestAlgorithms

import analysis::learning::DataPoints;
import analysis::learning::KMeans;
import analysis::learning::KNeirestNeighbors;
import analysis::learning::DecisionTrees;

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
  
  println("Training DecisionTrees");
  tree = buildDecTree(trainers, maxDepth=5, minSize=10);
  println("Testing the decision tree");
  correct = (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := respond(tree, t));
  println("Accuracy of DecisionTree is <correct> out of <size(tests)> = <percent(correct, size(tests))>%");
}