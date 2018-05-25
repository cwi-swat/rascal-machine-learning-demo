@doc{
.Synopsis machine learning algorithm based on the idea of decision trees with bootstrap aggregation.

.Description

We learned this code from Python examples on <https://machinelearningmastery.com/> by Jason Brownlee
and translated the Python to Rascal. Bootstrap aggregation generates a number of decision trees
based on randomized subsets of the training corpus and then lets the trees vote democratically for
the right response.
}
module analysis::learning::BootstrapAggregation

extend analysis::learning::DecisionTrees;

set[DecTree] buildDecTrees(set[Point] corpus, int maxDepth=5, int minSize=10, num sampleRatio=0.2, int treeCount=10)
  = {buildDecTree(subsample(corpus, sampleRatio), maxDepth, minSize) | _ <- [0..treeCount]}; 

set[Response] respond(set[DecTree] model, Point instance) 
  = vote([*respond(t, instance) | t <- model]); 