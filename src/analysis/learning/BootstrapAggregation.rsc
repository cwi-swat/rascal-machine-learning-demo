@doc{
.Synopsis machine learning algorithm based on the idea of decision trees with bootstrap aggregation.

.Description

The point of bootstrap aggregation is to counter over-fitting of the training data by
randomly sub-sampling and letting a set of random trees vote on a test point. This may 
help a bit (it's a kind of squinting your eyes, or zooming out). 

We learned this code from Python examples on <https://machinelearningmastery.com/> by Jason Brownlee
and translated the Python to Rascal. Bootstrap aggregation generates a number of decision trees
based on randomized subsets of the training corpus and then lets the trees vote democratically for
the right response.
}
module analysis::learning::BootstrapAggregation

extend analysis::learning::DecisionTrees;

// a set of trees is created by simply subsampling the corpus a number of times 
set[DecTree] buildDecTrees(set[Point] corpus, int maxDepth=5, int minSize=10, num sampleRatio=0.2, int treeCount=10)
  = {buildDecTree(subsample(corpus, round(size(corpus) * 1.0 * sampleRatio)), maxDepth, minSize) | _ <- [0..treeCount]}; 

// we predict by letting each tree predict individually, and then counting the most popular responses.
// note that voting is done in two layers here, one time for each leaf node in each tree, and once to
// aggregate the trees' responses
set[Response] respond(set[DecTree] trees, Point instance) 
  = vote([*respond(t, instance) | t <- trees]); 