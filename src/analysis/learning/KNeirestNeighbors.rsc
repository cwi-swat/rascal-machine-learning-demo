@doc{
.Synopsis machine learning algorithm based on the nearest neighbors principle.

.Description

We learned this code from Python examples on <https://machinelearningmastery.com/> by Jason Brownlee
and translated (heavily) the Python to Rascal.
}
module analysis::learning::KNeirestNeighbors

extend analysis::learning::DataPoints;

@doc{
.Synopsis 
Compute the k nearest neighbors in the corpus to the instance point.

}
@memo
set[Point] neighbors(set[Point] corpus, Point instance, int k) {
  bool nearest(Point a, Point b) = nearerTo(instance, a, b);
  
  return {*sort(corpus, nearest)[..k]}; 
}
  
private bool nearerTo(Point pivot, Point a, Point b) 
  = dist(pivot, a) < dist(pivot, b);
  
@doc{
.Synopsis Return the most frequent responses in the k nearest neighborhood of the instance point.
}  
set[Response] respond(set[Point] corpus, Point instance, int k)
  = vote(neighbors(corpus, instance, k));  
  