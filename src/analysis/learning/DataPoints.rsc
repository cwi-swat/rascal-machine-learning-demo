module analysis::learning::DataPoints

import analysis::statistics::Frequency;
import util::Math;
import List;
import Set;

@doc{
.Synopsis Responses can be any data, classification or action associated  with data points.
.Description

The training set of a learner consists of data points and their association to something
which must be learned. This is called a *Response* here. You can add new kinds of 
responses, but be careful not to include closures since the responses are used for
training usually based on the structural equality of the responses.

Basic classification schemes would attach classes, e.g. `class("Animal")` and
`class("Plant")`. The `silence()` constructor is a default bottom value used for
erroneous situations. More interesting and structured responses are advised, for
example one could introduce an expression language for instructing a robot here,
where the point data would represent telemetry and the expression language could
represent telecommands which are the learned responses to measured circumstances. 
} 
data Response 
  = class(str name) 
  | silence()
  ;
 
@doc{
.Synopsis a point is a vector of reals with optionally an associated learned Response.
.Description

All learners in this library use this specific input type: Point. If your data
is not a vector of reals, then it must first be converted to it. 
}  
data Point = point(list[real] vec, Response resp = silence());

@doc{
.Synopsis Generate a point for testing purposes
}
Point arbPoint(int d) = point([arbReal() | _ <- [0..d]]);

@doc{
.Synopsis Produce a randomized subset of a corpus of a given size relative to the original corpus
}
set[Point] subsample(set[Point] corpus, num ratio) 
  = { l[arbInt(size(corpus))] | _ <- [0..round(size(corpus) * ratio)]}
  when l := [*corpus];
  
@doc{
.Synopsis 
Return the number of dimensions for a point or a set of points.

.Description
For a set of points it is assumed they all have the same amount of dimensions
}
int dim(Point p) = size(p.vec);
int dim(set[Point] _:{Point h, *_}) = dim(h);
int dim({}) = 0;

@doc{
.Synopsis Compute the maximum number of dimensions used in this set of points
}
int maxDim(set[Point] corpus) = max({size(p.vec) | p <- corpus});
int maxDim({}) = 0;

@doc{
.Synopsis Compute the minumum number of dimensions used in this set of points
}
int minDim(set[Point] corpus) = min({size(p.vec) | p <- corpus});
int minDim({}) = 0;
 
private real sqr(real r) = r * r;

@doc{
.Synopsis 
A measure of distance (the square root of Euclidean distance in fact).

.Description

We compute pairwise the square of the absolute distance of each dimension to arrive
at the square of the Euclidian distance between two n-dimensional points.

The dist function is robust against points with different dimensionalities and will
fill the shorter point with .0 values until the dimensions are the same.
}
@memo
real dist(Point p, Point q)
  = (.0 | it + sqr(abs((p.vec[i]?.0) - (q.vec[i]?.0))) | i <- [0..max(size(p.vec),size(q.vec))]);

@doc{
.Synopsis split a data set into two parts, with `frac` portion associated 
to a training set and the rest of the data moved to a testing set.
}
tuple[set[Point] trainers, set[Point] tests] split(set[Point] corpus, num frac) {
  trainers = {*([*corpus][..round(frac * size(corpus))])};
  tests = corpus - trainers;
  return <trainers, tests>;
}

@doc{
.Synopsis Split a data-set into k almost equal parts.

.Description

This is handy for generating an initial "random" classification into k classes,
or to split a data set for k-fold cross-validation in equal parts.
}
set[set[Point]] kFolds(set[Point] corp, int k) = kFolds([*corp], k);
set[set[Point]] kFolds(list[Point] corp, 0) = {};
set[set[Point]] kFolds(list[Point] corp, 1) = {{*corp}};
set[set[Point]] kFolds(list[Point] corp, int k) = {{*corp[..s/k]}, *kFolds(corp[s/k..], k - 1)} 
  when s := size(corp);

@doc{
.Synopsis get the responses that occurs the most in a set of data points
}   
set[Response] vote(set[Point] points) {
  d = distribution({<p, p.resp> | p <- points});
  m = max(d<1>);
  return {r | r <- d, d[r] == m}; 
}

@doc{
.Synopsis return the responses that occur the most
}
set[Response] vote(list[Response] responses) {
  d = distribution(responses);
  return d<1,0>[max(d<1>)];
}