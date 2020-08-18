#pragma once

#include <iostream>
#include <tuple>
#include <math.h>
// #include <stdexcept>
#include <vector>
#include <memory>

//should try using pointers for data to allow for persistent temporaries
//should try making gradient actualize expression instead of taking tensor

/*
User side template signatures

Index<char indexIdentifier>

DirectSum<VectorTypes...>
VectorField<VectorType, dimensions, divisions>
Tensor<dimensions, rank, T=double>
TensorField<dimensions, rank, divisions, T=double>


--------------------------------------------------------------------------------------------------------
Tensor arithmetic syntax inspired by FTensor!
--------------------------------------------------------------------------------------------------------


DirectSum<VectorTypes...>
#performs equivalent of a direct sum on the vector spaces given as template arguments
#instantiated with list of subvalues or default initialized

double DirectSum.dotProduct(other)
#element by element multiplication added together

double DirectSum.defaultSquareMagnitude()
#returns this->dotProduct(*this)

double DirectSum.defaultMagnitude()
#returns sqrt of squareMagnitude

tuple DirectSum.getData()
#returns tuple storing the data

get<Is...>(DirectSum<VT...>& input)
#returns recursive ith element of input by reference

project<Is...>(DirectSum<VT...>& input)
#returns recursive ith element of input by value

double dotProduct(left, right)
#returns dot product of left and right





VectorField<VectorType, dimensions, divisions>
#mimic a vector field. each dimension has division many slices,
#and each point has a VectorType value
#instantiated default or with vector<VectorType> or with VectorType*
#has [] operator to access element by reference
#has begin() and end() for forloop purposes.





Index<char indexIdentifier>
#serves as type differentiator for tensor indices.





Tensor<dimensions, rank, T=double>
#generic indexable tensor class
#instantiated with default or with Vector<T> or with T*
#has () operator with indices passed to allow for intuitive tensor arithmentic

vector<T> Tensor.getDataCopy()
#creates copy of underlying data and passes back as vector<T>





TensorField<dimensions, rank, divisions, T=double>
#based on VectorField<Tensor<dimensions, rank, T>, dimensions, divisions>
#uses same intuitive tensor arithmetic with () operator
#instantiated with default, vector<Tensor<dimensions, rank, T>>, or Tensor<dimensions, rank, T>*


*/

#include "TemplateHelpers.h"

#include "DirectSums.h"

// #include "VectorFields.h"

#include "Tensors.h"
// using namespace std;
#include "TensorFields.h"
