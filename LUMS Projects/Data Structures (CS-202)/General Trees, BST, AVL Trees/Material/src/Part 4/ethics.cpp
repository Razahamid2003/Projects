#include <iostream>
#include <memory>
#include "../../include/avl.h"

// This function calculates the maximum applicants that u can shortlist
template <class T, class S, class C> 
int AVL<T,S,C> :: number_to_shortlist (shared_ptr<node<T,S,C>> root){
}

// This function returns shortlisted candidates on the right most path to leaf 
template <class T, class S, class C> 
vector<T> AVL<T,S,C> :: right_most (shared_ptr<node<T,S,C>> root){
}

// This function returns shortlisted candidates in-order
template <class T, class S, class C> 
vector<T> AVL<T,S,C> :: in_order (shared_ptr<node<T,S,C>> root){
}

// This function returns shortlisted candidates in level order 
template <class T, class S, class C> 
vector<T> AVL<T,S,C> :: level_order (shared_ptr<node<T,S,C>> root){
}

// This function calculates the bias in the tree
template <class T, class S, class C> 
vector<float> AVL<T,S,C> :: bias (shared_ptr<node<T,S,C>> root){
}

