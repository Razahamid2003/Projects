#include <iostream>
#include <memory>
#include "../../include/avl.h"

// Constructor
template <class T, class S, class C>
AVL<T,S,C>::AVL(bool isAVL){
}

// This function inserts a given node in the tree
template <class T, class S, class C> 
void AVL<T,S,C>::insertNode(shared_ptr<node<T,S,C>> newNode){
}

// This function searches a node in a tree by its key
template <class T, class S, class C>
shared_ptr<node<T,S,C>> AVL<T,S,C>::searchNode(T k){
} 

// This function deletes a given node from the tree
template <class T, class S, class C> 
void AVL<T,S,C>::deleteNode(T k){
}

// This function returns the root of the tree
template <class T, class S, class C> 
shared_ptr<node<T,S,C>> AVL<T,S,C>::getRoot(){
}

// This function calculates and returns the height of the tree
template <class T, class S, class C> 
int AVL<T,S,C>::height(shared_ptr<node<T,S,C>> p){
}
