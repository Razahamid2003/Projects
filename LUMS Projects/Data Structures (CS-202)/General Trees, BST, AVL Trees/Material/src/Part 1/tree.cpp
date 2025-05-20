#include "../../include/tree.h"

// You aren't allowed to edit these method declarations or declare global variables

// Constructor
template <class T, class S>
Tree<T,S>::Tree(shared_ptr<node<T,S>> root) {
}

// This function finds a key in the tree and returns the respective node
template <class T, class S>
shared_ptr<node<T,S>> Tree<T,S>::findKey(T key) {
}

// Helper function to find a key in the tree
template <class T, class S>
shared_ptr<node<T,S>> Tree<T,S>::findKeyHelper(shared_ptr<node<T,S>> currNode, T key) {
}

// This function inserts the given node as a child of the given key
template <class T, class S>
bool Tree<T,S>::insertChild(shared_ptr<node<T,S>> newNode, T key) {
}

// This function returns all the children of a node with the given key
template <class T, class S>
vector<shared_ptr<node<T,S>>> Tree<T,S>::getAllChildren(T key) {
}

// This function returns the height of the tree
template <class T, class S>
int Tree<T,S>::findHeight() {
}

// Helper function to find height of the tree
template <class T, class S>
int Tree<T,S>::findHeightHelper(shared_ptr<node<T,S>> currNode) {
}

// This function deletes the node of a given key (iff it is a leaf node)
template <class T, class S>
bool Tree<T,S>::deleteLeaf(T key) {
}

// Helper function to delete leaf node
template <class T, class S>
shared_ptr<node<T,S>> Tree<T,S>::deleteLeafHelper(shared_ptr<node<T,S>> currNode, T key) {
}

// This function deletes the tree
template <class T, class S>
void Tree<T,S>::deleteTree(shared_ptr<node<T,S>> currNode) {
}

// This function returns the root of the tree
template <class T, class S>
shared_ptr<node<T,S>>  Tree<T,S>::getRoot() {
}

