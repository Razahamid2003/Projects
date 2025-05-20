#include <memory>
#include <vector>
#include <cstdlib>
using namespace std;

#ifndef __AVL_H
#define __AVL_H

// Node of the tree
template <class T, class S, class C>
struct node {
	S fullName;
	T workExperience; 
	string gender; 
	shared_ptr<node> left;
	shared_ptr<node> right;
    int height; 

	node(T w, S n, C g) {
		this->fullName = n; 
		this->workExperience = w; 
		this->gender = g; 
		left = NULL;
		right = NULL;
        height = 1; 
	}
};

// AVL Class (This will be used for both BST and AVL Tree implementation)
template <class T, class S, class C>
class AVL {
    shared_ptr<node<T,S,C>> root;
    bool isAVL;
    
public:
    // Part 3 Functions 
    AVL(bool);
    void insertNode(shared_ptr<node<T,S,C>>);
    void deleteNode(T k);    
    shared_ptr<node<T,S,C>>getRoot();
    shared_ptr<node<T,S,C>>searchNode(T k);
    int height (shared_ptr <node<T,S,C>> p);

    // Part 4 Functions 
    int number_to_shortlist(shared_ptr<node<T,S,C>> root); 
    vector<T> right_most(shared_ptr<node<T,S,C>> root); 
    vector<T> in_order(shared_ptr<node<T,S,C>> root); 
    vector<T> level_order(shared_ptr<node<T,S,C>> root); 
    vector<float> bias(shared_ptr<node<T,S,C>> root); 

    // Declare helper functions here
    int getBalance(shared_ptr<node<T, S, C>> p);
    shared_ptr<node<T, S, C>> rightRotate(shared_ptr<node<T, S, C>> y);
    shared_ptr<node<T, S, C>> leftRotate(shared_ptr<node<T, S, C>> x);
    shared_ptr<node<T, S, C>> insertNodeHelper(shared_ptr<node<T, S, C>> currentNode, shared_ptr<node<T, S, C>> newNode);
    shared_ptr<node<T, S, C>> deleteNodeHelper(shared_ptr<node<T, S, C>> currentNode, T key);
    shared_ptr<node<T, S, C>> searchNodeHelper(shared_ptr<node<T, S, C>> currentNode, T key);
    shared_ptr<node<T, S, C>> minValueNode(shared_ptr<node<T, S, C>> currentNode);
    shared_ptr<node<T, S, C>> balanceTree(shared_ptr<node<T, S, C>> currentNode, shared_ptr<node<T, S, C>> newNode);
    void inOrder_helper(shared_ptr<node<T,S,C>> node, vector<T>& result, int num);
    vector<float> biasHelper(vector<T> order, vector<float> ratios);
    void leftCalculator(shared_ptr<node<T, S, C>> node, vector<string> &genders);
};


#endif
