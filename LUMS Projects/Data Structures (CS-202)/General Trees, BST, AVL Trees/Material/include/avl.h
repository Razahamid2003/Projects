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

};


#endif
