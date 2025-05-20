#ifndef BPLUSTREE_H
#define BPLUSTREE_H

#include <iostream>
#include <vector>
using namespace std;
// Define the maximum degree of the B+ tree
#define DEGREE 3

// Structure to represent a B+ tree node
struct BPlusNode {
    vector<int> keys;  // Keys in the node
    vector<int> values; // Values associated with the keys (for simplicity, using integers)
    BPlusNode* children[DEGREE + 1]; // Pointers to children nodes
    bool isLeaf;

    BPlusNode();
    ~BPlusNode();
};

// B+ tree class
class BPlusTree {
private:
    BPlusNode* root;

    BPlusNode* findLeafNode(int key);
    void removeInternal(BPlusNode* node, int key);
    void clearTree(BPlusNode* node);
    BPlusNode* findParent(BPlusNode* current, BPlusNode* child);
    vector<int> inOrderTraversalHelper(BPlusNode* node);
    void insertInternal(int key, BPlusNode* cursor, BPlusNode* child);
    void mergeNodes(BPlusNode* parent, int index);
    void redistributeKeys(BPlusNode* parent, int index); 
    void printTree();

public:
    BPlusTree();
    ~BPlusTree();

    void insert(int key, int value);
    void remove(int key);
    vector<int> inOrderTraversal();
    void traverse();
    int findMinKey();
    int findMaxKey();
    int findKey(int key);
    void clearTree();
};

#endif
