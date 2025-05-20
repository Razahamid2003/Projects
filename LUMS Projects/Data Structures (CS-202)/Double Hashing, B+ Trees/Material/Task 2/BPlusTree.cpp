#include "BPlusTree.h"
using namespace std;

BPlusNode::BPlusNode() {
    for (int i = 0; i < DEGREE + 1; ++i) {
        children[i] = nullptr;
    }
}

BPlusNode::~BPlusNode() {
    for (int i = 0; i < DEGREE + 1; ++i) {
        delete children[i];
    }
}

BPlusTree::BPlusTree() : root(nullptr) {}

BPlusTree::~BPlusTree() {
    delete root;
}

void BPlusTree::insert(int key, int value) {
    // TODO: Implement B+ tree insertion logic
}

void BPlusTree::insertInternal(int key, BPlusNode* cursor, BPlusNode* child) {}

BPlusNode* BPlusTree::findParent(BPlusNode* current, BPlusNode* child) {}

void BPlusTree::remove(int key) {
    // TODO: Implement B+ tree deletion logic
}

void BPlusTree::removeInternal(BPlusNode* node, int key) {}
void BPlusTree::redistributeKeys(BPlusNode* parent, int index) {}
void BPlusTree::mergeNodes(BPlusNode* parent, int index) {}

void BPlusTree::printTree() {
    // TODO: Implement a function to print the B+ tree for debugging purposes
}

vector<int> BPlusTree::inOrderTraversalHelper(BPlusNode* node) {}

vector<int> BPlusTree::inOrderTraversal() {
    // TODO: Implement a function which returns the result 
    // of in order traversal of the B+ Tree
    return vector<int>{};
}

int BPlusTree::findKey(int key) {
    // TODO: Find the given key and return -1 in case not found
    return -1;
}

BPlusNode* BPlusTree::findLeafNode(int key) {}

int BPlusTree::findMinKey() {
    // TODO: Implement a function to find the minimum key in the B+ tree
    return -1; // Placeholder value, replace with actual result
}

int BPlusTree::findMaxKey() {
    // TODO: Implement a function to find the maximum key in the B+ tree
    return -1; // Placeholder value, replace with actual result
}

void BPlusTree::clearTree(BPlusNode* node) {}

void BPlusTree::clearTree() {
    // TODO: Implement a function to clear the entire B+ tree
}