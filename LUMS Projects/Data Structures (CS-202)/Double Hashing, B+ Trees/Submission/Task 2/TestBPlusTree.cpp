#include <iostream>
#include <cassert>
#include <csignal>
#include <vector>
#include "BPlusTree.h"
#include "BPlusTree.cpp"

using namespace std;

int test_case = 0;
int part = 1;
int total_marks = 0;
int points = 0;
int flag = 1;

void signal_handler(int);
void print_success();

int main() {
    // Create an instance of the BPlusTree class
    BPlusTree bPlusTree;

    // Test 1 - B+ Tree Insertion
    cout << "Test 1 - B+ Tree Insertion" << endl;
    signal(SIGSEGV, signal_handler);

    // Case-I: Insert key-value pair
    test_case = 1;
    bPlusTree.insert(5, 10);
    assert(bPlusTree.findMinKey() == 5);
    assert(bPlusTree.findMaxKey() == 5);
    print_success();

    // Case-II: Insert another key-value pair
    test_case = 2;
    bPlusTree.insert(8, 20);
    assert(bPlusTree.findMinKey() == 5);
    assert(bPlusTree.findMaxKey() == 8);
    print_success();

    // Case-III: Insert more key-value pairs
    test_case = 3;
    bPlusTree.insert(3, 30);
    bPlusTree.insert(10, 40);
    bPlusTree.insert(15, 50);
    assert(bPlusTree.findMinKey() == 3);
    assert(bPlusTree.findMaxKey() == 15);
    print_success();

    // Case-IV: Insert duplicate key
    test_case = 4;
    bPlusTree.insert(8, 60);
    assert(bPlusTree.findMinKey() == 3);
    assert(bPlusTree.findMaxKey() == 15);
    print_success();

    // Case-V: Insert key into full leaf node, causing a split
    test_case = 5;
    for (int i = 20; i <= 30; ++i) {
        bPlusTree.insert(i, i * 10);
    }
    //assert(bPlusTree.findMinKey() == 3);
    assert(bPlusTree.findMaxKey() == 30);
    print_success();

    // // Test 2 - B+ Tree Deletion
    // cout << "Test 2 - B+ Tree Deletion" << endl;

    // // Case-I: Delete a key
    // test_case = 1;
    // bPlusTree.remove(8);
    // assert(bPlusTree.findKey(8) == -1);
    // print_success();

    // // Case-II: Delete another key
    // test_case = 2;
    // bPlusTree.remove(5);
    // assert(bPlusTree.findKey(5) == -1);
    // print_success();

    // // Case-III: Delete a non-existing key
    // test_case = 3;
    // bPlusTree.remove(12);
    // assert(bPlusTree.findKey(12) == -1);
    // print_success();

    // // Case-IV: Delete key causing redistribution
    // test_case = 4;
    // bPlusTree.remove(20);
    // assert(bPlusTree.findKey(20) == -1);
    // print_success();

    // // Case-V: Delete root node
    // test_case = 5;
    // bPlusTree.remove(15);
    // assert(bPlusTree.findKey(15) == -1);
    // print_success();

    // // Test 3 - In-Order Traversal
    // cout << "Test 3 - In-Order Traversal" << endl;
    // test_case = 1;
    // vector<int> inOrderResult = bPlusTree.inOrderTraversal();
    // vector<int> expectedOrder = {3, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    // assert(inOrderResult.size() == expectedOrder.size());
    // for (size_t i = 0; i < inOrderResult.size(); ++i) {
    //     assert(inOrderResult[i] == expectedOrder[i]);
    // }

    // print_success();

    // Test 4 - Find Key
    cout << "Test 4 - Find Key" << endl;
    test_case = 1;
    assert(bPlusTree.findKey(21) == 210);
    assert(bPlusTree.findKey(5) == -1);
    print_success();

    // Test 5 - Clear Tree
    cout << "Test 5 - Clear Tree" << endl;
    test_case = 1;
    bPlusTree.clearTree();
    assert(bPlusTree.findMinKey() == -1);
    assert(bPlusTree.findMaxKey() == -1);
    print_success();

    cout << "\033[34m";
    cout << "\nTotal Points: " << points << " / " << total_marks << endl;
    cout << "\033[0m";

    float scaled_points = ((float)points / total_marks) * 20;
    cout << "\033[33m";
    cout << endl << "Total Marks: " << total_marks << " / " << total_marks << endl;
    cout << endl << "Scaled Marks: " << scaled_points << " / 20" << endl;
    cout << "\033[0m";

    if (total_marks == 20) {
        cout << "\nWoohoo! Take a break and onto the next one :)" << endl;
    }

    cout << endl;
    return 0;
}

void signal_handler(int signum) {
    cout << "   " << part << "." << test_case << ": \033[31mFailed\033[0m\n";
}

void print_success() {
    cout << "   " << part << "." << test_case << ": \033[32mPassed\033[0m\n";
}
