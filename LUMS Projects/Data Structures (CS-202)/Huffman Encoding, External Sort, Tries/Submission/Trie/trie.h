#include "trie.cpp"

#ifndef __TRIE_HPP
#define __TRIE_HPP

#include <iostream>
#include <vector>
#include <memory>
using namespace std;

struct Node
{
    char value;
    vector<shared_ptr<Node>> children;
};

class trie
{
    shared_ptr<Node> root;

public:
    // The explanation of what each function is supposed to do and its expected outcome has been given in the assignment manual. Please go through that in detail and reach out to the course staff in case of any concerns or queries.
    trie();
    shared_ptr<Node> getRoot() { return root; }
    void insertWord(string word);
    bool search(string word);
    string longestSubstr(string word);
    vector<string> getTrie();
    vector<string> getWords(shared_ptr<Node> node, string prefix);
    void deleteWord(string word);

    void traverseTrie(shared_ptr<Node> node, string& currentWord, vector<string>& words);
};

#endif