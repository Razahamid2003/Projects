#ifndef __TRIE_CPP
#define __TRIE_CPP

#include "trie.h"
#include <algorithm>
#include <stack>

trie::trie()
{
    root = shared_ptr<Node>(new Node);
    root->value = ' ';
}

void trie::insertWord(string word){}




bool trie::search(string word){
    return false;
}




string trie::longestSubstr(string word){
    return "";
}



vector<string> trie::getTrie(){
    return {};
}



vector<string> trie::getWords(shared_ptr<Node> node, string str = NULL) {
    return {};
}



void trie::deleteWord(string word) {}




#endif