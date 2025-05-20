#ifndef HUFFMAN_H
#define HUFFMAN_H
//Not allowed to import any other libraries
#include <string>
#include <unordered_map>

using namespace std;


// Huffman tree HuffNode
struct HuffNode
{
	char alphabet;
	int char_count;
	HuffNode *left, *right;
};


// Builds Huffman Tree and decode given input text
unordered_map<char, string> HuffmanTree(string inputText);

// Function to allocate a new tree HuffNode
HuffNode* getHuffNode(char ch, int freq, HuffNode* left, HuffNode* right);

// Function to traverse the Huffman Tree and store Huffman Codes in a map
void stringToBits(HuffNode* root, string str, unordered_map<char, string>& frequency_table);

// Function to traverse the Huffman Tree and decode the encoded string
string bitsToString(HuffNode* root, int& i, string str);

//Can add more functions here if needed

#endif // HUFFMANCODING_H

