// Not allowed to import any other libraries
#include "huffman.h"
#include <iostream> // add this line
#include <memory>
#include <vector>

using namespace std;

// Function to allocate a new tree Node
HuffNode *getHuffNode(char alp, int count, HuffNode *left, HuffNode *right)
{
    HuffNode *node = new HuffNode();
    node->alphabet = alp;
    node->char_count = count;
    node->left = left;
    node->right = right;
    return node;
}

// Function to traverse the Huffman Tree and store Huffman Codes in a map
void stringToBits(HuffNode *root, string input, unordered_map<char, string> &frequency_table)
{
    if (root == nullptr)
    {
        return;
    }

    if (root->left == nullptr && root->right == nullptr)
    {
        frequency_table[root->alphabet] = input;
        return;
    }

    stringToBits(root->left, input + "0", frequency_table);
    stringToBits(root->right, input + "1", frequency_table);
}

// Function to traverse the Huffman Tree and decode the encoded string
string bitsToString(HuffNode *root, int &i, string str)
{
    if (!root)
    {
        return "";
    }

    if (!root->left && !root->right)
    {
        return string(1, root->alphabet);
    }

    if (str[i] == '0')
    {
        return bitsToString(root->left, ++i, str);
    }

    else
    {
        return bitsToString(root->right, ++i, str);
    }
}

// Function to build Huffman Tree and decode given input text
unordered_map<char, string> huffmanTree(string input)
{
    unordered_map<char, int> freq;
    for (char c : input)
    {
        freq[c]++;
    }

    vector<HuffNode *> nodes;
    for (auto &pair : freq)
    {
        nodes.push_back(getHuffNode(pair.first, pair.second, nullptr, nullptr));
    }

    while (nodes.size() > 1)
    {
        int min1 = 0, min2 = 1;
        if (nodes[min1]->char_count > nodes[min2]->char_count)
        {
            swap(min1, min2);
        }
        for (size_t i = 2; i < nodes.size(); ++i)
        {
            if (nodes[i]->char_count < nodes[min1]->char_count)
            {
                min2 = min1;
                min1 = i;
            }
            else if (nodes[i]->char_count < nodes[min2]->char_count)
            {
                min2 = i;
            }
        }

        HuffNode *mergedNode = getHuffNode('\0', nodes[min1]->char_count + nodes[min2]->char_count, nodes[min1], nodes[min2]);

        nodes.erase(nodes.begin() + max(min1, min2));
        nodes.erase(nodes.begin() + min(min1, min2));

        nodes.push_back(mergedNode);
    }

    unordered_map<char, string> codes;
    if (!nodes.empty())
    {
        string code;
        traverseTree(nodes[0], code, codes);
    }

    return codes;
}

void traverseTree(HuffNode *root, string &code, unordered_map<char, string> &codes)
{
    if (root)
    {
        if (root->left == nullptr && root->right == nullptr)
        {
            codes[root->alphabet] = code;
        }
        if (root->left)
        {
            code.push_back('0');
            traverseTree(root->left, code, codes);
            code.pop_back();
        }
        if (root->right)
        {
            code.push_back('1');
            traverseTree(root->right, code, codes);
            code.pop_back();
        }
    }
}
