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

void trie::insertWord(string word)
{
    shared_ptr<Node> current = root;

    for (char c : word)
    {
        bool found = false;
        for (const auto &child : current->children)
        {
            if (child->value == c)
            {
                current = child;
                found = true;
                break;
            }
        }

        if (!found)
        {
            shared_ptr<Node> newNode = make_shared<Node>();
            newNode->value = c;
            current->children.push_back(newNode);
            current = newNode;
        }
    }
}

bool trie::search(string word)
{
    shared_ptr<Node> current = root;

    for (char c : word)
    {
        bool found = false;
        for (const auto &child : current->children)
        {
            if (child->value == c)
            {
                current = child;
                found = true;
                break;
            }
        }

        if (!found)
        {
            return false;
        }
    }
    return true;
}

string trie::longestSubstr(string word)
{
    shared_ptr<Node> current = root;
    string longestString = "";
    string result = "";

    for (char c : word)
    {
        bool found = false;
        for (const auto &child : current->children)
        {
            if (child->value == c)
            {
                current = child;
                result += c;
                found = true;
                break;
            }
        }
        if (!found)
        {
            if (result.size() > longestString.size())
            {
                longestString = result;
            }
            result = "";
            current = root;
        }
    }
    if (result.size() > longestString.size())
    {
        longestString = result;
    }

    return longestString;
}

vector<string> trie::getTrie()
{
    vector<string> words;
    string currentWord;
    traverseTrie(root, currentWord, words);

    return words;
}

void trie::traverseTrie(shared_ptr<Node> node, string &currentWord, vector<string> &words)
{
    if (node->value != '\0')
    {
        currentWord.push_back(node->value);
    }

    if (currentWord.size() > 0 && node->value == '\0')
    {
        words.push_back(currentWord);
    }

    for (const auto &child : node->children)
    {
        traverseTrie(child, currentWord, words);
    }

    if (node->value != '\0')
    {
        currentWord.pop_back();
    }
}

vector<string> trie::getWords(shared_ptr<Node> node, string str = NULL)
{
    vector<string> words;
    if (!node)
    {
        return words;
    }

    if (!str.empty())
    {
        words.push_back(str);
    }
    for (const auto &child : node->children)
    {
        vector<string> childWords = getWords(child, str + child->value);
        words.insert(words.end(), childWords.begin(), childWords.end());
    }

    return words;
}

void trie::deleteWord(string word)
{
    shared_ptr<Node> current = root;
    vector<shared_ptr<Node>> visitedNodes;

    for (char c : word)
    {
        bool found = false;
        for (const auto &child : current->children)
        {
            if (child->value == c)
            {
                current = child;
                found = true;
                break;
            }
        }
        if (!found)
        {
            return;
        }
        visitedNodes.push_back(current);
    }
    current->value = '\0';
    for (int i = visitedNodes.size() - 1; i >= 0; --i)
    {
        if (visitedNodes[i]->children.empty() && visitedNodes[i] != root)
        {
            shared_ptr<Node> parent = root;
            for (int j = 1; j < i; ++j)
            {
                for (const auto &child : parent->children)
                {
                    if (child == visitedNodes[j])
                    {
                        parent = child;
                        break;
                    }
                }
            }
            parent->children.erase(remove(parent->children.begin(), parent->children.end(), visitedNodes[i]), parent->children.end());
        }
    }
}
#endif