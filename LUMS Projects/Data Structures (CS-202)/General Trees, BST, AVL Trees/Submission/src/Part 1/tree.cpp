#include "../../include/tree.h"

// You aren't allowed to edit these method declarations or declare global variables

// Constructor
template <class T, class S>
Tree<T, S>::Tree(shared_ptr<node<T, S>> root)
{
    this->root = root;
}

// This function finds a key in the tree and returns the respective node
template <class T, class S>
shared_ptr<node<T, S>> Tree<T, S>::findKey(T key)
{
    return findKeyHelper(root, key);
}

// Helper function to find a key in the tree
template <class T, class S>
shared_ptr<node<T, S>> Tree<T, S>::findKeyHelper(shared_ptr<node<T, S>> currNode, T key)
{
    if (currNode == NULL)
    {
        return NULL;
    }

    if (currNode->key == key)
    {
        return currNode;
    }
    for (const auto &child : currNode->children)
    {
        auto result = findKeyHelper(child, key);
        if (result != NULL)
        {
            return result;
        }
    }
    return NULL;
}

// This function inserts the given node as a child of the given key
template <class T, class S>
bool Tree<T, S>::insertChild(shared_ptr<node<T, S>> newNode, T key)
{

    shared_ptr<node<T, S>> existingNode = findKey(newNode->key);
    if (existingNode)
    {
        return false;
    }

    auto parent = findKey(key);

    if (parent == NULL)
    {
        cout << "Parent Node is Null!\n";
        return false;
    }

    for (const auto &child : parent->children)
    {
        if (child->key == newNode->key)
        {
            return false;
        }
    }

    parent->children.push_back(newNode);
    return true;
}

// This function returns all the children of a node with the given key
template <class T, class S>
vector<shared_ptr<node<T, S>>> Tree<T, S>::getAllChildren(T key)
{
    shared_ptr<node<T, S>> parentNode = findKey(key);
    if (parentNode == nullptr || parentNode->children.empty())
    {
        return vector<shared_ptr<node<T, S>>>();
    }
    else
    {
        return parentNode->children;
    }
}

// This function returns the height of the tree
template <class T, class S>
int Tree<T, S>::findHeight()
{
    return findHeightHelper(root);
}

// Helper function to find height of the tree
template <class T, class S>
int Tree<T, S>::findHeightHelper(shared_ptr<node<T, S>> currNode)
{
    if (currNode == NULL)
    {
        return -1;
    }

    int maxHeight = -1;

    for (const auto &child : currNode->children)
    {
        int childHeight = findHeightHelper(child);
        maxHeight = max(maxHeight, childHeight);
    }

    return maxHeight + 1;
}

// This function deletes the node of a given key (iff it is a leaf node)
template <class T, class S>
bool Tree<T, S>::deleteLeaf(T key)
{
    root = deleteLeafHelper(root, key);
    return (root != nullptr);
}

// Helper function to delete leaf node
template <class T, class S>
shared_ptr<node<T, S>> Tree<T, S>::deleteLeafHelper(shared_ptr<node<T, S>> currentNode, T key)
{
    if (!currentNode)
    {
        return nullptr;
    }

    if (currentNode->key == key && currentNode->children.empty())
    {
        return nullptr;
    }

    vector<shared_ptr<node<T, S>>> newChildren;
    for (const auto &child : currentNode->children)
    {
        shared_ptr<node<T, S>> newChild = deleteLeafHelper(child, key);
        if (newChild)
        {
            newChildren.push_back(newChild);
        }
    }
    currentNode->children = newChildren;

    return currentNode;
}

// This function deletes the tree
template <class T, class S>
void Tree<T, S>::deleteTree(shared_ptr<node<T, S>> currNode)
{
    if (currNode == nullptr)
    {
        root = nullptr;
        return;
    }

    for (const auto &child : currNode->children)
    {
        deleteTree(child);
    }

    currNode->children.clear();
}

// This function returns the root of the tree
template <class T, class S>
shared_ptr<node<T, S>> Tree<T, S>::getRoot()
{
    return root;
}
