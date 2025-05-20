#include <iostream>
#include <memory>
#include "../../include/avl.h"

// Constructor
template <class T, class S, class C>
AVL<T, S, C>::AVL(bool isAVL)
{
    this->isAVL = isAVL;
    root = nullptr;
}

// This function inserts a given node in the tree
template <class T, class S, class C>
void AVL<T, S, C>::insertNode(shared_ptr<node<T, S, C>> newNode)
{
    if (!root)
    {
        root = newNode;
    }
    else
    {
        root = insertNodeHelper(root, newNode);
    }
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::insertNodeHelper(shared_ptr<node<T, S, C>> currentNode, shared_ptr<node<T, S, C>> newNode)
{
    if (!currentNode)
    {
        return newNode;
    }

    if (newNode->workExperience < currentNode->workExperience)
    {
        currentNode->left = insertNodeHelper(currentNode->left, newNode);
    }
    else if (newNode->workExperience > currentNode->workExperience)
    {
        currentNode->right = insertNodeHelper(currentNode->right, newNode);
    }
    else
    {
        // Duplicate keys are not allowed
        return currentNode;
    }

    if (isAVL == true)
    {
        currentNode->height = 1 + max(height(currentNode->left), height(currentNode->right));

        int balance = getBalance(currentNode);

        if (balance > 1 && newNode->workExperience < currentNode->left->workExperience)
        {
            return rightRotate(currentNode);
        }

        if (balance < -1 && newNode->workExperience > currentNode->right->workExperience)
        {
            return leftRotate(currentNode);
        }

        if (balance > 1 && newNode->workExperience > currentNode->left->workExperience)
        {
            currentNode->left = leftRotate(currentNode->left);
            return rightRotate(currentNode);
        }

        if (balance < -1 && newNode->workExperience < currentNode->right->workExperience)
        {
            currentNode->right = rightRotate(currentNode->right);
            return leftRotate(currentNode);
        }
    }
    return currentNode;
}

// This function searches a node in a tree by its key
template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::searchNode(T k)
{
    return searchNodeHelper(root, k);
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::searchNodeHelper(shared_ptr<node<T, S, C>> currentNode, T key)
{
    if (currentNode == nullptr)
    {
        return NULL;
    }

    if (currentNode->workExperience == key)
    {
        return currentNode;
    }

    if (key < currentNode->workExperience)
    {
        return searchNodeHelper(currentNode->left, key);
    }

    return searchNodeHelper(currentNode->right, key);
}

// This function deletes a given node from the tree
template <class T, class S, class C>
void AVL<T, S, C>::deleteNode(T key)
{
    if (!root)
    {
        return;
    }
    root = deleteNodeHelper(root, key);
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::deleteNodeHelper(shared_ptr<node<T, S, C>> currentNode, T key)
{
    if (!currentNode)
        return currentNode;

    if (key < currentNode->workExperience)
    {
        currentNode->left = deleteNodeHelper(currentNode->left, key);
    }
    else if (key > currentNode->workExperience)
    {
        currentNode->right = deleteNodeHelper(currentNode->right, key);
    }
    else
    {
        if (!currentNode->left || !currentNode->right)
        {
            shared_ptr<node<T, S, C>> temp = currentNode->left ? currentNode->left : currentNode->right;
            if (!temp)
            {
                temp = currentNode;
                currentNode = nullptr;
            }
            else
            {
                *currentNode = *temp;
            }
        }
        else
        {
            shared_ptr<node<T, S, C>> temp = minValueNode(currentNode->right);
            currentNode->workExperience = temp->workExperience;
            currentNode->right = deleteNodeHelper(currentNode->right, temp->workExperience);
        }
    }

    if (!currentNode)
    {
        return currentNode;
    }

    if (isAVL == true)
    {
        currentNode->height = 1 + max(height(currentNode->left), height(currentNode->right));

        int balance = getBalance(currentNode);

        if (balance > 1 && getBalance(currentNode->left) >= 0)
        {
            return rightRotate(currentNode);
        }

        if (balance > 1 && getBalance(currentNode->left) < 0)
        {
            currentNode->left = leftRotate(currentNode->left);
            return rightRotate(currentNode);
        }

        if (balance < -1 && getBalance(currentNode->right) <= 0)
        {
            return leftRotate(currentNode);
        }

        if (balance < -1 && getBalance(currentNode->right) > 0)
        {
            currentNode->right = rightRotate(currentNode->right);
            return leftRotate(currentNode);
        }
    }

    return currentNode;
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::minValueNode(shared_ptr<node<T, S, C>> currentNode)
{
    shared_ptr<node<T, S, C>> current = currentNode;

    while (current->left != nullptr)
    {
        current = current->left;
    }
    return current;
}

// This function returns the root of the tree
template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::getRoot()
{
    return root;
}

// This function calculates and returns the height of the tree
template <class T, class S, class C>
int AVL<T, S, C>::height(shared_ptr<node<T, S, C>> p)
{
    if (p == nullptr)
    {
        return 0;
    }
    return p->height;
}

template <class T, class S, class C>
int AVL<T, S, C>::getBalance(shared_ptr<node<T, S, C>> p)
{
    if (p == nullptr)
    {
        return 0;
    }
    return height(p->left) - height(p->right);
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::rightRotate(shared_ptr<node<T, S, C>> y)
{
    shared_ptr<node<T, S, C>> x = y->left;
    shared_ptr<node<T, S, C>> T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(height(y->left), height(y->right)) + 1;
    x->height = max(height(x->left), height(x->right)) + 1;

    return x;
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::leftRotate(shared_ptr<node<T, S, C>> x)
{
    shared_ptr<node<T, S, C>> y = x->right;
    shared_ptr<node<T, S, C>> T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(height(x->left), height(x->right)) + 1;
    y->height = max(height(y->left), height(y->right)) + 1;

    return y;
}

template <class T, class S, class C>
shared_ptr<node<T, S, C>> AVL<T, S, C>::balanceTree(shared_ptr<node<T, S, C>> currentNode, shared_ptr<node<T, S, C>> newNode)
{
    currentNode->height = 1 + max(height(currentNode->left), height(currentNode->right));

    int balanceFactor = getBalance(currentNode);

    // Left Left Case
    if (balanceFactor > 1 && newNode->workExperience < currentNode->left->workExperience)
        return rightRotate(currentNode);
    // Right Right Case
    if (balanceFactor < -1 && newNode->workExperience > currentNode->right->workExperience)
        return leftRotate(currentNode);
    // Left Right Case
    if (balanceFactor > 1 && newNode->workExperience > currentNode->left->workExperience)
    {
        currentNode->left = leftRotate(currentNode->left);
        return rightRotate(currentNode);
    }
    // Right Left Case
    if (balanceFactor < -1 && newNode->workExperience < currentNode->right->workExperience)
    {
        currentNode->right = rightRotate(currentNode->right);
        return leftRotate(currentNode);
    }

    return currentNode;
}