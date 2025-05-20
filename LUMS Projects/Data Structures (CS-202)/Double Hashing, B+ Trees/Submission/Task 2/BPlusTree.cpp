#include "BPlusTree.h"
#include <queue> //used in printTree()
using namespace std;

BPlusNode::BPlusNode()
{
    for (int i = 0; i < DEGREE + 1; ++i)
    {
        children[i] = nullptr;
    }
}

BPlusNode::~BPlusNode()
{
    for (int i = 0; i < DEGREE + 1; ++i)
    {
        if (children[i] != nullptr)
        {
            delete children[i];
        }
    }
}

BPlusTree::BPlusTree() : root(nullptr) {}

BPlusTree::~BPlusTree()
{
    delete root;
}

void BPlusTree::insert(int key, int value)
{
    if (root == nullptr)
    {
        root = new BPlusNode();
        root->keys.push_back(key);
        root->values.push_back(value);
        root->isLeaf = true;
        return;
    }

    BPlusNode *leafNode = findLeafNode(key);

    for (size_t i = 0; i < leafNode->keys.size(); ++i)
    {
        if (leafNode->keys[i] == key)
        {
            leafNode->values[i] = value;
            return;
        }
    }

    auto it = lower_bound(leafNode->keys.begin(), leafNode->keys.end(), key);
    int index = it - leafNode->keys.begin();
    leafNode->keys.insert(it, key);
    leafNode->values.insert(leafNode->values.begin() + index, value);

    if (leafNode->keys.size() > DEGREE)
    {
        BPlusNode *newLeafNode = new BPlusNode();
        int midIndex = leafNode->keys.size() / 2;

        newLeafNode->keys.assign(leafNode->keys.begin() + midIndex, leafNode->keys.end());
        newLeafNode->values.assign(leafNode->values.begin() + midIndex, leafNode->values.end());
        leafNode->keys.erase(leafNode->keys.begin() + midIndex, leafNode->keys.end());
        leafNode->values.erase(leafNode->values.begin() + midIndex, leafNode->values.end());

        newLeafNode->isLeaf = true;

        if (leafNode == root)
        {
            root = new BPlusNode();
            root->keys.push_back(newLeafNode->keys[0]);
            root->values.push_back(newLeafNode->values[0]);
            root->children[0] = leafNode;
            root->children[1] = newLeafNode;
            root->isLeaf = false;
        }
        else
        {
            BPlusNode *parent = findParent(root, leafNode);
            insertInternal(newLeafNode->keys[0], newLeafNode->values[0], parent, newLeafNode);
        }
    }
}

void BPlusTree::insertInternal(int key, int value, BPlusNode *cursor, BPlusNode *child)
{
    int index = 0;
    while (index < cursor->keys.size() && cursor->keys[index] < key)
    {
        ++index;
    }

    if (cursor->keys.size() >= DEGREE)
    {
        BPlusNode *newNode = new BPlusNode();
        newNode->isLeaf = cursor->isLeaf;

        int mid = (cursor->keys.size() + 1) / 2;
        int middleKey = cursor->keys[mid];
        int middleVal = cursor->values[mid];

        newNode->keys.assign(cursor->keys.begin() + mid + 1, cursor->keys.end());
        newNode->values.assign(cursor->values.begin() + mid + 1, cursor->values.end());
        cursor->keys.erase(cursor->keys.begin() + mid, cursor->keys.end());
        cursor->values.erase(cursor->values.begin() + mid, cursor->values.end());

        if (!cursor->isLeaf)
        {
            newNode->children[0] = cursor->children[mid + 1];
            cursor->children[mid + 1] = nullptr;
            for (int i = mid + 1; i <= DEGREE; ++i)
            {
                newNode->children[i - mid] = cursor->children[i];
                cursor->children[i] = nullptr;
            }
        }

        if (cursor == root)
        {
            BPlusNode *newRoot = new BPlusNode();
            newRoot->isLeaf = false;
            newRoot->keys.push_back(middleKey);
            newRoot->values.push_back(middleVal);
            newRoot->children[0] = cursor;
            newRoot->children[1] = newNode;
            root = newRoot;
        }
        else
        {
            BPlusNode *parent = findParent(root, cursor);
            insertInternal(middleKey, middleVal, parent, newNode);
        }

        if (key > middleKey)
        {
            cursor = newNode;
            index -= mid + 1;
        }
    }

    cursor->keys.insert(cursor->keys.begin() + index, key);
    cursor->values.insert(cursor->values.begin() + index, value);
    cursor->children[index + 1] = child;
}

BPlusNode *BPlusTree::findParent(BPlusNode *current, BPlusNode *child)
{
    if (current->isLeaf || current->children[0] == nullptr)
    {
        return nullptr;
    }

    for (int i = 0; i <= DEGREE; ++i)
    {
        if (current->children[i] == child)
        {
            return current;
        }
        if (current->children[i] != nullptr)
        {
            BPlusNode *parent = findParent(current->children[i], child);
            if (parent != nullptr)
            {
                return parent;
            }
        }
    }

    return nullptr;
}

void BPlusTree::remove(int key)
{
    if (root == nullptr)
    {
        return;
    }

    BPlusNode *leafNode = findLeafNode(key);

    if (leafNode == nullptr)
    {
        return;
    }

    removeInternal(leafNode, key);
}

void BPlusTree::removeInternal(BPlusNode *node, int key)
{
    auto it = lower_bound(node->keys.begin(), node->keys.end(), key);
    int index = it - node->keys.begin();

    if (index < node->keys.size() && node->keys[index] == key)
    {
        node->keys.erase(node->keys.begin() + index);
        node->values.erase(node->values.begin() + index);
    }

    if (node->keys.size() < (DEGREE + 1) / 2 && node != root)
    {
        BPlusNode *parent = findParent(root, node);

        int childIndex = 0;
        while (parent->children[childIndex] != node)
        {
            childIndex++;
        }

        if (childIndex > 0 && parent->children[childIndex - 1]->keys.size() > (DEGREE + 1) / 2)
        {
            redistributeKeys(parent, childIndex - 1);
        }
        else if (childIndex < parent->keys.size() && parent->children[childIndex + 1]->keys.size() > (DEGREE + 1) / 2)
        {
            redistributeKeys(parent, childIndex);
        }
        else
        {
            if (childIndex > 0)
            {
                mergeNodes(parent, childIndex - 1);
            }
            else
            {
                mergeNodes(parent, childIndex);
            }
        }
    }
}

void BPlusTree::redistributeKeys(BPlusNode *parent, int index)
{
    BPlusNode *child = parent->children[index];
    BPlusNode *sibling = parent->children[index + 1];

    parent->keys.insert(parent->keys.begin() + index, child->keys.back());
    child->keys.pop_back();

    child->keys.push_back(sibling->keys.front());
    sibling->keys.erase(sibling->keys.begin());

    if (!child->isLeaf)
    {
        child->children[child->keys.size()] = sibling->children[0];
        for (int i = 0; i < sibling->keys.size(); ++i)
        {
            sibling->children[i] = sibling->children[i + 1];
        }
        sibling->children[sibling->keys.size()] = nullptr;
    }
}

void BPlusTree::mergeNodes(BPlusNode *parent, int index)
{
    BPlusNode *child = parent->children[index];
    BPlusNode *sibling = parent->children[index + 1];

    child->keys.push_back(parent->keys[index]);

    child->keys.insert(child->keys.end(), sibling->keys.begin(), sibling->keys.end());
    child->values.insert(child->values.end(), sibling->values.begin(), sibling->values.end());

    if (!child->isLeaf)
    {
        for (int i = 0; i < sibling->keys.size() + 1; ++i)
        {
            child->children[child->keys.size() + i] = sibling->children[i];
        }
    }

    parent->keys.erase(parent->keys.begin() + index);

    for (int i = index + 1; i < parent->keys.size() + 1; ++i)
    {
        parent->children[i] = parent->children[i + 1];
    }
    parent->children[parent->keys.size() + 1] = nullptr;
    delete sibling;
}

void BPlusTree::printTree()
{
    if (root == nullptr)
    {
        cout << "Tree is empty." << endl;
        return;
    }
    cout << "B+ Tree:" << endl;
    printTreeHelper(root, 0);
}
void BPlusTree::printTreeHelper(BPlusNode *node, int depth)
{
    if (root == nullptr)
    {
        cout << "Tree is empty" << endl;
        return;
    }
    queue<BPlusNode *> nodesQueue;
    nodesQueue.push(root);
    while (!nodesQueue.empty())
    {
        int nodesInCurrentLevel = nodesQueue.size();
        for (int i = 0; i < nodesInCurrentLevel; ++i)
        {
            BPlusNode *currentNode = nodesQueue.front();
            nodesQueue.pop();
            cout << "[";
            for (int j = 0; j < currentNode->keys.size(); ++j)
            {
                cout << "(" << currentNode->keys[j] << " " << currentNode->values[j] << ")";
                if (j != currentNode->keys.size() - 1)
                {
                    cout << ", ";
                }
            }
            cout << "] ";
            if (!currentNode->isLeaf)
            {
                for (int j = 0; j < currentNode->keys.size() + 1; ++j)
                {
                    if (currentNode->children[j] != nullptr)
                    {
                        nodesQueue.push(currentNode->children[j]);
                    }
                }
            }
        }
        cout << endl;
    }
}
vector<int> BPlusTree::inOrderTraversalHelper(BPlusNode *node)
{
    vector<int> result;
    if (node == nullptr)
    {
        return result;
    }

    if (node->isLeaf)
    {
        for (int i = 0; i < node->keys.size(); ++i)
        {
            result.push_back(node->keys[i]);
        }
    }
    else
    {
        int i;
        for (i = 0; i < node->keys.size(); ++i)
        {
            vector<int> temp = inOrderTraversalHelper(node->children[i]);
            result.insert(result.end(), temp.begin(), temp.end());
            result.push_back(node->keys[i]);
        }
        vector<int> temp = inOrderTraversalHelper(node->children[i]);
        result.insert(result.end(), temp.begin(), temp.end());
    }
    return result;
}

vector<int> BPlusTree::inOrderTraversal()
{
    vector<int> x = inOrderTraversalHelper(root);
    vector<int> uniqueElements;

    for (int i = 0; i < x.size(); ++i)
    {
        bool isDuplicate = false;
        for (int j = 0; j < uniqueElements.size(); ++j)
        {
            if (x[i] == uniqueElements[j])
            {
                isDuplicate = true;
                break;
            }
        }

        if (!isDuplicate)
        {
            uniqueElements.push_back(x[i]);
        }
    }
    return uniqueElements;
}

int BPlusTree::findKey(int key)
{
    BPlusNode *leafNode = findLeafNode(key);
    for (int i = 0; i < leafNode->keys.size(); i++)
    {
        if (leafNode->keys[i] == key)
        {
            return leafNode->values[i];
        }
    }
    return -1;
}

BPlusNode *BPlusTree::findLeafNode(int key)
{
    BPlusNode *cursor = root;
    while (!cursor->isLeaf)
    {
        int index = 0;
        while (index < cursor->keys.size() && key >= cursor->keys[index])
        {
            ++index;
        }
        cursor = cursor->children[index];
    }
    return cursor;
}

int BPlusTree::findMinKey()
{
    if (root == nullptr)
    {
        return -1;
    }

    BPlusNode *current = root;
    while (!current->isLeaf)
    {
        current = current->children[0];
    }

    return current->keys[0];
}

int BPlusTree::findMaxKey()
{
    if (root == nullptr)
    {
        return -1;
    }
    BPlusNode *current = root;
    while (!current->isLeaf)
    {
        current = current->children[current->keys.size()];
    }

    return current->keys.back();
}

void BPlusTree::clearTree(BPlusNode *node)
{
    if (node == nullptr)
    {
        return;
    }

    if (!node->isLeaf)
    {
        for (int i = 0; i < node->keys.size() + 1; ++i)
        {
            if (node->children[i] != nullptr)
            {
                clearTree(node->children[i]);
                node->children[i] = nullptr;
            }
        }
    }
    delete node;
}

void BPlusTree::clearTree()
{
    clearTree(root);
}