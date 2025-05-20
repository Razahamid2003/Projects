#include "RecordManager.h"
#include "BPlusTree.cpp"

RecordManager::RecordManager(const std::string &filename) : filename(filename) {}

void RecordManager::loadRecordsFromFile()
{
    std::ifstream file(filename);
    if (file.is_open())
    {
        int key, value;
        while (file >> key >> value)
        {
            bPlusTree.insert(key, value);
        }
        file.close();
    }
    else
    {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }
}

void RecordManager::saveRecordsToFile()
{
    // TODO: Implement a function to save records from the B+ tree to the file
    std::ofstream file(filename);
    if (file.is_open())
    {
        std::vector<int> keys = bPlusTree.inOrderTraversal();
        for (int key : keys)
        {
            file << key << " " << bPlusTree.findKey(key) << std::endl;
        }
        file.close();
    }
    else
    {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

void RecordManager::addRecord(int key, int value)
{
    bPlusTree.insert(key, value);
}

void RecordManager::deleteRecord(int key)
{
    // TODO: Implement a function to delete a record from the B+ tree
}

void RecordManager::printRecords()
{
    vector<int> keys = bPlusTree.inOrderTraversal();
    for (int key : keys)
    {
        std::cout << key << " " << bPlusTree.findKey(key) << std::endl;
    }
}

void RecordManager::printRangeQuery(int start, int end)
{
    // TODO: Implement a function to print records within a specified range
    std::vector<int> keys = bPlusTree.inOrderTraversal();
    for (int key : keys)
    {
        if (key >= start && key <= end)
        {
            std::cout << "Key: " << key << ", Value: " << bPlusTree.findKey(key) << std::endl;
        }
    }
}

int RecordManager::findMinKey()
{
    // TODO: Implement a function to find the minimum key in the B+ tree
    return bPlusTree.findMinKey();
}

int RecordManager::findMaxKey()
{
    // TODO: Implement a function to find the maximum key in the B+ tree
    return bPlusTree.findMaxKey();
}

void RecordManager::clearAllRecords()
{
    bPlusTree.clearTree();
}
