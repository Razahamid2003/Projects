#include "DoubleHashing.h"

HashTable::HashTable() : table(TABLE_SIZE) {}

HashTable::~HashTable() {}

int HashTable::hashFunction1(int key)
{
    return key % TABLE_SIZE;
}

int HashTable::hashFunction2(int key)
{
    return 1 + (key % 5);
}

void HashTable::insert(int key, int value)
{

    int index = hashFunction1(key);
    int step = hashFunction2(key);

    int originalIndex = index;

    while (!table[index].isEmpty)
    {
        index = (index + step) % TABLE_SIZE;

        if (index == originalIndex)
        {
            std::cout << "Hash table is full. Cannot insert." << std::endl;
            return;
        }
    }

    table[index] = KeyValue(key, value, false);
}

int HashTable::search(int key)
{
    int index = hashFunction1(key);
    int step = hashFunction2(key);

    int originalIndex = index;

    while (!table[index].isEmpty && table[index].key != key)
    {
        index = (index + step) % TABLE_SIZE;

        if (index == originalIndex)
        {
            return -1;
        }
    }

    if (!table[index].isEmpty && table[index].key == key)
    {
        return table[index].value;
    }
    else
    {
        return -1;
    }
}

void HashTable::remove(int key)
{
    int index = hashFunction1(key);
    int step = hashFunction2(key);

    int originalIndex = index;

    while (!table[index].isEmpty && table[index].key != key)
    {
        index = (index + step) % TABLE_SIZE;

        if (index == originalIndex)
        {
            std::cout << "Key not found. Removal failed." << std::endl;
            return;
        }
    }

    if (!table[index].isEmpty && table[index].key == key)
    {
        table[index].isEmpty = true;
        std::cout << "Key " << key << " removed successfully." << std::endl;
    }
    else
    {
        std::cout << "Key not found. Removal failed." << std::endl;
    }
}

void HashTable::printTable()
{
    for (int index = 0; index < TABLE_SIZE; index++)
    {
        if (!table[index].isEmpty)
        {
            std::cout << "Bucket " << index << ":  (" << table[index].key << ", " << table[index].value << ")" << std::endl;
        }
        else
        {
            std::cout << "Bucket " << index << ":  " << std::endl;
        }
    }
}