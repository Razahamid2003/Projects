#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <iostream>
#include <vector>


class HashTable {
private:
    static const int TABLE_SIZE = 20; 

   
   struct KeyValue {
    int key;
    int value;
    bool isEmpty;

    KeyValue() : key(0), value(0), isEmpty(true) {}
    KeyValue(int k, int v, bool empty) : key(k), value(v), isEmpty(empty) {}

    bool operator==(const KeyValue& other) const {
        return key == other.key && value == other.value && isEmpty == other.isEmpty;
    }
};

    std::vector<KeyValue> table;


    // Hash functions to map a key to an index in the table
    int hashFunction1(int key);
    int hashFunction2(int key);

public:
    HashTable();
    ~HashTable();

    // Insert a key-value pair into the hash table
    void insert(int key, int value);

    // Search for a key and return its associated value
    int search(int key);

    // Delete a key and its associated value from the hash table
    void remove(int key);

    // Print the contents of the hash table
    void printTable();
};

#endif
