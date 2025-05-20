#include "DoubleHashing.h"

HashTable::HashTable() : table(TABLE_SIZE) {}

HashTable::~HashTable() {}

int HashTable::hashFunction1(int key) {
    return key % TABLE_SIZE;
}

int HashTable::hashFunction2(int key) {
    return 1 + (key % 5); 
}

void HashTable::insert(int key, int value) {
   
    // TODO: Implement the insertion logic using double hashing
    // Probing until an empty slot is found or the whole table is traversed
    // Insert the key-value pair into the table
}

int HashTable::search(int key) {
    
    // TODO: Implement the search logic using double hashing
    // Probing until the key is found or an empty slot is encountered
    // Return the associated value if the key is found, otherwise, return -1
    return -1;
}

void HashTable::remove(int key) {
    
    // TODO: Implement the removal logic using double hashing
    // Probing until the key is found or an empty slot is encountered
    // Remove the key-value pair from the table
}

void HashTable::printTable() {
    
    // TODO: Implement the logic to print the contents of the hash table
    // You can use this function to check the state of your hash table during testing
}