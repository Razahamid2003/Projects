#include "RecordManager.h"
#include <iostream>
#include <cassert>
#include <csignal>

using namespace std;


int main() {
    // Specify the test file name
    string testFileName = "test_data.txt";

    // Create a RecordManager instance
    RecordManager recordManager(testFileName);

    // Load records from the file into the B+ tree
    recordManager.loadRecordsFromFile();

    // Print records before any modifications
    cout << "Records before modifications:" << endl;
    recordManager.printRecords();

    // Add a record
    recordManager.addRecord(14, 140);

    // Delete a record
    recordManager.deleteRecord(5);

    // Print records after modifications
    cout << "\nRecords after modifications:" << endl;
    recordManager.printRecords();

    // Print range query
    cout << "\nRange query (key range 2 to 10):" << endl;
    recordManager.printRangeQuery(2, 10);

    // Find and print the minimum and maximum keys
    int minKey = recordManager.findMinKey();
    int maxKey = recordManager.findMaxKey();
    cout << "\nMinimum Key: " << minKey << endl;
    cout << "Maximum Key: " << maxKey << endl;

    // Save records back to the file
    recordManager.saveRecordsToFile();

    return 0;
}
