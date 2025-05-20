#include <fstream>
#include <sstream>

#define RECORDMANAGER_H

#include "BPlusTree.h"

class RecordManager {
private:
    BPlusTree bPlusTree;
    std::string filename;

public:
    RecordManager(const std::string& filename);

    void loadRecordsFromFile();
    void saveRecordsToFile();
    
    void addRecord(int key, int value);
    void deleteRecord(int key);
    
    void printRecords();
    void printRangeQuery(int start, int end);

    int findMinKey();
    int findMaxKey();

    void clearAllRecords();
};
