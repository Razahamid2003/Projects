#ifndef EXTERNAL_SORT_H
#define EXTERNAL_SORT_H

#include <string>
#include <vector>

using namespace std;


class ExternalSort {
public:
    // Constructor
    ExternalSort(std::string inputFileName, std::string outputFileName);

    // Destructor
    ~ExternalSort();

    // External sorting algorithm
    void sort();

private:
    std::string inputFileName;
    std::string outputFileName;
    int chunkSize; // Size of each chunk to fit into memory
    int memorySize; // Total memory size available
    std::vector<std::string> tempFileNames;

    // Helper functions
    void deleteTempFiles();
    void mergeSortedChunks(const vector<string>& inputFiles, const string& outputFileName);
    void writeChunkToTempFile(const vector<int>& chunk, const string& fileName);
};

#endif
