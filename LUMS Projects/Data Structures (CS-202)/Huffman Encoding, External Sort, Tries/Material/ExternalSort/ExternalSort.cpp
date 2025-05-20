#include "ExternalSort.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

using namespace std;

ExternalSort::ExternalSort(std::string inputFileName, std::string outputFileName) {
    this->inputFileName = inputFileName;
    this->outputFileName = outputFileName;
    chunkSize = 70; // Default chunk size
    memorySize = 70; // Default memory size
    
}

ExternalSort::~ExternalSort() {
    // Destructor
    deleteTempFiles();
}

void ExternalSort::sort() {
    // To-do: Implement sorting algorithm
    // Read input file in chunks
    // Sort each chunk 
 
}

void ExternalSort::writeChunkToTempFile(const std::vector<int>& chunk, const std::string& fileName) {
    // To-do: Implement writing chunk to temporary file
}

void ExternalSort::deleteTempFiles() {
    // To-do: Implement deletion of temporary files
}

void ExternalSort::mergeSortedChunks(const std::vector<std::string>& inputFiles, const std::string& outputFileName) {
    // To-do: Implement merging sorted chunks
    // Read sorted chunks from input files
    // Implement k-way merge logic 
    // Write merged data to the output file
}
