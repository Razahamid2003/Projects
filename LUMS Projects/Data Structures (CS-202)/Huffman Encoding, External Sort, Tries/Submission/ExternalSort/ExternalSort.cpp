#include "ExternalSort.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

using namespace std;

ExternalSort::ExternalSort(std::string inputFileName, std::string outputFileName)
{
    this->inputFileName = inputFileName;
    this->outputFileName = outputFileName;
    chunkSize = 70;  // Default chunk size
    memorySize = 70; // Default memory size
}

ExternalSort::~ExternalSort()
{
    // Destructor
    deleteTempFiles();
}

void ExternalSort::sort()
{
    // To-do: Implement sorting algorithm
    // Read input file in chunks
    // Sort each chunk
}

void ExternalSort::writeChunkToTempFile(const std::vector<int> &chunk, const std::string &fileName)
{
    ofstream outfile(fileName);
    for (int num : chunk)
    {
        outfile << num << endl;
    }
    outfile.close();
}

void ExternalSort::deleteTempFiles()
{
    for (const std::string &fileName : tempFileNames)
    {
        if (std::remove(fileName.c_str()) != 0)
        {
            std::cerr << "Error: Unable to delete file: " << fileName << std::endl;
        }
        else
        {
            std::cout << "Deleted temporary file: " << fileName << std::endl;
        }
    }
    // Clear the vector of temporary file names
    tempFileNames.clear();
}

void ExternalSort::mergeSortedChunks(const std::vector<std::string> &inputFiles, const std::string &outputFileName)
{
    // To-do: Implement merging sorted chunks
    // Read sorted chunks from input files
    // Implement k-way merge logic
    // Write merged data to the output file
}
