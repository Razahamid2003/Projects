#include "ExternalSort.h"
#include <cassert>
#include <fstream>
#include <vector>
#include <iostream>
#include <random>
#include <set>

void createInputFile(const std::string& filename, const std::vector<int>& data) {
    std::ofstream outFile(filename);
    for (int value : data) {
        outFile << value << std::endl;
    }
    outFile.close();
}

bool isSorted(const std::string& filename) {
    std::ifstream inFile(filename);
    int prevValue = 0;
    int value;
    while (inFile >> value) {
        if (value < prevValue) {
            return false;
        }
        prevValue = value;
    }
    return true;
}
bool checkSorting(const std::string& fileName) {
    std::ifstream inputFile(fileName);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << fileName << std::endl;
        return false;
    }

    int expectedValue = 1;
    int value;
    int count=0;
    while (inputFile >> value) {
        count++;
        if (value != expectedValue) {
            std::cerr << "Error: Unexpected value found in file " << fileName << std::endl;
            return false;
        }
        ++expectedValue;
    }
    if (count!=350){
        return false;
    }

    inputFile.close();
    return true;
}

int main() {
   int minRange = 1;
    int maxRange = 350;
    int numIntegers = 350; 


    std::ofstream outputFile("random_input.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open output file" << std::endl;
        return 1;
    }

   
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(minRange, maxRange);

    std::set<int> generatedNumbers;

    while (generatedNumbers.size() < numIntegers) {
        int randomNumber = dis(gen); 
        if (generatedNumbers.insert(randomNumber).second) { 
            outputFile << randomNumber << std::endl;
        }
    }

   
    outputFile.close();


    std::cout << "Random integers file created successfully." << std::endl;
    std::string inputFileName = "input.txt";
    std::string outputFileName = "output.txt";

    // Create input file with random numbers
    std::vector<int> inputData;
    int num;
    std::ifstream inputFile("random_input.txt");
    while (inputFile >> num) {
        inputData.push_back(num);
    }
    inputFile.close();
    createInputFile(inputFileName, inputData);

    ExternalSort externalSort(inputFileName, outputFileName);
    externalSort.sort();


    bool sorted = checkSorting(outputFileName);
    remove(inputFileName.c_str());
    remove(outputFileName.c_str());
    if (sorted) {
        std::cout << "External sorting test passed! 30/30" << std::endl;
    } else {
        std::cout << "External sorting test failed!" << std::endl;
    }

    return 0;
}
