#include <iostream>
#include <vector>
#include <future>
#include <chrono>
#include <memory>
#include <thread>
#include "DoubleHashing.h"
#include "DoubleHashing.cpp"
using namespace std;

bool timeOut;
std::promise<bool> done;

void timer(std::future<bool> done_future) {
    std::chrono::seconds span(3);
    if (done_future.wait_for(span) == std::future_status::timeout) {
        timeOut = true;
    }
}

double test(std::vector<int> keys, std::vector<int> values, std::promise<bool> done_future) {
    using namespace std::chrono;

    std::shared_ptr<HashTable> myTable(new HashTable());
    double score = 0;
    high_resolution_clock::time_point timeStart = high_resolution_clock::now();

    std::cout << "Starting Tests" << std::endl;
    std::cout << "\nTesting Insert:    ";

    // Inserting entries into the hash table
    for (size_t i = 0; i < keys.size(); i++) {
        myTable->insert(keys[i], values[i]);
        myTable->printTable();
        if (timeOut) {
            std::cout << "Failed! The test timed out." << std::endl;
            return score;
        }
    }

    score += 10;
    std::cout << "Passed!" << std::endl;
    std::cout << "Testing Search:    ";

    // Searching entries from the hash table
    for (size_t i = 0; i < keys.size(); i++) {
        int result = myTable->search(keys[i]);
        if (result != values[i]) {
            std::cout << "Failed!" << std::endl;
            return score;
        }

        if (timeOut) {
            std::cout << "Failed! The test timed out." << std::endl;
            return score;
        }
    }

    score += 10;
    std::cout << "Passed!" << std::endl;
    std::cout << "Testing Remove:    ";

    // Deleting entries from the hash table
    for (size_t i = 0; i < keys.size(); i++) {
        myTable->remove(keys[i]);
        if (timeOut) {
            std::cout << "Failed! The test timed out." << std::endl;
            return score;
        }
    }

    // Check if keys are not found after removal
    for (size_t i = 0; i < keys.size(); i++) {
        if (myTable->search(keys[i]) != -1) {
            std::cout << "Failed!" << std::endl;
            return score;
        }

        if (timeOut) {
            std::cout << "Failed! The test timed out." << std::endl;
            return score;
        }
    }

    score += 10;
    std::cout << "Passed!" << std::endl;

    done_future.set_value(true);
    high_resolution_clock::time_point timeEnd = high_resolution_clock::now();
    duration<double> totalTime = duration_cast<duration<double>>(timeEnd - timeStart);

    std::cout << "\nTest Passed in: " << totalTime.count() << " seconds." << std::endl;
    return score;
}

int main() {
    std::vector<int> keys = {5, 2, 8, 1, 3, 11, 7, 20, 15, 25, 6, 10, 17, 22, 30};
    std::vector<int> values = {25, 12, 40, 5, 15, 55, 35, 100, 75, 125, 30, 50, 85, 110, 150};


    timeOut = false;

    future<bool> done_future = done.get_future();
    thread first(timer, std::move(done_future));
    double score = test(keys, values, std::move(done));
    first.join();

    std::cout << "Total Score: " << score << "/30" << std::endl;

    return 0;
}
