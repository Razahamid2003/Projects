#include "../src/Part 3/avl.cpp" 
#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;

const int NUM_ELEMENTS = 10000;

void generate_avl_data()
{
    AVL<int, string, char> avl(true);   // AVL tree

    ofstream avlData("../src/Part 5/AVL_InsertionData.csv");

    avlData << "Elements,Time(us)\n";

    // Insert into AVL and measure time
    for (int i = 0; i < NUM_ELEMENTS; ++i) 
    {
        int randomExperience = rand() % NUM_ELEMENTS;
        auto start = high_resolution_clock::now();
        avl.insertNode(make_shared<node<int, string, char>>(randomExperience, "Name_" + to_string(randomExperience), 'F'));
        auto stop = high_resolution_clock::now();
        avlData << i + 1 << "," << duration_cast<microseconds>(stop - start).count() << "\n";
        if (i % 1000 == 0 || i + 1 == NUM_ELEMENTS)
            cout << "Num Elements: " << i << " Time: " << duration_cast<microseconds>(stop - start).count() << endl;
    }

    avlData.close();

    cout << "\nAVL Trees - data generation completed." << endl;
}

void generate_bst_data()
{
    AVL<int, string, char> bst(false); // BST

    ofstream bstData("../src/Part 5/BST_InsertionData.csv");

    bstData << "Elements,Time(us)\n";

    // Insert into BST and measure time
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < NUM_ELEMENTS; ++i) 
    {
        auto start = high_resolution_clock::now();
        bst.insertNode(make_shared<node<int, string, char>>(i, "Name_" + to_string(i), 'M'));
        auto stop = high_resolution_clock::now();
        bstData << i + 1 << "," << duration_cast<microseconds>(stop - start).count() << "\n";
        if (i % 1000 == 0 || i + 1 == NUM_ELEMENTS)
            cout << "Num Elements: " << i << " Time: " << duration_cast<microseconds>(stop - start).count() << endl;
    }

    bstData.close();
    cout << "\nBST - data generation completed." << endl;
}

int main()
{
    cout << "AVL - Insertions begins:\n";
    generate_avl_data();

    cout << "\nBST - Insertions begins:\n";
    generate_bst_data();

    return 0;
}