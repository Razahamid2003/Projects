#ifndef __POLYCALCULATOR_H
#define __POLYCALCULATOR_H

#include "LinkedList.cpp" 

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

template <class T>
class PolyCalculator
{
    LinkedList<T> list1; 
    LinkedList<T> list2;
    string stored;
    //You may add more helper variables or helper functions

public:

   
    // Constructor
    PolyCalculator();

    // Destructor
    ~PolyCalculator();

    //Required Methods

    // Input Function
    void input(PolyCalculator<int> & p1);//O(n^2)

    //FUnction to search for highest exponent in poly
    int highestExp(string poly);//O(n)

    //String sorting Function
    string sort(string to_be_sorted);//O(n^2)

    // Display Function
    void display(PolyCalculator<int> &p1);//O(n^2)

    // Addition Function
    void add(PolyCalculator<int>& p1);//O(n^2)

    // Subtraction Function
    void sub(PolyCalculator<int>& p1);//O(n^2)

    // Multiplication Function
    void mul(PolyCalculator<int>& p1);//O(n^2)

    // Read Data from File Function
    void readData(string filename,PolyCalculator<int>& p1);//O(n^2)

    // Write Data to File Function
    void writeData(string filename,PolyCalculator<int>& p1);//O(n^2)

    // Parse Polynomial Expression Function
    bool parse(string str, PolyCalculator<int>& p1);//O(n^2)

    //Your defined functions here
    
};

#endif
