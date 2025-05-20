#include <iostream>
#include <string>
#include <cstdlib>
#include <sys/param.h>
#include <unistd.h>
#include <iomanip> 

using namespace std;

int interface() 
{
    const char* parts[] = {"Exit", "General Tree", "Domain Name System", "Balanced Search Trees", "Ethical Implications", "Graphical Analysis"};
    const double marks[] = {0, 20, 25, 30, 20, 5};
    int input;

    // Print the header
    cout << string(90, '-') << endl;
    cout << left << setw(30) << "Index" << setw(50) << "Part" << setw(10) << "Marks" << endl;
    cout << string(90, '-') << endl;

    // Iterate and print each part with its index and marks
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i) 
        cout << left << setw(30) << i << setw(50) << parts[i] << setw(10) << marks[i] << endl;
    cout << string(90, '-') << endl;

    cout << "Enter the index of the test that you want to run: ";
    cin >> input;

    return input;
}

void start() 
{
    chdir("test");
    int flag = 1;

    while (flag)
    {
        switch (interface()) 
        {
        case 0:
            cout << "Exiting...\n";
            flag = 0;
            break;

        case 1:
            cout << endl;
            system("g++ test1.cpp -std=c++11 && ./a.out");
            break;
        
        case 2:
            cout << endl;
            system("g++ test2.cpp -std=c++11 && ./a.out");
            break;

        case 3:
            cout << endl;
            system("g++ test3.cpp -std=c++11 && ./a.out");
            break;
        
        case 4:
            cout << endl;
            system("g++ test4.cpp -std=c++11 && ./a.out");
            break;

        case 5:
            cout << endl;
            system("g++ test5.cpp -std=c++11 && ./a.out");
            chdir("../src/Part 5");
            system("python3 graphs.py");
            chdir("../../test");
            break;

        default:
            cout << "No such test exists\n";
            break;
        }
    }
}

int main()
{
    start();

    return 0;
}