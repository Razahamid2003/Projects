#include <iostream>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <signal.h>
#include <algorithm>
#include <random>
#include <array>
#include "../src/Part 2/dns.cpp"
#include <chrono>

using namespace std;

int part, test_case, total_points = 0;

void signal_handler(int signum)
{
	cout << "   2." << part << "." << test_case  << ":\033[31m Failed\033[0m\n";
}

void print_success()
{
	cout << "   2." << part << "." << test_case  << ":\033[32m Passed\033[0m\n";
}

void load_webpages(int fileno, DNS* tester)
{
    fstream f1;
    string s1;

    f1.open("urls" + to_string(fileno) + ".txt",ios::in);
    getline(f1, s1);

    int length = stoi(s1);
    for (int i = 0; i < length; ++i)
    {
        getline(f1, s1);
        tester->addWebpage(s1);
    }

    f1.close();
}

void output_match(string filename, int num, int pt, int& points)
{
    part = num;
    test_case = pt;

    fstream f1, f2;
    string s1, s2;

    f1.open(filename + ".txt", ios::in);
    f2.open("output.txt", ios::in);

    getline(f1, s1);
    getline(f2, s2);

    if (s1 != s2)
    {
        raise(SIGSEGV);
        return;
    }
    
    int length = stoi(s1);

    for (int i = 0; i < length; i++)
    {
        getline(f1, s1);
        getline(f2, s2);

        if (s1 != s2)
        {
            raise(SIGSEGV);
            return;
        }
    }

    f1.close();
    f2.close();

    points++;
    print_success();
}

void test_add_page()
{
    cout << "\nTest 2.1 - Adding Webpages" << endl;
    DNS d1;
    int points = 0, flag;

    cout<<"\nlil nig\n";
    d1.addWebpage("quillbot.com/grammar-check/file1");
    part = test_case = 1;
    if (d1.getDomainTree()->getRoot()->key == "ROOT" && d1.getDomainTree()->getRoot()->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->key == "com" && d1.getDomainTree()->getRoot()->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->key == "quillbot" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->key == "grammar-check" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[0]->key == "file1")
    {
        points++;
        flag = 1;
        print_success();
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    test_case = 2;
    cout<<"\nlil nig\n";
    d1.addWebpage("quillbot.com/flow");
    if (flag && (d1.getDomainTree()->getRoot()->key == "ROOT" && d1.getDomainTree()->getRoot()->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->key == "com" && d1.getDomainTree()->getRoot()->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->key == "quillbot" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->key == "grammar-check" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[1]->key == "flow" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[0]->key == "file1"))
    {
        points++;
        flag = 1;
        print_success();
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    test_case = 3;
    cout<<"\nlil nig\n";
    d1.addWebpage("zambeel.net/login/bye");
    if (flag && (d1.getDomainTree()->getRoot()->key == "ROOT" && d1.getDomainTree()->getRoot()->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->key == "com" && d1.getDomainTree()->getRoot()->children[1]->key == "net" && d1.getDomainTree()->getRoot()->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->key == "quillbot" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->key == "grammar-check" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[1]->key == "flow" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[0]->key == "file1" && d1.getDomainTree()->getRoot()->children[1]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->key == "zambeel" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->key == "login" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[0]->key == "bye"))
    {
        points++;
        flag = 1;
        print_success();
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    test_case = 4;
    cout<<"\nlil nig\n";
    d1.addWebpage("zambeel.net/login/bye2/foo");
    if (flag && (d1.getDomainTree()->getRoot()->key == "ROOT" && d1.getDomainTree()->getRoot()->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->key == "com" && d1.getDomainTree()->getRoot()->children[1]->key == "net" && d1.getDomainTree()->getRoot()->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->key == "quillbot" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->key == "grammar-check" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[1]->key == "flow" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[0]->key == "file1" && d1.getDomainTree()->getRoot()->children[1]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->key == "zambeel" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->key == "login" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[0]->key == "bye" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->key == "bye2" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children[0]->key == "foo"))
    {
        points++;
        flag = 1;
        print_success();
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    test_case = 5;
    cout<<"\nlil nig\n";
    d1.addWebpage("zambeel.net/login/bye2/tom");
    if (flag && (d1.getDomainTree()->getRoot()->key == "ROOT" && d1.getDomainTree()->getRoot()->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->key == "com" && d1.getDomainTree()->getRoot()->children[1]->key == "net" && d1.getDomainTree()->getRoot()->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->key == "quillbot" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->key == "grammar-check" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[1]->key == "flow" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[0]->key == "file1" && d1.getDomainTree()->getRoot()->children[1]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->key == "zambeel" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->key == "login" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[0]->key == "bye" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->key == "bye2" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children[0]->key == "foo" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children[1]->key == "tom"))
    {
        points++;
        flag = 1;
        print_success();
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    test_case = 6;
    d1.addWebpage("quillbot.com/grammar-check/file2/phi");
    if (flag && (d1.getDomainTree()->getRoot()->key == "ROOT" && d1.getDomainTree()->getRoot()->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->key == "com" && d1.getDomainTree()->getRoot()->children[1]->key == "net" && d1.getDomainTree()->getRoot()->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->key == "quillbot" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->key == "grammar-check" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[1]->key == "flow" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[0]->key == "file1" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[1]->key == "file2" && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[1]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[0]->children[0]->children[0]->children[1]->children[0]->key == "phi" && d1.getDomainTree()->getRoot()->children[1]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->key == "zambeel" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children.size() == 1 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->key == "login" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[0]->key == "bye" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->key == "bye2" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children.size() == 2 && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children[0]->key == "foo" && d1.getDomainTree()->getRoot()->children[1]->children[0]->children[0]->children[1]->children[1]->key == "tom"))
    {
        points++;
        flag = 1;
        print_success();
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    total_points += points;

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";
}

void test_all_webpages()
{
    DNS t2, t3;
    load_webpages(20, &t2);
    load_webpages(50, &t3);
    int points = 0;

    cout << "\nTest 2.3 - Webpages Retrival (by TLD)" << endl;

    t2.generateOutput(t2.getAllWebpages("org"));
    output_match("org20", 3, 1, points);
    t2.generateOutput(t2.getAllWebpages("com"));
    output_match("com20", 3, 2, points);
    t2.generateOutput(t2.getAllWebpages("net"));
    output_match("net20", 3, 3, points);
    t3.generateOutput(t3.getAllWebpages("org"));
    output_match("org50", 3, 4, points);
    t3.generateOutput(t3.getAllWebpages("com"));
    output_match("com50", 3, 5, points);
    t3.generateOutput(t3.getAllWebpages("net"));
    output_match("net50", 3, 6, points);

    total_points += points;

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";
}

void test_domain_pages()
{
    DNS t2;
    fstream f1, f2;
    string s1, s2;
    int points = 0, flag;
    part = 2;

    cout << "\nTest 2.2 - Webpages Retrival (by domain)" << endl;
    
    load_webpages(50, &t2);
    string domains[] = {"fitnessfreaks", "petparadise", "gourmetguide", "greenearth", "cinespot"};
    int num_domains[] = {3,2,3,3,3};
    f2.open("answers.txt", ios::in);

    for (int i = 0; i < 5; i++)
    {
        flag = 1;
        test_case = i + 1;
        t2.generateOutput(t2.getDomainPages(domains[i]));
        f1.open("output.txt", ios::in);
        getline(f1, s1);
        int length = stoi(s1);
        if (length == num_domains[i])
        {
            for (int k = 0; k < length; k++)
            {
                getline(f1, s1);
                getline(f2, s2);
                if (s1 != s2)
                {
                    raise(SIGSEGV);
                    flag = 0;
                    break;
                }
            }
        }
        else
            raise(SIGSEGV);
        
        if (flag)
        {
            points++;
            print_success();
        }

        f1.close();
    }
    f2.close();

    total_points += points;

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 5" << endl;
	cout << "\033[0m";
}

void test_remaining()
{
    cout << "\nTest 2.4 - Finding TLD corresponding to a domain name" << endl;
    DNS t2;
    int flag = 1, points = 0;
    part = 4; test_case = 1;

    load_webpages(20, &t2);

    string domains[] = {"clientvoices", "example", "portfoliohub", "example", "educateweb", "marketplace", "conferences", "insights", "technews", "globalconnect", "conferences", "helpdesk", "projectverse", "connectsite"};
    string TLDs[] = {"org", "com", "org", "com", "com", "net", "org", "net", "net", "com", "org", "org", "net", "com"};

    array<int, 14> pos = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(pos.begin(), pos.end(), default_random_engine(seed));

    for (int i : pos)
    {
        if (t2.findTLD(domains[i]) != TLDs[i])
        {
            raise(SIGSEGV);
            flag = 0;
            break;
        }
    }

    if (flag)
    {
        print_success();
        points += 2;
    }

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 2" << endl;
	cout << "\033[0m";

    total_points += points;

    points = 0; part = 5; flag = 1;
    cout << "\nTest 2.5 - Number of Registered TLDs" << endl;
    DNS t3, t1;

    if (t3.numRegisteredTLDs() == 0)
    {
        flag = 1;
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }
    if (flag && t2.numRegisteredTLDs() == 3)
    {
        flag = 1;
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    load_webpages(50, &t3);
    if (flag && t3.numRegisteredTLDs() == 3)
    {
        flag = 1;
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    t1.addWebpage("zambeel.net/login/bye2/tom");
    if (flag && t1.numRegisteredTLDs() == 1)
    {
        flag = 1;
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }
    t1.addWebpage("quillbot.com/login/bye2/tom");
    if (flag && t1.numRegisteredTLDs() == 2)
    {
        flag = 1;
    }
    else
    {
        raise(SIGSEGV);
        flag = 0;
    }

    if (flag)
    {
        print_success();
        points++;
    }

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 1" << endl;
	cout << "\033[0m";

    total_points += points;

}


int main()
{
    cout << "Test 2 - Domain Name System\n";
    test_add_page();
    test_domain_pages();
    test_all_webpages();
    test_remaining();

	cout << "\033[33m";
    float scaled_points = ((float)total_points/20.0)*25;
	cout << endl << "Total Marks: " << total_points << " / 20" << endl;
	cout << endl << "Scaled Marks: " << scaled_points << " / 25" << endl;

	cout << "\033[0m";

	if (total_points == 20)
	{
		cout << "\nWoohoo! Take a break and onto the next one :)" << endl;
	}


    return 0;
}

