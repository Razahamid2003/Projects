#include "../../include/dns.h"

using namespace std;

// You aren't allowed to edit these method declarations or declare global variables
// .....................................................................................

DNS::DNS()
{
    domainTree = make_shared<Tree<string,string>>(make_shared<node<string,string>>("ROOT", "ROOT"));
}

void DNS::generateOutput(vector<shared_ptr<Webpage>> pages)
{
    ofstream file("output.txt", ios::out);
    shared_ptr<Webpage> ptr;

    string line;
    int length = pages.size();

    for (int i = 0; i <= length; i++)
    {
        if (!i)
            file << length << endl;
        else
        {
            ptr = pages.at(i - 1);
            line =  ptr->domain + "." + ptr->TLD;
            if (ptr->subdomains.size())
            {
                for (auto &element : ptr->subdomains)
                    line += "/" + element;
            }
            file << line << endl;
        }
    }

    file.close();
}

// .....................................................................................
// To Implement

void DNS::addWebpage(string url)
{}

int DNS::numRegisteredTLDs()
{}

vector<shared_ptr<Webpage>> DNS::getAllWebpages(string TLD)
{}

vector<shared_ptr<Webpage>> DNS::getDomainPages(string domain)
{}

void DNS::getDomainPagesHelper(shared_ptr<node<string,string>> currDomain, string currentPath, shared_ptr<vector<string>> results)
{}

shared_ptr<Tree<string, string>> DNS::getDomainTree()
{}

string DNS::findTLD(string domain)
{}

