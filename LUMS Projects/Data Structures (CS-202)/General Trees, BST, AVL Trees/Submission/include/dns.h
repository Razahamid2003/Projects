#ifndef __DNS_H
#define __DNS_H

#include "../src/Part 1/tree.cpp"
#include <string>
#include <fstream>
#include "tree.h"

struct Webpage
{
    string TLD;
    string domain;
    vector<string> subdomains;

    Webpage(string url)
    {
        int i = 0;
        for (i = 0; i < url.length(); i++) 
        {
            if (url[i] != '.')
                domain += url[i];
            else
                break;
        }

        for (int k = 0; k < 3; k++)
            TLD += url[++i];
        
        string temp;

        for (int k = i + 2; k < url.length(); k++)
        {
            if (url[k] == '/')
            {
                subdomains.push_back(temp);
                temp = "";
            }
            else
            {
                temp += url[k];
                if (k + 1 == url.length())
                    subdomains.push_back(temp);
            }
        }
    }
    Webpage()
    {
        string TLD = "";
        string domain = "";
        vector<string> subdomains;
    }
};

class DNS
{
    shared_ptr<Tree<string, string>> domainTree;
    void getDomainPagesHelper(shared_ptr<node<string,string>> currDomain, string currentPath, shared_ptr<vector<string>> results);

public:
    DNS();
    void addWebpage(string url);
    int numRegisteredTLDs();
    vector<shared_ptr<Webpage>> getDomainPages(string domain);
    vector<shared_ptr<Webpage>> getAllWebpages(string TLD);
    shared_ptr<Tree<string, string>> getDomainTree();
    string findTLD(string domain);
    void generateOutput(vector<shared_ptr<Webpage>> pages);
};
#endif
