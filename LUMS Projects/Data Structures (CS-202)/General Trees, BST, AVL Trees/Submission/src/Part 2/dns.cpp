#include "../../include/dns.h"

using namespace std;

// You aren't allowed to edit these method declarations or declare global variables
// .....................................................................................

DNS::DNS()
{
    domainTree = make_shared<Tree<string, string>>(make_shared<node<string, string>>("ROOT", "ROOT"));
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
            line = ptr->domain + "." + ptr->TLD;
            if (ptr->subdomains.size())
            {
                for (auto& element : ptr->subdomains)
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
{
    Webpage temp(url);
    string tld = temp.TLD;
    string domain = temp.domain;
    vector <string> sub_domain = temp.subdomains;

    shared_ptr<node<string, string>> root = domainTree->getRoot();
    shared_ptr<node<string, string>> node_tld = make_shared<node<string, string>>(tld, tld);
    domainTree->insertChild(node_tld, root->key);

    shared_ptr<node<string, string>> node_domain = make_shared<node<string, string>>(domain, domain);
    domainTree->insertChild(node_domain, node_tld->key);

    shared_ptr<node<string, string>> node_subdomain = make_shared<node<string, string>>(sub_domain[0], sub_domain[0]);
    domainTree->insertChild(node_subdomain, node_domain->key);


    for (int i = 1; i < sub_domain.size(); i++)
    {
        shared_ptr<node<string, string>> node_sub_subdomain = make_shared<node<string, string>>(sub_domain[i], sub_domain[i]);
        domainTree->insertChild(node_sub_subdomain, node_subdomain->key);
        node_subdomain = node_sub_subdomain;

    }
}

int DNS::numRegisteredTLDs()
{
    vector<shared_ptr<node<string, string>>> children = domainTree->getAllChildren("ROOT");
    return children.size();
}

vector<shared_ptr<Webpage>> DNS::getAllWebpages(string TLD)
{
    vector<shared_ptr<node<string, string>>> domains = domainTree->getAllChildren(TLD);
    vector<shared_ptr<Webpage>> pages;
    for (int i = 0; i < domains.size(); i++)
    {
        shared_ptr<Webpage> temp;
        temp->TLD = TLD;
        temp->domain = domains[i]->key;

        shared_ptr<node<string, string>> current = domains[i];
        vector<string> sub_domains;
        while (current != NULL || sub_domains.empty() == false)
        {
            while (current != NULL)
            {
                sub_domains.push_back(current->key);
                current = current;
            }
        }
        pages.push_back(temp);
    }
    return pages;
}

vector<shared_ptr<Webpage>> DNS::getDomainPages(string domain)
{
    string tld = findTLD(domain);
    vector<shared_ptr<Webpage>> allpages = getAllWebpages(tld);
    for (int i = 0; i < allpages.size(); i++)
    {
        
    }
    string currentPath = domain + "." + tld;
    vector<shared_ptr<node<string, string>>> results = domainTree->getAllChildren(domain);

}

void DNS::getDomainPagesHelper(shared_ptr<node<string,string>> currDomain, string currentPath, shared_ptr<vector<string>> results)
{
    
}


shared_ptr<Tree<string, string>> DNS::getDomainTree()
{
    return domainTree;
}

string DNS::findTLD(string domain)
{
    vector<shared_ptr<node<string, string>>> tlds = domainTree->getAllChildren("ROOT");
    vector<shared_ptr<node<string, string>>> domains;
    string temp = "";
    for (int i = 0; i < tlds.size(); i++)
    {
        temp = tlds[i]->key;
        domains = domainTree->getAllChildren(temp);
        for (int i = 0; i < domains.size(); i++)
        {
            if (domains[i]->key == domain)
            {
                return temp;
            }
        }
    }
    temp = "";
    return temp;
}