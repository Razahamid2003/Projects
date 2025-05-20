#include <iostream>
#include <stdlib.h>
#include <signal.h>
#include "../src/Part 1/tree.cpp"
#include <memory>

using namespace std;

int test_case = 0;
int part = 0;
int total_marks = 0;
int points = 0;
int flag = 1;

void signal_handler(int);
void print_success();
bool compare_vectors(vector<shared_ptr<node<int, string>> >, vector<int>);

int main()
{
	cout << endl;
	cout << "Test 1 - General Tree" << endl;
	cout << endl;
	signal(SIGSEGV, signal_handler);

	shared_ptr<Tree<int,string>> myTree = make_shared<Tree<int,string>>(make_shared<node<int,string>>(1,"A"));

	cout << "Test 1.1 - Insertion\n";
	part = 1;
	//Case-I : Insert first child of root
	test_case = 1;
	if(myTree->insertChild(make_shared<node<int,string>>(2,"B"), 1))
	{
		shared_ptr<node<int, string>> ptr = myTree->getRoot()->children[0];
		if(ptr->key!=2 && myTree->getRoot()->children.size() != 1)
		{
			raise(SIGSEGV);
			flag = 0;
		}
	}
	else if (!(myTree->insertChild(make_shared<node<int,string>>(2,"B"), 1)))
	{
		raise(SIGSEGV);
		flag = 0;
	}

	if (flag)
	{
		points++;
		print_success();
	}
	
	flag = 1;
	//Case-II : Insert second child of root
	test_case = 2;
	if(myTree->insertChild(make_shared<node<int,string>>(3,"C"), 1))
	{
		shared_ptr<node<int, string>> ptr = myTree->getRoot()->children[1];
		if(ptr->key!=3 && myTree->getRoot()->children.size() != 2)
		{
			raise(SIGSEGV);
			flag  = 0;
		}
	}
	else if (!(myTree->insertChild(make_shared<node<int,string>>(3,"C"), 1)))
	{
		raise(SIGSEGV);
		flag = 0;
	}

	if (flag)
	{
		points++;
		print_success();
	}

	flag = 1;
	//Case-III : Insert third child of root
	test_case = 3;
	if(myTree->insertChild(make_shared<node<int,string>>(4,"D"), 1))
	{
		shared_ptr<node<int, string>> ptr = myTree->getRoot()->children[2];
		if(ptr->key!=4 && myTree->getRoot()->children.size() != 3)
		{
			raise(SIGSEGV);
			flag = 0;
			//exit(-1);
		}
	}
	else
	{
		raise(SIGSEGV);
		flag = 0;
		//exit(-1);
	}

	if (flag)
	{
		points++;
		print_success();
	}

	flag = 1;
	//Case-IV : Insert child of first child of root
	test_case = 4;
	if(myTree->insertChild(make_shared<node<int,string>>(5,"E"), 2))
	{
		shared_ptr<node<int, string>> ptr = myTree->getRoot()->children[0];
		if(ptr->children.size() != 1 && ptr->children[0]->key != 5 && myTree->getRoot()->children.size() != 3)
		{
			raise(SIGSEGV);
			flag = 0;
			//exit(-1);
		}
	}
	else
	{
		raise(SIGSEGV);
		flag = 0;
		//exit(-1);
	}

	if (flag)
	{
		points++;
		print_success();
	}

	flag = 1;
	//Case-V : Insert child of last child of root
	test_case = 5;
	if(myTree->insertChild(make_shared<node<int,string>>(6,"F"), 4))
	{
		shared_ptr<node<int, string>> ptr = myTree->getRoot()->children[2];
		if(ptr->children.size() != 1 && ptr->children[2]->key != 6 && myTree->getRoot()->children.size() != 3)
		{
			raise(SIGSEGV);
			flag = 0;
			//exit(-1);
		}
	}
	else
	{
		raise(SIGSEGV);
		flag = 0;
		//exit(-1);
	}
	if (flag)
	{
		points++;
		print_success();
	}

	//Case-VI : Insert duplicate key
	flag = 1;
	test_case = 6;
	if(myTree->insertChild(make_shared<node<int,string>>(6,"FF"), 3))
	{
		raise(SIGSEGV);
		flag = 0;
		//exit(-1);
	}
	if(myTree->getRoot() == NULL)
	{
		raise(SIGSEGV);
		flag = 0;		
	}
	if (flag)
	{
		points++;
		print_success();
	}

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";

	total_marks = total_marks + points;


	// Test 1.2  - General Tree Search
	points = 0;
	cout << "\nTest 1.2 - Search\n";
	part = 2;

	flag = 1;
	//Case-I : Search existing node
	test_case = 1;
	shared_ptr<node<int, string>> ptr = myTree->findKey(5);
	if(ptr==NULL)
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (flag)
	{
		points = points + 4;
		print_success();
	}

	flag = 1;
	//Case-II : Search non-existent node
	test_case = 2;
	ptr = myTree->findKey(7);
	if(ptr!=NULL)
	{
		raise(SIGSEGV);
		flag = 0;
		//exit(-1);
	}
	if (myTree->getRoot() == NULL)
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (flag)
	{
		points = points + 2;
		print_success();
	}

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";
	total_marks = total_marks + points;

	flag = 1;
	points = 0;
	cout << "\nTest 1.3 - Height\n";
	part = 3;
	//Case-I : Find height of tree
	test_case = 1;
	// cout << "My height: " << myTree->findHeight() << endl;
	if(myTree->findHeight()!=2)
	{
		raise(SIGSEGV);
		flag = 0;;
	}
	else
	{	
		points = 6;
		print_success();
	}

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";
	total_marks = total_marks + points;

	points = 0;
	flag = 1;
	cout << "\nTest 1.4 - Get All Children\n";
	part = 4;
	//Case-I : Find children of root
	test_case = 1;
	vector<shared_ptr<node<int, string>> > v1 = myTree->getAllChildren(1);
	vector<int> v2;
	v2.push_back(2);
	v2.push_back(3);
	v2.push_back(4);
	if(!compare_vectors(v1,v2))
	{
		raise(SIGSEGV);
		flag = 0;;
	}
	else
	{
		print_success();
	}

	flag = 1;
	//Case-II : Find children of some node other than root
	test_case = 2;
	myTree->insertChild(make_shared<node<int,string>>(7,"G"), 3);
	myTree->insertChild(make_shared<node<int,string>>(8,"H"), 3);
	myTree->insertChild(make_shared<node<int,string>>(9,"I"), 3);
	v1.clear();
	v1 = myTree->getAllChildren(3);
	v2.clear();
	v2.push_back(7);
	v2.push_back(8);
	v2.push_back(9);
	if(!compare_vectors(v1,v2))
	{
		raise(SIGSEGV);
		flag = 0;;
	}
	else
	{
		points = points + 2;
		print_success();
	}

	flag = 1;
	//Case-III : Find children of node with no child
	test_case = 3;
	v1.clear();
	v1 = myTree->getAllChildren(8);
	v2.clear();
	if(!compare_vectors(v1,v2))
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (myTree->getRoot() == NULL)
	{
		raise(SIGSEGV);
		flag = 0;
	}
	else
	{
		points = points + 2;
		print_success();
	}

	flag = 1;
	//Case-IV : Find children of non-existent node
	test_case = 4;
	v1.clear();
	v1 = myTree->getAllChildren(10);
	v2.clear();
	if(!compare_vectors(v1,v2))
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (myTree->getRoot() == NULL)
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (flag)
	{
		points = points + 2;
		print_success();
	}

	cout << "\033[34m";
 	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";
	total_marks = total_marks + points;

	points = 0;
	flag = 1;
	cout << "\nTest 1.5 - Deletion\n";
	part = 5;
	//Case-I : Delete child with no sibling
	test_case = 1;
	if(myTree->deleteLeaf(5))
	{
		v1.clear();
		v1 = myTree->getAllChildren(2);
		v2.clear();
		if(!compare_vectors(v1,v2))
		{
			raise(SIGSEGV);
			flag = 0;
		}
	}
	else
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (flag)
	{
		points++;
		print_success();
	}

	flag = 1;
	//Case-II : Delete parent of deleted node which is now itself a leaf node
	test_case = 2;
	if(myTree->deleteLeaf(2))
	{
		v1.clear();
		v1 = myTree->getAllChildren(1);
		v2.clear();
		v2.push_back(3);
		v2.push_back(4);
		if(!compare_vectors(v1,v2))
		{

			raise(SIGSEGV);
			flag = 0;;
		}
	}
	else
	{
		raise(SIGSEGV);
		flag = 0;;
	}
	if (flag)
	{
		points = points + 2;
		print_success();
	}

	flag = 1;
	//Case-III : Delete non-leaf node
	test_case = 3;
	if(myTree->deleteLeaf(3))
	{
		raise(SIGSEGV);
		flag = 0;;
	}
	if (myTree -> getRoot() == NULL)
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (flag)
	{
		points++;
		print_success();
	}

	flag = 1;
	//Case-IV : Delete complete tree and try deleting root
	if (myTree -> getRoot() == NULL)
	{
		raise(SIGSEGV);
		flag = 0;
	}
	test_case = 4;
	myTree->deleteLeaf(7);
	myTree->deleteLeaf(8);
	myTree->deleteLeaf(9);
	myTree->deleteLeaf(3);
	myTree->deleteLeaf(6);
	myTree->deleteLeaf(4);
	if(myTree->deleteLeaf(1))
	{
		raise(SIGSEGV);
		flag = 0;
	}
	if (flag)
	{
		points = points + 2;
		print_success();
	}

	cout << "\033[34m";
	cout << "\nTotal Points: " << points << " / 6" << endl;
	cout << "\033[0m";
	total_marks = total_marks + points;
	float scaled_points = ((float)total_marks/30.0)*20;

	cout << "\033[33m";
	cout << endl << "Total Marks: " << total_marks << " / 30" << endl;
	cout << endl << "Scaled Marks: " << scaled_points << " / 20" << endl;

	cout << "\033[0m";

	if (total_marks == 30)
	{
		cout << "\nWoohoo! Take a break and onto the next one :)" << endl;
	}

	cout << endl;
	return 0;
}

void signal_handler(int signum)
{
	cout << "   1." << part << "." << test_case  << ":\033[31m Failed\033[0m\n";
}

void print_success()
{
	cout << "   1." << part << "." << test_case  << ":\033[32m Passed\033[0m\n";
}


bool compare_vectors(vector<shared_ptr<node<int, string>> > v1, vector<int> v2)
{
	if(v1.size()!=v2.size())
		return false;

	for(int i=0; i<v1.size(); i++)
	{
		bool isFound = false;
		for(int j=0; j<v2.size(); j++)
		{
			if(v1[i]->key==v2[j])
			{
				isFound = true;
				continue;
			}
		}
		if(!isFound)
			return false;
	}
	return true;
}