#include "../src/Part 3/avl.cpp"
#include "../src/Part 4/ethics.cpp"
#include <vector>


using namespace std;

void test_num_to_shortlist(shared_ptr<AVL<int,string,string>> tree);
void test_right_most(shared_ptr<AVL<int,string,string>> tree);
void test_in_order(shared_ptr<AVL<int,string,string>> tree);
void test_level_order(shared_ptr<AVL<int,string,string>> tree);
void test_bias(shared_ptr<AVL<int,string,string>> tree);


int main()
{
	shared_ptr<AVL<int,string,string>> tree(new AVL<int,string,string>(true));

	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(34,"a", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(7,"b", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(53,"c", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(12,"d", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(68,"e", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(90,"f", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(24,"g", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(71,"h", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(13,"i", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(3,"j", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(1,"k", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(61,"l", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(18,"m", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(30,"n", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(50,"o", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(4,"p", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(11,"q", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(52,"r", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(54,"s", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(56,"t", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(55,"u", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(80,"v", "M")));
	

	cout << "Test 4 - Ethical Implications"<<endl<<endl;
	test_num_to_shortlist(tree);
	test_right_most(tree);
	test_in_order(tree);
	test_level_order(tree);
	test_bias(tree);

	cout << "See you in the next one!" << endl; 
	return 0;
}



void test_num_to_shortlist(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 4.1 - Number to Shortlist"<<endl;

	// Case 3.1.1
	shared_ptr<AVL<int,string,string>>checktree (new AVL<int,string,string>(false));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(20,"Saad", "M")));

	int num = 0;
	if (checktree->getRoot() != nullptr){

		num = checktree->number_to_shortlist(checktree->getRoot()); 

		if (num == 1)
		{
			cout << "   Single node case ... \033[32m Passed\033[0m" << endl;
		}
		else
		{
			cout << "   Singe node case ... \033[31m Failed\033[0m" << endl;
		}
	}

	// Case 3.1.2
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(10,"Haseeb","M")));
	num = checktree->number_to_shortlist(checktree->getRoot());

	if (num==1)
	{
		cout << "   Left subtree case ... \033[32m Passed\033[0m" << endl;
	}
	else
	{
		cout << "   Left subtree case ... \033[31m Failed\033[0m" << endl;
	}


	// Case 3.1.3
	num = tree->number_to_shortlist(tree->getRoot()); 

	if (num==5)
	{
		cout << "   General case ...  \033[32m Passed\033[0m" << endl;
	}
	else
	{
		cout << "   General case ... \033[31m Failed\033[0m" << endl;
	}
	
	cout << endl; 
	return;

}

void test_right_most(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 4.2 - Right Most"<<endl;

	// Case 3.2.1
	shared_ptr<AVL<int,string,string>>checktree (new AVL<int,string,string>(false));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(20,"Saad", "M")));
	vector<int> test_vector; test_vector.push_back(20); 
	vector<int> solution_vector = checktree->right_most(checktree->getRoot());
	if(test_vector == solution_vector){
		cout << "   Single node case ... \033[32m Passed\033[0m" << endl; 
	} else {
		cout << "   Single node case ... \033[31m Failed\033[0m" << endl; 
	}

	// Case 3.2.2
	test_vector.clear(); 
	test_vector.push_back(34); 
	test_vector.push_back(56); 
	test_vector.push_back(68); 
	test_vector.push_back(80); 
	test_vector.push_back(90);
	solution_vector.clear(); 
	solution_vector = tree->right_most(tree->getRoot()); 

	if(test_vector == solution_vector){
		cout << "   General case ... \033[32m Passed\033[0m"; 
	} else {
		cout << "   General case ... \033[31m Failed\033[0m"; 
	}
		

	cout << endl << endl; 
	return; 
}

void test_in_order(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 4.3 - In Order"<<endl;

	// Case 3.3.1
	shared_ptr<AVL<int,string,string>>checktree (new AVL<int,string,string>(false));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(20,"Saad", "M")));
	vector<int> test_vector; test_vector.push_back(20); 
	vector<int> solution_vector = checktree->in_order(checktree->getRoot());



	if(test_vector == solution_vector){
		cout << "   Single node case ... \033[32m Passed\033[0m" << endl; 
	} else {
		cout << "   Single node case ... \033[31m Failed\033[0m" << endl; 
	}

	// Case 3.2.2
	test_vector.clear(); 
	test_vector.push_back(1); 
	test_vector.push_back(3); 
	test_vector.push_back(4); 
	test_vector.push_back(7); 
	test_vector.push_back(11);
	solution_vector.clear(); 
	solution_vector = tree->in_order(tree->getRoot()); 


	if(test_vector == solution_vector){
		cout << "   General case ... \033[32m Passed\033[0m"; 
	} else {
		cout << "   General case ... \033[31m Failed\033[0m"; 
	}

	cout << endl << endl; 
	return; 
}

void test_level_order(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 4.4 - Level Order"<<endl;

	// Case 3.4.1
	shared_ptr<AVL<int,string,string>>checktree (new AVL<int,string,string>(false));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(20,"Saad", "M")));
	vector<int> test_vector; test_vector.push_back(20); 
	vector<int> solution_vector = checktree->level_order(checktree->getRoot());

	if(test_vector == solution_vector){
		cout << "   Single node case ... \033[32m Passed\033[0m" << endl; 
	} else {
		cout << "   Single node case ... \033[31m Failed\033[0m" << endl; 
	}

	// Case 3.2.2
	test_vector.clear(); 
	test_vector.push_back(34); 
	test_vector.push_back(12); 
	test_vector.push_back(56); 
	test_vector.push_back(3); 
	test_vector.push_back(18);
	solution_vector.clear(); 
	solution_vector = tree->level_order(tree->getRoot()); 

	if(test_vector == solution_vector){
		cout << "   General case ... \033[32m Passed\033[0m";
	} else {
		cout << "   General case ... \033[31m Failed\033[0m"; 
	}

	cout << endl << endl; 
	return; 
}


void test_bias(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 4.5 - Bias"<<endl;

	// Case 3.4.1
	vector<float> solution_vector = tree->bias(tree->getRoot());
	vector<float> test_vector = {0.7, 0.6, 0.6, 1, 1, 0.5, 0, 1, 0}; 
	if(test_vector == solution_vector){
		cout << "   Bias test ... \033[32m Passed\033[0m";
	} else {
		cout << "   Bias test ... \033[31m Failed\033[0m"; 
	}

	cout << endl << endl; 
	return; 
}