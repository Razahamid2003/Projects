// Test file for AVL Implementation
#include "../src/Part 3/avl.cpp"
#include <vector>


using namespace std;

void test_insert(shared_ptr<AVL<int,string,string>> tree);
int trav(shared_ptr<node<int,string,string>> P,int i);
int trav2(shared_ptr<node<int,string,string>> P,int i);
vector<int> find_delete(int k,vector<int> v);
void test_delete(shared_ptr<AVL<int,string,string>> tree);
void test_search(shared_ptr<AVL<int,string,string>> tree);
void test_replace(shared_ptr<AVL<int,string,string>> tree);
void test_height(shared_ptr<AVL<int,string,string>> tree);
vector<int> test_vec;
vector<int> test_vec2;
vector<int> ref_vec;
int total_score = 0;

int main()
{
	shared_ptr<AVL<int,string,string>> tree(new AVL<int,string,string>(true));

	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(34,"a", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(7,"b", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(53,"c", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(12,"d", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(68,"e", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(90,"f", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(24,"g", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(71,"h", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(13,"i", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(3,"j", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(1,"k", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(61,"l", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(18,"m", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(30,"n", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(50,"o", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(4,"p", "M")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(11,"q", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(52,"r", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(54,"r", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(56,"r", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(55,"r", "F")));
	tree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(80,"r", "F")));
	
//---------------------------------------------------------

	ref_vec.push_back(1);
	ref_vec.push_back(3);
	ref_vec.push_back(4);
	ref_vec.push_back(7);
	ref_vec.push_back(11);
	ref_vec.push_back(12);
	ref_vec.push_back(13);
	ref_vec.push_back(18);
	ref_vec.push_back(24);
	ref_vec.push_back(30);
	ref_vec.push_back(34);
	ref_vec.push_back(50);
	ref_vec.push_back(52);
	ref_vec.push_back(53);
	ref_vec.push_back(54);
	ref_vec.push_back(55);
	ref_vec.push_back(56);
	ref_vec.push_back(61);
	ref_vec.push_back(68);
	ref_vec.push_back(71);
	ref_vec.push_back(80);
	ref_vec.push_back(90);
//-----------------------------------------------------------

	cout << "Test 3 - AVL Trees"<<endl<<endl;
	test_insert(tree);
	test_search(tree);
	test_delete(tree);
	test_height(tree);

	cout << "\033[33m";
	cout << "\nTotal score: "<<total_score<<" / 30"<<endl;
	cout << "\033[0m";

	return 0;
}


// Test 2.1 - AVL Insertion
void test_insert(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 3.1 - AVL Insertion"<<endl;
	test_vec2.clear();

	// Case 2.1.1
	shared_ptr<AVL<int,string,string>>checktree (new AVL<int,string,string>(true));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(20,"Ayain", "M")));

	int num = 0;
	if (checktree->getRoot() != NULL)
	{
		num = checktree -> getRoot()->workExperience;
	}
	if (num == 20)
	{
		cout << "   Testing single insertion ... \033[32m Passed 1/1 \033[0m" << endl;
		total_score++;
	}
	else
	{
		cout << "   Testing single insertion ... Failed 0/1" << endl;
	}

	// Case 2.1.2
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(12,"Ushna", "F")));
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(10,"Abdullah","M")));

	int a=trav(checktree->getRoot(),0);
	int b=trav2(checktree->getRoot(),0);

	int arra[3] = {10,20,12};

	int score1a = 0;
	for (int i=0;i<test_vec2.size();i++)
	{
		if (test_vec2[i]==arra[i])
		{
			score1a++;
		}		
	}
	if (score1a==3)
	{
		cout << "   Testing right rotation ... \033[32m Passed 1/1 \033[0m" << endl;
		total_score++;
	}
	else
	{
		cout << "   Testing right rotation ... \033[31m Failed 0/1 \033[0m" << endl;
	}

	test_vec2.clear();

	// Case 2.1.3
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(30,"Ayesha", "F")));
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(40,"Fatima","F")));


	int arr2a[5] = {10,20,40,30,12};

	int score2a = 0;
	a=trav2(checktree->getRoot(),0);
	for (int i=0;i<test_vec2.size();i++)
	{
		if (test_vec2[i]==arr2a[i])
		{
			score2a++;
		}		
	}

	if (score2a==5)
	{
		cout << "   Testing left rotation ...  \033[32m Passed 1/1\033[0m" << endl;
		total_score++;
	}
	else
	{
		cout << "   Testing left rotation ... \033[31m Failed 0/1\033[0m" << endl;
	}

	test_vec2.clear();

	// Case 2.1.4
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(50,"Dr. Zafar", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(45,"Dr. Ihsan","M")));


	int arr3a[7] = {10,20,12,40,50,45,30};

	int score3a = 0;

	a=trav(checktree->getRoot(),0);
	b=trav2(checktree->getRoot(),0);

	for (int i=0;i<test_vec2.size();i++)
	{
		if (test_vec2[i]==arr3a[i])
		{
			score3a++;
		}		
	}

	if (score3a==7)
	{
		cout << "   Testing right-left rotation ...  \033[32m Passed 2/2\033[0m" << endl;
		total_score+=2;
	}
	else
	{
		cout << "   Testing right-left rotation ... \033[31m Failed 0/2\033[0m" << endl;
	}

	test_vec2.clear();

	// Case 2.1.5
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(5,"Maryam", "F")));
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(7,"Aleena","F")));


	int arr4a[9] = {5,10,7,20,12,40,50,45,30};

	int score4a= 0;

	a=trav(checktree->getRoot(),0);
	b=trav2(checktree->getRoot(),0);
	for (int i=0;i<test_vec2.size();i++)
	{
		if (test_vec2[i]==arr4a[i])
		{
			score4a++;
		}		
	}

	if (score4a==9)
	{
		cout << "   Testing left-right rotation ... \033[32m Passed 2/2\033[0m" << endl;
		total_score+=2;
	}
	else
	{
		cout << "   Testing left-right rotation ... \033[31m Failed 0/2\033[0m" << endl;
	}

	test_vec2.clear();

	// Case 2.1.6
	checktree->insertNode(shared_ptr<node<int,string,string>>(make_shared<node<int,string,string>>(11,"Saad", "M")));


	int arr5a[10] = {5,7,11,20,12,10,40,50,45,30};
	int score5a=0;
	a=trav(checktree->getRoot(),0);
	b=trav2(checktree->getRoot(),0);

	for (int i=0;i<test_vec2.size();i++)
	{
		if (test_vec2[i]==arr5a[i])
		{
			score5a++;
		}		
	}

	if (score5a==10)
	{
		cout << "   The ultimate test ... \033[32m Passed 3/3 \033[0m" << endl;
		total_score +=3;
	}
	else
	{
		cout << "   The ultimate test ... \033[31m Failed 0/3\033[0m" << endl;
	}

	test_vec2.clear();
	cout << endl << endl; 
	return;

}

// Test 2.3 - AVL Deletion
void test_delete(shared_ptr<AVL<int,string,string>> tree)
{
	cout<<"Test 3.3 - AVL Deletion"<<endl;

	// Case 2.3.1
	tree->deleteNode(90);
	test_vec2.clear(); 
	int numa = trav2(tree->getRoot(),0);

	int arr1[21] = {1, 3, 4, 7, 11, 12, 13, 18, 24, 30, 34, 50, 52, 53, 54, 55, 56, 61, 68, 71, 80};
	int arr1a[21] = {1,4,11,7,3,13,30,24,18,12,52,50,55,54,53,61,71,80,68,56,34};
	int scorea=0;
	if (test_vec2.size() == 21)
	{	
		scorea++;
	}
	for (int i = 0; i < (test_vec2.size()); i++)
	{
		if (test_vec2[i] == arr1a[i])
		{
			scorea++;
		}
	}
	if (scorea==22)
	{
		cout << "   Deleting leaf (no rotations)... \033[32m Passed 1/1\033[0m" << endl;
		total_score++;
	}
	else
	{
		cout << "   Deleting leaf (no rotations)... \033[31m Failed 0/1\033[0m" << endl;

	}

	// Case 2.3.2
	tree->deleteNode(54);
	test_vec2.clear(); 
	int num2a = trav2(tree->getRoot(), 0); 

	int arr2[20] = {1, 3, 4, 7, 11, 12, 13, 18, 24, 30, 34, 50, 52, 53, 55, 56, 61, 68, 71, 80};
	int arr2a[20] = {1,4,11,7,3,13,30,24,18,12,52,50,55,53,61,71,80,68,56,34};
	int score2 = 0; 
	int score2a=0;
	if (test_vec2.size() == 20)
	{
		score2a++;
	}
	for (int i = 0; i < (test_vec2.size()); i++)
	{
		if (test_vec2[i] == arr2a[i])
		{
			score2a++;
		}
	}
	if (score2a==21)
	{
		cout << "   Deleting node with one child (no rotations) ... \033[32m Passed 1/1\033[0m" << endl;
		total_score+=1;
	}
	else
	{
		cout << "   Deleting node with one child (no rotations) ... \033[31m Failed 0/1\033[0m" << endl;
	}


	// Case 2.3.4
	tree->deleteNode(7);
	test_vec2.clear(); 
	int num3a = trav2(tree->getRoot(), 0); 
	int arr3a[19] = {1,11,4,3,13,30,24,18,12,52,50,55,53,61,71,80,68,56,34};

	int score3 = 0;
	int score3a = 0;

	if (test_vec2.size() == 19)
	{
		score3a++;
	}
	for (int i = 0; i < (test_vec2.size()); i++)
	{
		if (test_vec2[i] == arr3a[i])
		{
			score3a++;
		}
	}
	if (score3a==20)
	{
		cout << "   Deleting node with two children (no rotations) ... \033[32m Passed 4/4 \033[0m" << endl;
		total_score+=4;
	}
	else
	{
		cout << "   Deleting node with two children (no rotations) ... \033[31m Failed 0/4\033[0m" << endl;
	}

	// Case 2.3.5
	tree->deleteNode(13); 
	tree->deleteNode(61);
	test_vec2.clear(); 
	int num4 = trav(tree->getRoot(), 0); 
	int num4a = trav2(tree->getRoot(), 0); 

	int arr4a[17] = {1,11,4,3,18,30,24,12,52,50,55,53,68,80,71,56,34};

	int score4 = 0; 
	int score4a =0;
	if (test_vec2.size() == 17)
	{
		score4a++;
	}
	for (int i = 0; i < (test_vec2.size()); i++)
	{
		if (test_vec2[i] == arr4a[i])
		{
			score4a++;
		}
	}
	if (score4a==18)
	{
		cout << "   Deleting leaf node (rotation required) ... \033[32m Passed 1/1 \033[0m" << endl;
		total_score+=1;
	}
	else
	{
		cout << "   Deleting leaf node (rotation required)... \033[31m Failed 0/1\033[0m" << endl;
	}

	// Case 2.3.6
	tree->deleteNode(56);
	test_vec2.clear(); 
	int num5 = trav(tree->getRoot(), 0); 
	int num5a = trav2(tree->getRoot(), 0); 

	int arr5a[16] = {1,11,4,3,18,30,24,12,50,53,52,68,80,71,55,34};


	int score5a =0;
	if (test_vec2.size() == 16)
	{
		score5a++;
	}
	for (int i = 0; i < (test_vec2.size()); i++)
	{
		if (test_vec2[i] == arr5a[i])
		{
			score5a++;
		}
	}
	if (score5a==17)
	{
		cout << "   Deleting node with two children (rotation required) ... \033[32m Passed 5/5\033[0m" << endl;
		total_score+=5;
	}
	else
	{
		cout << "   Deleting node with two (rotation required)... \033[31m Failed 0/5\033[0m" << endl;
	}

	cout << endl;
	return;


}

// Test 2.2 : Search
void test_search(shared_ptr<AVL<int,string,string>> tree)
{
	cout<< "Test 3.2 - AVL Search"  << endl;
	vector<int> ref_search;
	int counter=0;
	test_vec.clear();

	int arr[5] = {13,11,52,90,24};

	// Case 2.2.1
	for (int i = 0; i < 5; i++)
	{
		if (tree->searchNode(arr[i]))
		{
			test_vec.push_back(tree->searchNode(arr[i])->workExperience);
		}
	}

	ref_search.push_back(13);
	ref_search.push_back(11);
	ref_search.push_back(52);
	ref_search.push_back(90);
	ref_search.push_back(24);

	for (int i=0;i<test_vec.size();i++)
	{
		if (test_vec[i]==ref_search[i])
		{
			counter++;
		}

	}

	counter = (counter/ref_search.size())*5;
	if (counter == 5)
	{
		cout << "   Testing existing keys ... \033[32m Passed 3/3\033[0m" << endl;
		total_score+=3;
		counter = 3;
	}
	else
	{
		counter = 0;
		cout << "   Testing existing keys ... \033[31m Failed 0/3\033[0m" << endl;
	}

	// Case 2.2.2
	if (tree->searchNode(14) == NULL && tree->searchNode(90)!=NULL)
	{
		counter = counter + 2;
		cout << "   Testing non-existent key ... \033[32m Passed 2/2\033[0m" << endl;
		total_score+=2;
	}
	else
	{
		cout << "   Testing non-existent key ... \033[31m Failed 0/2\033[0m" << endl;
	}

	cout << endl << endl; 
	return;
}

// Test 2.4 : Height
void test_height(shared_ptr<AVL<int,string,string>> tree)
{
	cout<< endl << "Test 3.4 - AVL Height"<<endl;
	test_vec.clear();
	int counter=0;

	// Case 2.4.1 - Single Node Tree Height
	shared_ptr<AVL<int,string,string>> checktree(new AVL<int,string,string>(true));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(1,"singlennodetest", "F")));
	int h = checktree->height(checktree->getRoot());

	if (h==1)
	{
		cout << "   Testing singe node ... \033[32m Passed 1/1\033[0m" << endl;
		total_score+=1;
		counter = counter + 2;
	}
	else
	{
		cout << "   Testing single node ... \033[31m Failed 0/1\033[0m" << endl;
	}

	// Case 2.4.2 - Main Height Test
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(2,"heighttest", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(3,"heighttest", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(4,"heighttest", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(5,"heighttest", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(6,"heighttest", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(7,"heighttest", "M")));
	checktree->insertNode(shared_ptr<node<int,string,string>> (make_shared<node<int,string,string>>(8,"heighttest", "M")));
	h=checktree->height(checktree->getRoot());
	if (h==4)
	{	
		cout << "   Testing general case ... \033[32m Passed 2/2\033[0m" << endl;
		total_score+=2;
		counter= counter + 3;
	}
	else
	{
		cout << "   Testing general case ... \033[31m Failed\033[0m" << endl;
	}

	return; 

}

int trav(shared_ptr<node<int,string, string>> P,int i)
{
	if (P == NULL)
	{
		return 0;
	}
	if(P->left)
		i=trav(P->left,i);
	test_vec.push_back(P->workExperience);
	i++;
	if(P->right)
		i=trav(P->right,i);
	return i;
}

int trav2(shared_ptr<node<int,string, string>> P,int i)
{
	if (P == NULL)
	{
		return 0;
	}
	if(P->left)
		i=trav2(P->left,i);
	if(P->right)
		i=trav2(P->right,i);
	test_vec2.push_back(P->workExperience);
	i++;
	return i;
}

