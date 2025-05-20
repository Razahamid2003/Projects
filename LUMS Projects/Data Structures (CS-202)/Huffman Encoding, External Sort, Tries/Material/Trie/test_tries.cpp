#include "trie.cpp"

double insertCalculator(shared_ptr<Node> myNode, int val);
double insertionTest(trie &myTrie, string array[])
{
    double points = 0;
    for (int i = 0; i < 4; i++)
    {
        myTrie.insertWord(array[i]);
    }
    double val1 = 0, val2 = 0, val3 = 0, val4 = 0, val5 = 0;
    val1 = insertCalculator(myTrie.getRoot(), 1); // 588
    val2 = insertCalculator(myTrie.getRoot(), 2); // 697
    val3 = insertCalculator(myTrie.getRoot(), 3); // 703
    val4 = insertCalculator(myTrie.getRoot(), 4); // 606
    myTrie.insertWord(array[4]);
    val5 = insertCalculator(myTrie.getRoot(), 5); // 592
    if (val1 == 588 && val2 == 697 && val3 == 703 && val4 == 606 && val5 == 592)
    {
        points += 10;
    }
    return points;
}
double insertCalculator(shared_ptr<Node> myNode, int val)
{
    if (val == 1)
    {
        if (myNode->children.size())
            return (myNode->children[0]->value + insertCalculator(myNode->children[0], 1));
        return 0;
    }
    if (val == 2)
    {
        if (myNode->children.size())
        {
            if (myNode->children[0]->value == 110)
                return (myNode->children[1]->value + insertCalculator(myNode->children[1], 2));
            return (myNode->children[0]->value + insertCalculator(myNode->children[0], 2));
        }
        return 0;
    }
    if (val == 3)
    {
        if (myNode->children.size())
        {
            if (myNode->value == ' ')
                return (myNode->children[1]->value + insertCalculator(myNode->children[1], 3));
            return (myNode->children[0]->value + insertCalculator(myNode->children[0], 3));
        }
        return 0;
    }
    if (val == 4)
    {
        if (myNode->children.size())
        {
            if (myNode->value == ' ')
                return (myNode->children[1]->value + insertCalculator(myNode->children[1], 4));
            if (myNode->children.size() == 2)
                return (myNode->children[1]->value + insertCalculator(myNode->children[1], 4));
            return (myNode->children[0]->value + insertCalculator(myNode->children[0], 4));
        }
        return 0;
    }
    if (val == 5)
    {
        if (myNode->children.size())
        {
            if (myNode->value == ' ')
                return (myNode->children[1]->value + insertCalculator(myNode->children[1], 5));
            return (myNode->children[0]->value + insertCalculator(myNode->children[0], 5));
        }
        return 0;
    }
    else
    return 0;
}
double searchTest(trie &myTrie, string array[])
{
    double points = 0;
    bool searchArray[20];
    bool check1 = true, check2 = true, check3 = true;
    for (int i = 5; i < 10; i++)
        myTrie.insertWord(array[i]);
    string newArray[10] = {"Ahmad", "Hamid", "Rohail", "Rameen", "Musa", "Javed", "Mati", "Noman", "Naima", "Nabeel"};
    for (int i = 0; i < 10; i++)
        myTrie.insertWord(newArray[i]);
    if (myTrie.getRoot()->children.size() != 10)
        return 0;
    for (int i = 0; i < 10; i++)
        searchArray[i] = myTrie.search(array[i]);
    for (int i = 10; i < 20; i++)
        searchArray[i] = myTrie.search(newArray[i - 10]);
    for (int i = 0; i < 20; i++)
    {
        if (!searchArray[i])
        {
            check1 = false;
            break;
        }
    }
    if (check1)
        points += 4;
    if (myTrie.search("Hasnain"))
        check2 = false;
    if (myTrie.search("Dania"))
        check3 = false;
    if (check2 && check3)
        points += 2;
    return points;
}
double subStrTest(trie &myTrie)
{
    double points = 0;
    string word1 = myTrie.longestSubstr("Mahnoor");
    string word2 = myTrie.longestSubstr("Dania");
    string word3 = myTrie.longestSubstr("Bilawal");
    string word4 = myTrie.longestSubstr("Hassaam");
    if (word1 == "Ma" && word2 == "Dani" && word3 == "Bila" && word4 == "Hassaa")
        points += 4;
    return points;
}
double vectorTest(trie &myTrie, string array[])
{
    double points = 0;
    bool check = true, check2 = true;
    vector<string> myVec = myTrie.getTrie();
    for (int i = 0; i < 20; i++)
    {
        if (myVec[i] != array[i])
        {
            cout<<array[i]<<myVec[i];
            check = false;
            break;
        }
    }
    trie newTrie;
    string myArray[7] = {"Salman", "Saman", "Sohaib", "Saleem", "Zara", "Zubair", "Zaeem"};
    string sortedArray[7] = {"Saleem", "Salman", "Saman", "Sohaib", "Zaeem", "Zara", "Zubair"};
    for (int i = 0; i < 7; i++)
        newTrie.insertWord(myArray[i]);
    myVec.clear();
    myVec = newTrie.getTrie();
    for (int i = 0; i < 7; i++)
    {
        if (myVec[i] != sortedArray[i])
        {
            check2 = false;
            break;
        }
    }
    if (check && check2)
        points += 10;
    return points;
}
double deleteCalculator(shared_ptr<Node> myNode, int val);
double deleteTest(string array[], string myArray[])
{
    double points = 0;
    bool check = true, check2 = false, check3 = true;
    trie myTrie;
    for (int i = 0; i < 10; i++)
        myTrie.insertWord(myArray[i]);
    myTrie.deleteWord("Areena");
    myTrie.deleteWord("Hassaan");
    myTrie.deleteWord("Daniyal");
    myTrie.deleteWord("Haseeb");
    vector<string> myVec = myTrie.getTrie();
    for (int i = 0; i < 6; i++)
    {
        if (myVec[i] != array[i])
        {
            // cout<<"!!"<<array[i]<<myVec[i];
            check = false;
            break;
        }
    }
    myTrie.deleteWord("Salman");
    double val1 = 0, val2 = 0, val3 = 0, val4 = 0, val5 = 0;
    val1 = deleteCalculator(myTrie.getRoot(), 1); // 576
    val2 = deleteCalculator(myTrie.getRoot(), 2); // 697
    val3 = deleteCalculator(myTrie.getRoot(), 3); // 484
    val4 = deleteCalculator(myTrie.getRoot(), 4); // 592
    val5 = deleteCalculator(myTrie.getRoot(), 5); // 606
    if (val1 == 576 && val2 == 697 && val3 == 484 && val4 == 592 && val5 == 606)
        check2 = true;
    if (check && check2)
        points += 8;
    myTrie.deleteWord("Mahnoor");
    myTrie.deleteWord("Hamza");
    myVec = myTrie.getTrie();
    for (int i = 0; i < 5; i++)
    {
        if (myVec[i] != array[i])
        {
            check3 = false;
            break;
        }
    }
    if (check3)
        points += 2;
    return points;
    return 0;
}
double deleteCalculator(shared_ptr<Node> myNode, int val)
{
    if (val == 1)
    {
        if (myNode->children.size())
            return (myNode->children[0]->value + deleteCalculator(myNode->children[0], 1));
        return 0;
    }
    if (val == 2)
    {
        if (myNode->children.size())
        {
            if (myNode->children.size() == 2)
                return (myNode->children[1]->value + deleteCalculator(myNode->children[1], 2));
            return (myNode->children[0]->value + deleteCalculator(myNode->children[0], 2));
        }
        return 0;
    }
    if (val == 3)
    {
        if (myNode->children.size())
        {
            if (myNode->value == ' ')
                return (myNode->children[1]->value + deleteCalculator(myNode->children[1], 3));
            return (myNode->children[0]->value + deleteCalculator(myNode->children[0], 3));
        }
        return 0;
    }
    if (val == 4)
    {
        if (myNode->children.size())
        {
            if (myNode->value == ' ')
                return (myNode->children[2]->value + deleteCalculator(myNode->children[2], 4));
            return (myNode->children[0]->value + deleteCalculator(myNode->children[0], 4));
        }
        return 0;
    }
    if (val == 5)
    {
        if (myNode->children.size())
        {
            if (myNode->value == ' ')
                return (myNode->children[3]->value + deleteCalculator(myNode->children[3], 5));
            return (myNode->children[0]->value + deleteCalculator(myNode->children[0], 5));
        }
        return 0;
    }
    else 
    return 0;
}
int main()
{
    trie myTrie;
    string words[10] = {"Areesha", "Areena", "Hassan", "Hassaan", "Faisal", "Daniyal", "Areeba", "Haseeb", "Salman", "Bilal"};
    string sorted[20] = {"Ahmad", "Areeba", "Areena", "Areesha", "Bilal", "Daniyal", "Faisal", "Hamid", "Haseeb", "Hassaan", "Hassan", "Javed", "Mati", "Musa", "Nabeel", "Naima", "Noman", "Rameen", "Rohail", "Salman"};
    string newArray[10] = {"Ahmad", "Hamid", "Rohail", "Rameen", "Musa", "Javed", "Mati", "Noman", "Naima", "Nabeel"};
    string leftover[6] = {"Areeba", "Areesha", "Bilal", "Faisal", "Hassan", "Salman"};
    // STARTING TEST CASES
    double totpoints = 0;

    // INSERTION TEST
    totpoints += insertionTest(myTrie, words);
    cout << "\nInsertion test: ";
    cout << totpoints << "/10 points\n"
         << endl;

    // SEARCH TEST
    double pts = searchTest(myTrie, words);
    totpoints += pts;
    cout << "Search test: ";
    cout << pts << "/6 points\n"
         << endl;

    // LONGEST SUBSTRING TEST
    pts = subStrTest(myTrie);
    totpoints += pts;
    cout << "Longest Substring test: ";
    cout << pts << "/4 points\n"
         << endl;

    // TRAVERSING TRIE TEST
    pts = vectorTest(myTrie, sorted);
    totpoints += pts;
    cout << "Traversing Trie test: ";
    cout << pts << "/10 points\n"
         << endl;

    // DELETE WORD TEST
    pts = deleteTest(leftover, words);
    totpoints += pts;
    cout << "Deletion test: ";
    cout << pts << "/10 points\n\n"
         << endl;

    cout << "TOTAL POINTS: " << totpoints << "/40 points\n\n";
}