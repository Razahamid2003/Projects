
#include "huffman.cpp"
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace std;


void testHuffmanCoding() {
    cout << "Running Huffman Coding Test...\n";
    int score = 0; 

    // Test input texts
    string inputText1 = "asadddasdssdasddsa";
    string inputText2 = "aaabbbbcccccddddddeeeeeeee";
    string inputText3 = "qwwwwweeeeeeerrrrrrrrrtttttttttttyyyyyyyyyyyyyyuuuuuuuuuuuuuuu";
    string inputText4 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabccdddd";
    string inputText5 = "bb ccccccddddeeeee";
    


    // Test case 1
    unordered_map<char, string> code1 = HuffmanTree(inputText1);
    assert(code1['a'] == "10");
    assert(code1['s'] == "11");
    assert(code1['d'] == "0");
    score+=6; 
    cout << "Test case 1 passed. Score: "<<score<<"/30 \n";

    // Test case 2
    unordered_map<char, string> code2 = HuffmanTree(inputText2);
    assert(code2['a'] == "100");
    assert(code2['b'] == "101");
    assert(code2['c'] == "00");
    assert(code2['d'] == "01");
    assert(code2['e'] == "11");

    score+=6; 
    cout << "Test case 2 passed. Score: "<<score<<"/30 \n";

    // Test case 3
    unordered_map<char, string> code3 = HuffmanTree(inputText3);
    assert(code3['q'] == "0000");
    assert(code3['w'] == "0001");
    assert(code3['e'] == "001");
    assert(code3['r'] == "110");
    assert(code3['t'] == "111");
    assert(code3['y'] == "01");
    assert(code3['u'] == "10");
  
    score+=6; 
    cout << "Test case 3 passed. Score: "<<score<<"/30 \n";

    // Test case 4
    unordered_map<char, string> code4 = HuffmanTree(inputText4);
   
    assert(code4['a'] == "1");
    assert(code4['b'] == "000");
    assert(code4['c'] == "001");
    assert(code4['d'] == "01");
    
    score+=6; 
    cout << "Test case 4 passed. Score: "<<score<<"/30 \n";

    // Test case 5
    unordered_map<char, string> code5 = HuffmanTree(inputText5);
    
    assert(code5['b'] == "001");
    assert(code5['c'] == "11");
    assert(code5['d'] == "01");
    assert(code5['e'] == "10");
    assert(code5[' '] == "000");
   
    score+=6; 
    cout << "Test case 5 passed. Score: "<<score<<"/30 \n";


    cout << "All test cases passed. Score: "<<score<<"/30 \n";
    }

int main() {
    testHuffmanCoding();
    return 0;
}
