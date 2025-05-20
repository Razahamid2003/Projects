#ifndef __LIST_H
#define __LIST_H
#include <cstdlib>
using namespace std;

/* This class just holds a single data item. */
template <class T>
struct ListItem
{
    T value;
    shared_ptr<ListItem<T>> next;
    shared_ptr<ListItem<T>> prev;

    ListItem(T theVal)//O(6)
    {
        this->value = theVal;
        this->next = NULL;
        this->prev = NULL;
    }
};

/* This is the generic List class */
template <class T>
class LinkedList
{
    shared_ptr<ListItem<T>> head;
    shared_ptr<ListItem<T>> tail;

public:

    // Constructor
    LinkedList();//O(3)

    // Copy Constructor
    LinkedList(const LinkedList<T>& otherList);//O(n)

    // Insertion Functions
    void insertAtHead(T item);//O(11)
    void insertAtTail(T item);//O(11)
    void insertAfter(T toInsert, T afterWhat);//O(18)

    // Lookup Functions
    shared_ptr<ListItem<T>> getHead();//O(3)
    shared_ptr<ListItem<T>> getTail();//O(3)
    shared_ptr<ListItem<T>> searchFor(T item);//O(n)

    // Deletion Functions
    void deleteElement(T item);//O(n)
    void deleteHead();//O(10)
    void deleteTail();//O(10)

    // Utility Functions
    int length();//O(n)
};

#endif
