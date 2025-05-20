#ifndef __QUEUE_H
#define __QUEUE_H
#include "LinkedList.cpp"

/* This is the generic Queue class */
template <class T>
class Queue
{
    LinkedList<T> list;

  public:

    // Constructor
    Queue();//O(3)

    // Copy Constructor
    Queue(Queue<T>& otherQueue);//O(16)

    // Destructor
    ~Queue();

    // Required Methods
    void enqueue(T item);//O(11)
    T front();//O(13)
    T dequeue();//O(16)
    int length();//O(n)
    bool isEmpty();//O(n)
};

#endif
