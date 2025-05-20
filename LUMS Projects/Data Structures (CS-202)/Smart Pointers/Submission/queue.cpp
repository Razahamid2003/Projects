#ifndef __QUEUE_CPP
#define __QUEUE_CPP
#include "queue.h"

template <class T>
Queue<T>::Queue()//O(3)
{
    LinkedList<T> list;
}
    
template <class T>
Queue<T>::Queue(Queue<T>& otherQueue)//O(16)
{
    // LinkedList<T>(otherQueue.list);
    shared_ptr<ListItem<T>> temp = otherQueue.list.getHead();
    while(temp != NULL)
    {
        enqueue(temp->value);
        temp = temp -> next;
    }
}

template <class T>
Queue<T>::~Queue()
{
    
}

template <class T>
void Queue<T>::enqueue(T item)//O(11)
{
    list.insertAtTail(item);
}

template <class T>
T Queue<T>::front()//O(13)
{
    shared_ptr<ListItem<T>> temp = list.getHead();
    // T item = temp.value;
    return temp->value;
}

template <class T>
T Queue<T>::dequeue()//O(16)
{
    shared_ptr<ListItem<T>> temp = list.getHead();
    list.deleteHead();
    return temp->value;
}

template <class T>
int Queue<T>::length()//O(n)
{
    return list.length();
}

template <class T>
bool Queue<T>::isEmpty()//O(n)
{
    int l = list.length();
    if(l == 0)
    {
        return true;
    }
    return false;
}

#endif
