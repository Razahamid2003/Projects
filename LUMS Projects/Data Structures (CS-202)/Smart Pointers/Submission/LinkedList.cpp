#ifndef __LIST_CPP
#define __LIST_CPP

#include <cstdlib>
#include <memory>
#include <iostream>
#include "LinkedList.h"
using namespace std;

template <class T>
LinkedList<T>::LinkedList()//O(3)
{
    head = make_shared<ListItem<T>>(T());
    tail = head;
}

template <class T>
LinkedList<T>::LinkedList(const LinkedList<T>& otherLinkedList) //O(n)
{
    head = make_shared<ListItem<T>>(T());
    tail = head;
    shared_ptr<ListItem<T>> current = otherLinkedList.head->next;
    shared_ptr<ListItem<T>> dummy = head;
    while (current)
    {
        shared_ptr<ListItem<T>> newItem = make_shared<ListItem<T>>(current->value);
        dummy->next = newItem;
        newItem->prev = dummy;
        dummy = newItem;
        current = current->next;
    }
    tail->prev = dummy;
}

template <class T>
void LinkedList<T>::insertAtHead(T item)//O(11)
{
    shared_ptr<ListItem<T>> newItem = make_shared<ListItem<T>>(item);
    if (head->next == NULL)
    {
        head->next = newItem;
        tail->prev = newItem;
    }
    else
    {
        shared_ptr<ListItem<T>> temp = head->next;
        temp->prev = newItem;
        newItem->next = temp;
        head->next = newItem;
    }
}

template <class T>
void LinkedList<T>::insertAtTail(T item)//O(11)
{
    shared_ptr<ListItem<T>> newItem = make_shared<ListItem<T>>(item);
    if (tail->prev == NULL)
    {
        head->next = newItem;
        tail->prev = newItem;
    }
    else
    {
        shared_ptr<ListItem<T>> temp = tail->prev;
        temp->next = newItem;
        newItem->prev = temp;
        tail->prev = newItem;
    }
}

template <class T>
void LinkedList<T>::insertAfter(T toInsert, T afterWhat)//O(18)
{
    shared_ptr<ListItem<T>> temp = head->next;
    shared_ptr<ListItem<T>> newItem = make_shared<ListItem<T>>(toInsert);
    while (temp->value != afterWhat)
    {
        temp = temp->next;
    }

    if (temp->next != NULL)
    {
        shared_ptr<ListItem<T>> nxt_ptr = temp->next;
        temp->next = newItem;
        newItem->next = nxt_ptr;
        nxt_ptr->prev = newItem;
        newItem->prev = temp;
    }
    else
    {
        temp->next = newItem;
        newItem->prev = temp;
    }
}

template <class T>
shared_ptr<ListItem<T>> LinkedList<T>::getHead()//O(3)
{
    shared_ptr<ListItem<T>> temp = head->next;
    return temp;
}

template <class T> 
shared_ptr<ListItem<T>> LinkedList<T>::getTail()//O(3)
{
    shared_ptr<ListItem<T>> temp = tail->prev;
    return temp;
}

template <class T>
shared_ptr<ListItem<T>> LinkedList<T>::searchFor(T item)//O(n)
{
    shared_ptr<ListItem<T>> temp = head;
    for (int i = 0; i < length(); i++)
    {
        temp = temp->next;
        if (temp->value == item)
        {
            return temp;
        }
    }
}

template <class T>
void LinkedList<T>::deleteElement(T item)//O(n)
{
    shared_ptr<ListItem<T>> temp = head;
    for (int i = 0; i < length(); i++)
    {
        temp = temp->next;
        if (temp->value == item)
        {
            if (temp->prev == NULL)
            {
                deleteHead();
            }
            else if (temp->next == NULL)
            {
                deleteTail();
            }
            else
            {
                shared_ptr<ListItem<T>> nxt = temp->next;
                shared_ptr<ListItem<T>> prv = temp->prev;
                prv->next = nxt;
                nxt->prev = prv;
            }
        }
    }
}

template <class T>
void LinkedList<T>::deleteHead()//O(10)
{
    shared_ptr<ListItem<T>> temp = head->next;
    if (temp->next == NULL)
    {
        head->next = NULL;
        tail->prev = NULL;
    }
    else
    {
        shared_ptr<ListItem<T>> nxt = temp->next;
        head->next = nxt;
        nxt->prev = NULL;
    }
}

template <class T>
void LinkedList<T>::deleteTail()//O(10)
{
    shared_ptr<ListItem<T>> temp = tail->prev;
    if (temp->prev == NULL)
    {
        head->next = NULL;
        tail->prev = NULL;
    }
    else
    {
        shared_ptr<ListItem<T>> prv = temp->prev;
        tail->prev = prv;
        prv->next = NULL;
    }
}

template <class T>
int LinkedList<T>::length()//O(n)
{
    shared_ptr<ListItem<T>> current = head;
    int l = 0;
    while (current->next != NULL)
    {
        l++;
        current = current->next;
    }
    return l;
}
#endif
