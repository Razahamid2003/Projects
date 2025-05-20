#include <iostream>
#include <memory>
#include "../../include/avl.h"

// This function calculates the maximum applicants that u can shortlist
template <class T, class S, class C>
int AVL<T, S, C>::number_to_shortlist(shared_ptr<node<T, S, C>> root)
{
    if (root == NULL)
    {
        return 0;
    }
    else
    {
        return 1 + number_to_shortlist(root->right);
    }
}

// This function returns shortlisted candidates on the right most path to leaf
template <class T, class S, class C>
vector<T> AVL<T, S, C>::right_most(shared_ptr<node<T, S, C>> root)
{
    vector<T> vec;
    while (root != NULL)
    {
        vec.push_back(root->workExperience);
        root = root->right;
    }
    return vec;
}

// This function returns shortlisted candidates in-order
template <class T, class S, class C>
vector<T> AVL<T, S, C>::in_order(shared_ptr<node<T, S, C>> root)
{
    vector<T> result;
    int num = number_to_shortlist(root);
    in_order_helper(root, result, num);
    vector<T> final;
    for (int i = 0; i < num; i++)
    {
        final.push_back(result[i]);
    }
    return final;
}

template <class T, class S, class C>
void in_order_helper(shared_ptr<node<T, S, C>> node, vector<T> &result, int num)
{
    if (node)
    {
        in_order_helper(node->left, result, num);
        result.push_back(node->workExperience); // Assuming workExperience is the key
        in_order_helper(node->right, result, num);
    }
}

// This function returns shortlisted candidates in level order
template <class T, class S, class C>
vector<T> AVL<T, S, C>::level_order(shared_ptr<node<T, S, C>> root)
{
    vector<T> result;
    if (root == nullptr)
    {
        return result;
    }

    vector<shared_ptr<node<T, S, C>>> q;
    q.push_back(root);

    while (!q.empty())
    {
        shared_ptr<node<T, S, C>> current = q.front();
        q.erase(q.begin());

        result.push_back(current->workExperience);

        if (current->left != nullptr)
        {
            q.push_back(current->left);
        }

        if (current->right != nullptr)
        {
            q.push_back(current->right);
        }
    }

    vector<T> final;
    int num = number_to_shortlist(root);
    for (int i = 0; i < num; i++)
    {
        final.push_back(result[i]);
    }

    return final;
}

// This function calculates the bias in the tree
template <class T, class S, class C>
vector<float> AVL<T, S, C>::bias(shared_ptr<node<T, S, C>> root)
{
    vector<float> ratios;

    if (root == nullptr)
    {
        return ratios;
    }

    vector<T> order = level_order(root);

    return biasHelper(order, ratios);
}

template <class T, class S, class C>
vector<float> AVL<T, S, C>::biasHelper(vector<T> order, vector<float> ratios)
{
    for (int i = 0; i < order.size(); i++)
    {
        float males = 0;
        float females = 0;
        shared_ptr<node<T, S, C>> current_node = searchNode(order[i]);
        if (current_node->left != NULL)
        {
            vector<string> genders;
            leftCalculator(current_node->left, genders);
            for (int i = 0; i < genders.size(); i++)
            {
                if (genders[i] == "M")
                {
                    males++;
                }
                else
                {
                    females++;
                }
            }

            if (females != 0)
            {
                ratios.push_back(females / (males + females));
            }
            if (females == 0)
            {
                ratios.push_back(0);
            }
        }
    }
    return ratios;
}

template <class T, class S, class C>
void AVL<T, S, C>::leftCalculator(shared_ptr<node<T, S, C>> node, vector<string> &genders)
{
    if (node)
    {
        leftCalculator(node->left, male, female);
        genders.push_back(node->gender);
        leftCalculator(node->right, male, female);
    }
}
