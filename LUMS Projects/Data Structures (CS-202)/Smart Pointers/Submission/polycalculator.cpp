#ifndef __POLYCALCULATOR_CPP
#define __POLYCALCULATOR_CPP

#include "polycalculator.h"

// Constructor
template <class T>
PolyCalculator<T>::PolyCalculator() 
{
    
}

template <class T>
PolyCalculator<T>::~PolyCalculator() 
{

}

// Input Function
template <class T>
void PolyCalculator<T>::input(PolyCalculator<int>& p1)//O(n^2)
{
    string exp1, exp2;
    cout<<"Enter First Polynomial Expression: ";
    cin>>exp1;
    int power = highestExp(exp1);
    exp1 = sort(exp1);
    int temp;
    for (int i = 0; i < exp1.length(); i++)
    {
        if (exp1[i] == ',')
        {
            int pos = i - 1;
            string num = "";
            while (isdigit(exp1[pos]))
            {
                num = exp1[pos] + num;
                pos--;
                if (pos<0)
                {
                    break;
                }
            }
            temp = stoi(num);
            if (pos > 0)
            {
                if (exp1[pos] == '-')
                {
                    temp *= -1;
                }
            }
            list1.insertAtTail(temp);
            list2.insertAtTail(power);
            power--;
        }
    }

    cout<<"Enter Second Polynomial Expression: ";
    cin>>exp2;
    power = highestExp(exp2);
    exp2 = sort(exp2);

    for (int i = 0; i < exp2.length(); i++)
    {
        if (exp2[i] == ',')
        {
            int pos = i - 1;
            string num = "";
            while (isdigit(exp2[pos]))
            {
                num = exp2[pos] + num;
                pos--;
                if (pos<0)
                {
                    break;
                }
            }
            temp = stoi(num);
            if (pos > 0)
            {
                if (exp2[pos] == '-')
                {
                    temp *= -1;
                }
            }
            p1.list1.insertAtTail(temp);
            p1.list2.insertAtTail(power);
            power--;
        }
    }
}

// Display Function
template <class T>
void PolyCalculator<T>::display(PolyCalculator<int> &p1)//O(n)
{
   int len1 = list1.length();
   shared_ptr<ListItem<T>> temp1 = list1.getHead();
   cout<<"Exp1:\t";
   for (int i = 0; i < len1; i++)
   {
        if (temp1->value != 0)
        {
            cout<<temp1->value<<"x^"<<len1 - i - 1;
        }
        temp1 = temp1->next;
   }
   cout<<"\n";

   int len2 = p1.list1.length();
   shared_ptr<ListItem<T>> temp2 = p1.list1.getHead();
   cout<<"Exp2:\t";
   for (int i = 0; i < len2; i++)
   {
        if (temp2->value != 0)
        {
            cout<<temp2->value<<"x^"<<len2 - i - 1;
        }
        temp2 = temp2->next;
   }
   cout<<"\n";
}

template <class T>
void PolyCalculator<T>::add(PolyCalculator<int>& p1)//O(n^2)
{
    stored = "";
    shared_ptr<ListItem<T>> val1 = list1.getHead();
    shared_ptr<ListItem<T>> expo1 = list2.getHead();

    cout<<"Exp1 + Exp2:\t";

    for(int i = 0; i < list1.length(); i++)
    {
        shared_ptr<ListItem<T>> val2 = p1.list1.getHead();
        shared_ptr<ListItem<T>> expo2 = p1.list2.getHead();
        for (int j = 0; j < list2.length(); j++)
        {
            if (expo1->value == expo2->value)
            {
                if (val1->value + val2->value > 0 && i == 0)
                {
                    cout<<val1->value + val2->value<<"x^"<<expo1->value<<" ";
                    stored += to_string(val1->value + val2->value) + "x^" + to_string(expo1->value) + " ";
                }
                else if (val1->value + val2->value < 0)
                {
                    cout<<val1->value + val2->value<<"x^"<<expo1->value<<" ";
                    stored += to_string(val1->value + val2->value) + "x^" + to_string(expo1->value) + " ";
                }
                else if (val1->value + val2->value != 0)
                {
                    cout<<"+"<<val1->value + val2->value<<"x^"<<expo1->value<<" ";
                    stored += "+" + to_string(val1->value + val2->value) + "x^" + to_string(expo1->value) + " ";
                }
            }
            val2 = val2->next;
            expo2 = expo2->next;
        }
        val1 = val1->next;
        expo1 = expo1->next;
    }
    cout<<"\n";
}

template <class T>
void PolyCalculator<T>::sub(PolyCalculator<int>& p1)//O(n^2)
{
    stored = "";
    shared_ptr<ListItem<T>> val1 = list1.getHead();
    shared_ptr<ListItem<T>> expo1 = list2.getHead();

    cout<<"Exp1 - Exp2:\t";

    for(int i = 0; i < list1.length(); i++)
    {
        shared_ptr<ListItem<T>> val2 = p1.list1.getHead();
        shared_ptr<ListItem<T>> expo2 = p1.list2.getHead();
        for (int j = 0; j < list2.length(); j++)
        {
            if (expo1->value == expo2->value)
            {
                if (val1->value - val2->value > 0 && i == 0)
                {
                    cout<<val1->value - val2->value<<"x^"<<expo1->value;
                    stored += to_string(val1->value - val2->value) + "x^" + to_string(expo1->value);
                }
                else if (val1->value - val2->value < 0)
                {
                    cout<<val1->value - val2->value<<"x^"<<expo1->value;
                    stored += to_string(val1->value - val2->value) + "x^" + to_string(expo1->value);
                }
                else if (val1->value - val2->value != 0)
                {
                    cout<<"+"<<val1->value - val2->value<<"x^"<<expo1->value;
                    stored += "+" + to_string(val1->value - val2->value) + "x^" + to_string(expo1->value);
                }
            }
            val2 = val2->next;
            expo2 = expo2->next;
        }
        val1 = val1->next;
        expo1 = expo1->next;
    }
    cout<<"\n";
}

template <class T>
void PolyCalculator<T>::mul(PolyCalculator<int>& p1)//O(n^2)
{
    stored = "";
    shared_ptr<ListItem<T>> val1 = list1.getHead();
    shared_ptr<ListItem<T>> expo1 = list2.getHead();
    string result = "";

    for(int i = 0; i < list1.length(); i++)
    {
        shared_ptr<ListItem<T>> val2 = p1.list1.getHead();
        shared_ptr<ListItem<T>> expo2 = p1.list2.getHead();
        for (int j = 0; j < list2.length(); j++)
        {
            if (val1->value * val2->value < 0)
            {
                result += to_string(val1->value * val2->value) + "x^" + to_string(expo1->value + expo2->value);
            }
            else if (val1->value * val2->value != 0)
            {
                result += "+" + to_string(val1->value * val2->value) + "x^" + to_string(expo1->value + expo2->value);
            }
            val2 = val2->next;
            expo2 = expo2->next;
        }
        val1 = val1->next;
        expo1 = expo1->next;
    }
    cout<<"Exp1 * Exp2:\t";

    string final = sort(result);
    int power = highestExp(result);
    int temp;
    for (int i = 0; i < final.length(); i++)
    {
        if (final[i] == ',')
        {
            int pos = i - 1;
            string num = "";
            while (isdigit(final[pos]))
            {
                num = final[pos] + num;
                pos--;
                if (pos<0)
                {
                    break;
                }
            }
            temp = stoi(num);
            if (pos > 0)
            {
                if (final[pos] == '-')
                {
                    temp *= -1;
                }
            }
            if (power == highestExp(result) && temp > 0)
            {
                cout<<temp<<"x^"<<power;
                int x = temp;
                stored += to_string(temp) + "x^" + to_string(power);
            }
            else if (temp < 0)
            {
                cout<<temp<<"x^"<<power;
                stored += to_string(temp) + "x^" + to_string(power);
            }
            else if (temp > 0)
            {
                cout<<"+"<<temp<<"x^"<<power;
                stored += "+" + to_string(temp) + "x^" + to_string(power);
            }
            power--;
        }

    }
    cout<<"\n";
}

template <class T>
void PolyCalculator<T>::readData(string filename,PolyCalculator<int>& p1)//O(n^2)
{
    string line1, line2;
    ifstream inputFile("filename");
    if (inputFile.is_open())
    {
        getline(inputFile, line1);
        getline(inputFile, line2);
        inputFile.close();
    }
    else
    {
        cout << "File does not exist." << endl;
    }

    for (int i = 0; i < list1.length(); i++)
    {
        list1.deleteTail();
        list2.deleteTail();
    }
    for (int i = 0; i < p1.list2.length(); i++)
    {
        p1.list1.deleteTail();
        p1.list2.deleteTail();
    }


    int power = highestExp(line1);
    line1 = sort(line1);
    int temp;
    for (int i = 0; i < line1.length(); i++)
    {
        if (line1[i] == ',')
        {
            int pos = i - 1;
            string num = "";
            while (isdigit(line1[pos]))
            {
                num = line1[pos] + num;
                pos--;
                if (pos<0)
                {
                    break;
                }
            }
            temp = stoi(num);
            if (pos > 0)
            {
                if (line1[pos] == '-')
                {
                    temp *= -1;
                }
            }
            list1.insertAtTail(temp);
            list2.insertAtTail(power);
            power--;
        }
    }

    line2 = sort(line2);
    power = highestExp(line2);
    for (int i = 0; i < line2.length(); i++)
    {
        if (line2[i] == ',')
        {
            int pos = i - 1;
            string num = "";
            while (isdigit(line2[pos]))
            {
                num = line2[pos] + num;
                pos--;
                if (pos<0)
                {
                    break;
                }
            }
            temp = stoi(num);
            if (pos > 0)
            {
                if (line2[pos] == '-')
                {
                    temp *= -1;
                }
            }
            p1.list1.insertAtTail(temp);
            p1.list2.insertAtTail(power);
            power--;
        }
    }
}

template <class T>
void PolyCalculator<T>::writeData(string filename,PolyCalculator<int>& p1)//O(n^2)
{
    fstream my_file;
	my_file.open(filename, ios::out);
	if (!my_file) 
    {
		cout << "File not found!";
	}

    string poly1 = "", poly2 = "";

    shared_ptr<ListItem<T>> val1 = list1.getHead();
    shared_ptr<ListItem<T>> expo1 = list2.getHead();

    for(int i = 0; i < list1.length(); i++)
    {
        if (i == 0 && val1->value != 0)
        {
            poly1 += to_string(val1->value) + "x^" + to_string(expo1->value);
        }
        else if (val1->value < 0)
        {
            poly1 += to_string(val1->value) + "x^" + to_string(expo1->value);
        }
        else if (val1->value > 0)
        {
            poly1 += "+" + to_string(val1->value) + "x^" + to_string(expo1->value);
        }
        val1 = val1->next;
        expo1 = expo1->next;
    }

    shared_ptr<ListItem<T>> val2 = p1.list1.getHead();
    shared_ptr<ListItem<T>> expo2 = p1.list2.getHead();

    for(int i = 0; i < p1.list1.length(); i++)
    {
        if (i == 0 && val2->value != 0)
        {
            poly2 += to_string(val2->value) + "x^" + to_string(expo2->value);
        }
        else if (val2->value < 0)
        {
            poly2 += to_string(val2->value) + "x^" + to_string(expo2->value);
        }
        else if (val2->value > 0)
        {
            poly2 += "+" + to_string(val2->value) + "x^" + to_string(expo2->value);
        }
        val2 = val2->next;
        expo2 = expo2->next;
    }

    my_file << "Exp1:\t" << poly1 << "\n";
    my_file << "Exp2:\t" << poly2 << "\n";
    add(p1);
    my_file << "Exp1 + Exp2:\t" << stored << "\n";
    sub(p1);
    my_file << "Exp1 - Exp2:\t" << stored << "\n";
    mul(p1);
    my_file << "Exp1 * Exp2:\t" << stored << "\n";

    my_file.close();
}

template <class T>
bool PolyCalculator<T>::parse(string str, PolyCalculator<int>& p1)//O(n^2)
{
    if (highestExp(str) < 0)
    {
        return false;
    }
    for (int i = 0; i , str.length(); i++)
    {
        if (str[i] != '-' && str[i] != '+' && str[i] != 'x' && str[i] != '^' && !(isdigit(str[i])))
        {
            return false;
        }
        if ((str[i] == '^' && i < 2) || (str[i] == '^' && i == str.length() - 1))
        {
            return false;
        }
        else if(str[i] == '^' && (i >= 2 && i < str.length() - 1))
        {
            if (!(isdigit(str[i+1])) || str[i-1] != 'x' || !(isdigit(str[i-2])))
            {
                return false;
            }
        }
    }
    string sorted = sort(str);
    for (int i = 0; i < p1.list1.length(); i++)
    {
        p1.list1.deleteHead();
        p1.list2.deleteHead();
    }

    int power = highestExp(str);
    int temp;

    for (int i = 0; i < sorted.length(); i++)
    {
        if (sorted[i] == ',')
        {
            int pos = i - 1;
            string num = "";
            while (isdigit(sorted[pos]))
            {
                num = sorted[pos] + num;
                pos--;
                if (pos<0)
                {
                    break;
                }
            }
            temp = stoi(num);
            if (pos > 0)
            {
                if (sorted[pos] == '-')
                {
                    temp *= -1;
                }
            }
            p1.list1.insertAtTail(temp);
            p1.list2.insertAtTail(power);
            power--;
        }
    }

}

template <class T>
int PolyCalculator<T>::highestExp(string poly)//O(n)
{
    int l = poly.length();
    int highest = -1;
    string highest_str = "";
    for (int i = 0; i < l; i++)
    {
        if (poly[i] == '^')
        {
            int exp_len = 0, j = i + 1;
            while (isdigit(poly[j]))
            {
                exp_len++;
                j++;
            }
            highest_str = poly.substr(i+1, exp_len);
            if (stoi(highest_str) > highest)
            {
                highest = stoi(highest_str);
            }
        }
    }
    return highest;
}

template <class T>
string PolyCalculator<T>::sort(string to_be_sorted)//O(n^2)
{
    int max_exp = highestExp(to_be_sorted) + 1;
    int* all_coeff = new int[max_exp];
    for (int i = 0; i < max_exp; i++)
    {
        all_coeff[i] = 0;
    }
    for (int outer = 0; outer < max_exp; outer++)
    {
        for (int inner = 0; inner < to_be_sorted.length(); inner++)
        {
            if (to_be_sorted[inner] == '^')
            {
                int exp_len = 0, j = inner + 1;
                while (isdigit(to_be_sorted[j]))
                {
                    exp_len++;
                    j++;
                }
                int temp = stoi(to_be_sorted.substr(inner + 1, exp_len));
                if (temp == outer)
                {
                    int num_len = 0;
                    int i = inner - 2;
                    while (isdigit(to_be_sorted[i]))
                    {
                        num_len++;
                        i--;
                        if (i < 0)
                        {
                            break;
                        }
                    }
                    int temp_num = stoi(to_be_sorted.substr(inner - 1 - num_len, num_len + 1));
                    if (i > 0)
                    {
                        string sign = to_be_sorted.substr(inner - 2 - num_len, 1);
                        if (sign == "-")
                        {
                            temp_num = -1 * temp_num;
                        }
                    }
                    all_coeff[outer] += temp_num;
                }
            }
        }
    }
    string final = "";
    for (int i = max_exp - 1; i >= 0; i--)
    {
        final = final + to_string(all_coeff[i]) + ",";
    }
    return final;
}

int main() 
{
    PolyCalculator<int> p1;
    PolyCalculator<int> p2;
    p1.input(p2);
    p1.writeData("file2.txt", p2);

    return 0;
   
}


#endif
