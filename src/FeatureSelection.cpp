#include <iostream>
#include <vector>
#include <set>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

double evalStub(const set<int> &features)
{
    return static_cast<double>(rand() % 10000) / 100;
}

void printSet(const set<int> &feats)
{
    int count = 0;
    cout << "{";
    for (int feat : feats)
    {
        if (count > 0)
        {
            cout << ",";
        }
        cout << feat;
        count++;
    }
    cout << "}";
}

void forwardSelection(int numFeat)
{
    set<int> curr;
    cout << "Using no features and \"random\" evaluation, I get an accuracy of "
         << fixed << setprecision(1) << evalStub(curr) << "%\n"
         << "\n";
    cout << "Beginning search.\n \n";

    double bestMaxAcc = 0.0;
    set<int> bestFeatSet;

    for (int i = 0; i < numFeat; i++)
    {
        int bestFeat = -1;
        double bestAcc = 0.0;

        for (int j = 1; j <= numFeat; j++)
        {
            if (curr.find(j) == curr.end())
            {
                set<int> possibleSet = curr;
                possibleSet.insert(j);
                double acc = evalStub(possibleSet);
                cout << "Using feature(s) ";
                printSet(possibleSet);
                cout << " accuracy is " << fixed << setprecision(1) << acc << "%\n";

                if (acc > bestAcc)
                {
                    bestAcc = acc;
                    bestFeat = j;
                }
            }
        }

        if (bestFeat != -1)
        {
            curr.insert(bestFeat);
            cout << "\n Feature set ";
            printSet(curr);
            cout << " was best, accuracy is " << fixed << setprecision(1) << bestAcc << "%\n \n";

            if (bestAcc > bestMaxAcc)
            {
                bestMaxAcc = bestAcc;
                bestFeatSet = curr;
            }
            else
            {
                cout << "(Warning, Accuracy has decreased!)\n \n";
            }
        }
    }

    cout << "Finished search! The best feature subset is ";
    printSet(bestFeatSet);
    cout << " with an accuracy of " << fixed << setprecision(1) << bestMaxAcc << "%\n";
}

void backwardElimination(int numFeat)
{
    set<int> curr;
    for (int i = 1; i <= numFeat; i++)
    {
        curr.insert(i);
    }

    double bestMaxAcc = evalStub(curr);
    cout << "Using all features and \"random\" evaluation, I get an accuracy of "
         << fixed << setprecision(1) << bestMaxAcc << "%\n \n";
    cout << "Beginning search.\n \n";

    set<int> bestFeatSet = curr;

    for (int i = numFeat; i > 0; i--)
    {
        int bestFeat = -1;
        double bestAcc = 0.0;

        for (int feat : curr)
        {
            set<int> possibleSet = curr;
            possibleSet.erase(feat);
            double acc = evalStub(possibleSet);
            cout << "Using feature(s) ";
            printSet(possibleSet);
            cout << " accuracy is " << fixed << setprecision(1) << acc << "%\n";

            if (acc > bestAcc)
            {
                bestAcc = acc;
                bestFeat = feat;
            }
        }

        if (bestFeat != -1)
        {
            curr.erase(bestFeat);
            cout << "\n Feature set ";
            printSet(curr);
            cout << " was best, accuracy is " << fixed << setprecision(1) << bestAcc << "%\n \n";

            if (bestAcc > bestMaxAcc)
            {
                bestMaxAcc = bestAcc;
                bestFeatSet = curr;
            }
        }
    }

    cout << "Finished search! The best feature subset is ";
    printSet(bestFeatSet);
    cout << " with an accuracy of " << fixed << setprecision(1) << bestMaxAcc << "%\n";
}

int main()
{
    srand(time(0));
    cout << "Welcome to Julian Rodriguez Feature Selection Algorithm.\n";
    cout << "Please enter total number of features: ";
    int numFeat;
    cin >> numFeat;

    cout << "Type the number of the algorithm you want to run.\n";
    cout << "1) Forward Selection\n";
    cout << "2) Backward Elimination\n";
    int c;
    cin >> c;

    if (c == 1)
    {
        forwardSelection(numFeat);
    }
    else if (c == 2)
    {
        backwardElimination(numFeat);
    }
    else
    {
        cout << "Invalid choice. Exiting program.\n";
    }

    return 0;
}
