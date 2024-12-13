#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <ctime>
#include <iomanip>

using namespace std;

vector<int> labels;
vector<vector<double>> allFeatures;

bool loadData(const string &fileName)
{
    // I manually changed it to small-test-dataset for testing
    ifstream dataFile(fileName);
    if (!dataFile.is_open())
    {
        cout << "Error opening file: " << fileName << endl;
        return false;
    }
    string line = "";
    while (getline(dataFile, line))
    {
        if (line.empty())
            continue;
        istringstream lineStream(line);
        double label;
        lineStream >> label;
        labels.push_back(label);

        vector<double> feats;
        double value;
        while (lineStream >> value)
        {
            feats.push_back(value);
        }
        allFeatures.push_back(feats);
    }
    cout << "There are " << labels.size() << " instances" << endl;
    return true;
}

void normalizeData()
{
    if (labels.empty())
        return;
    int numFeatures = allFeatures[0].size();
    int labelSize = labels.size();
    vector<double> meanVals(numFeatures, 0.0);
    vector<double> stdVals(numFeatures, 0.0);

    for (int i = 0; i < labelSize; i++)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            meanVals[j] += allFeatures[i][j];
        }
    }
    for (int i = 0; i < numFeatures; i++)
    {
        meanVals[i] /= labelSize;
    }

    for (int i = 0; i < labelSize; i++)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            double diff = allFeatures[i][j] - meanVals[j];
            stdVals[j] += diff * diff;
        }
    }
    for (int i = 0; i < numFeatures; i++)
    {
        stdVals[i] = sqrt(stdVals[i] / labelSize);
        if (stdVals[i] == 0.0)
            stdVals[i] = 1.0;
    }

    for (int i = 0; i < labelSize; i++)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            allFeatures[i][j] = (allFeatures[i][j] - meanVals[j]) / stdVals[j];
        }
    }
}

double euclideanDistance(const vector<double> &testFeatures, const vector<double> &trainFeatures, const set<int> &featureSubset)
{
    double dist = 0.0;
    for (auto feature : featureSubset)
    {
        int index = feature - 1;
        double diff = testFeatures[index] - trainFeatures[index];
        dist += diff * diff;
    }
    return sqrt(dist);
}

class Classifier
{
private:
    vector<int> trainLabels;
    vector<vector<double>> trainFeatures;

public:
    void Train(const vector<int> &tl, const vector<vector<double>> &tf)
    {
        trainLabels = tl;
        trainFeatures = tf;
    }

    int Test(const vector<double> &testFeatures, const set<int> &featureSubset, int testIndex)
    {
        double minDistance = numeric_limits<double>::max();
        int predictedLabel = -1;
        int tlLen = trainLabels.size();

        for (int i = 0; i < tlLen; i++)
        {
            if (i == testIndex)
                continue;

            double dist = euclideanDistance(testFeatures, trainFeatures[i], featureSubset);

            if (dist < minDistance)
            {
                minDistance = dist;
                predictedLabel = trainLabels[i];
            }
        }

        return predictedLabel;
    }
};

class Validator
{
public:
    double LeaveOneOutEvaluation(const set<int> &featureSubset, Classifier &nn, clock_t start)
    {
        int labelLen = labels.size();
        int correct = 0;

        for (int i = 0; i < labelLen; i++)
        {
            int predicted = nn.Test(allFeatures[i], featureSubset, i);
            if (predicted == labels[i])
                correct++;

            double elapsed = double(clock() - start) / CLOCKS_PER_SEC;
            // cout << "Instance " << i << ": Predicted=" << predicted
            //      << ", Actual=" << labels[i]
            //      << ", Elapsed Time=" << elapsed << "s" << endl;
        }

        return (double)correct / labelLen;
    }
};

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

void forwardSelection(int numFeat, Validator &validator, Classifier &nn)
{
    set<int> curr;
    cout << "Using no features and \"random\" evaluation, I get an accuracy of "
         << fixed << setprecision(2) << validator.LeaveOneOutEvaluation(curr, nn, clock()) * 100 << "%\n\n";
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
                double acc = validator.LeaveOneOutEvaluation(possibleSet, nn, clock());
                cout << "Using feature(s) ";
                printSet(possibleSet);
                cout << " accuracy is " << fixed << setprecision(2) << acc << "%\n";

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
            cout << " was best, accuracy is " << fixed << setprecision(2) << bestAcc << "%\n \n";

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
    cout << " with an accuracy of " << fixed << setprecision(2) << bestMaxAcc << "%\n";
}

void backwardElimination(int numFeat, Validator &validator, Classifier &nn)
{
    set<int> curr;
    for (int i = 1; i <= numFeat; i++)
    {
        curr.insert(i);
    }

    double bestMaxAcc = validator.LeaveOneOutEvaluation(curr, nn, clock());
    cout << "Using all features and \"random\" evaluation, I get an accuracy of "
         << fixed << setprecision(2) << bestMaxAcc << "%\n \n";
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
            double acc = validator.LeaveOneOutEvaluation(possibleSet, nn, clock());
            cout << "Using feature(s) ";
            printSet(possibleSet);
            cout << " accuracy is " << fixed << setprecision(2) << (acc * 100) << "%\n";

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
            cout << " was best, accuracy is " << fixed << setprecision(2) << (bestMaxAcc * 100) << "%\n \n";

            if (bestAcc > bestMaxAcc)
            {
                bestMaxAcc = bestAcc;
                bestFeatSet = curr;
            }
        }
    }

    cout << "Finished search! The best feature subset is ";
    printSet(bestFeatSet);
    cout << " with an accuracy of " << fixed << setprecision(2) << bestMaxAcc << "%\n";
}

int main()
{
    srand(time(0));
    clock_t startTime = clock();

    cout << "Please enter the name of the dataset file: ";
    string fileName;
    cin >> fileName;

    cout << "Loading data." << endl;

    if (!loadData(fileName))
    {
        return 1;
    }

    cout << "Normalizing data." << endl;
    normalizeData();
    double afterNormalize = double(clock() - startTime) / CLOCKS_PER_SEC;
    cout << "Normalization completed in " << afterNormalize << " seconds." << endl;

    Classifier nn;
    nn.Train(labels, allFeatures);

    Validator validator;

    cout << "Welcome to Julian Rodriguez Feature Selection Algorithm.\n";
    int numFeat = allFeatures[0].size();

    cout << "Type the number of the algorithm you want to run.\n";
    cout << "1) Forward Selection\n";
    cout << "2) Backward Elimination\n";
    int c;
    cin >> c;

    if (c == 1)
    {
        forwardSelection(numFeat, validator, nn);
    }
    else if (c == 2)
    {
        backwardElimination(numFeat, validator, nn);
    }
    else
    {
        cout << "Invalid choice. Exiting program.\n";
    }

    return 0;
}
