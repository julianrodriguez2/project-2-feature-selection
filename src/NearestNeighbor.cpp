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

using namespace std;

vector<int> labels;
vector<vector<double>> allFeatures;

bool loadData()
{
    // I manually changed it to small-test-dataset for testing
    string fileName = "large-test-dataset.txt";
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

double leaveOneOutValidation(const set<int> &featureSubset, Classifier &nn, clock_t globalStart)
{
    int labelLen = labels.size();
    cout << "Starting Leave-One-Out Validation on " << labelLen << " instances." << endl;
    int correct = 0;

    for (int i = 0; i < labelLen; i++)
    {
        int predicted = nn.Test(allFeatures[i], featureSubset, i);
        if (predicted == labels[i])
            correct++;

        double elapsed = double(clock() - globalStart) / CLOCKS_PER_SEC;
        cout << "Instance " << i << ": Predicted=" << predicted
             << ", Actual=" << labels[i]
             << ", Elapsed Time=" << elapsed << "s" << endl;
    }

    return (double)correct / labelLen;
}

int main()
{
    clock_t startTime = clock();

    cout << "Loading data." << endl;
    if (!loadData())
    {
        return 1;
    }
    double afterLoad = double(clock() - startTime) / CLOCKS_PER_SEC;
    cout << "Data loading completed in " << afterLoad << " seconds." << endl;

    cout << "Normalizing data." << endl;
    normalizeData();
    double afterNormalize = double(clock() - startTime) / CLOCKS_PER_SEC;
    cout << "Normalization completed in " << afterNormalize << " seconds." << endl;

    set<int> featureSubset = {1, 15, 27};
    cout << "Using feature subset {1,15,27} on large-test-dataset." << endl;

    Classifier nn;
    nn.Train(labels, allFeatures);

    cout << "Beginning evaluation." << endl;
    double accuracy = leaveOneOutValidation(featureSubset, nn, startTime);
    double endTime = double(clock() - startTime) / CLOCKS_PER_SEC;

    cout << "Using feature subset {1,15,27}, expected accuracy ~95%." << endl;
    cout << "Calculated Accuracy: " << accuracy * 100 << "%" << endl;
    cout << "Time taken: " << endTime << " seconds" << endl;

    return 0;
}