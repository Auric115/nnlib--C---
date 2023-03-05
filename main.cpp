#include "nnlib/nnlib.h"
#include <ctime>

const int numTrainingSets = 4;

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main()
{
    srand(time(0));

    NeuralNetwork myNet(2, 0.7);
    myNet.AddLayer(4);
    myNet.AddLayer(1);
    myNet.Display();

    double training_inputs[numTrainingSets][MAXNODES] = {{0.0, 0.0},
                                                         {1.0, 0.0},
                                                         {0.0, 1.0},
                                                         {1.0, 1.0}};
    double training_outputs[numTrainingSets][MAXNODES] = {{0.0},
                                                          {1.0},
                                                          {1.0},
                                                          {0.0}};

    int trainingSetOrder[] = {0, 1, 2, 3};

    int numberOfEpochs = 5000;

    for (int epochs = 0; epochs < numberOfEpochs; epochs++)
    {
        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++)
        {
            int i = trainingSetOrder[x];
            myNet.Train(training_inputs[i], training_outputs[i]);
        }

        if (epochs % 1000 == 0)
        {
            myNet.Display();
        }
    }
    myNet.Display();
    return 0;
}