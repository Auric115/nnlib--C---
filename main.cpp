#include "nnlib/nnlib.h"

int main()
{
    NeuralNetwork myNet(2, 0.7);
    myNet.AddLayer(4);
    myNet.AddLayer(1);
    myNet.Display();

    return 0;
}