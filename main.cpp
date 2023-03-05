#include "nnlib/nnlib.h"
#include <ctime>

int main()
{
    srand(time(0));

    NeuralNetwork myNet(2, 0.7);
    myNet.AddLayer(4);
    myNet.AddLayer(1);
    myNet.Display();

    NeuralNetwork *newNet = readNetFile("nets/net5.txt");
    newNet->Display();
    return 0;
}