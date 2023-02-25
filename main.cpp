#include "nnlib/nnlib.h"

int main()
{
    NeuralNetwork MyNet(2, 0.7);
    MyNet.Display();
    MyNet.AddLayer(4);
    MyNet.AddLayer(1);
    MyNet.Display();
    return 0;
}