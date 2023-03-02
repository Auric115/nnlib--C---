#include "nnlib/nnlib.h"

int main()
{
    NeuralNetwork MyNet(2, 0.7);
    MyNet.AddLayer(4);
    MyNet.AddLayer(1);
    MyNet.Display();
    double ins[MAXNEURONS] = {1.0, 1.0};
    MyNet.Test(ins);
    MyNet.DisplayResults();
    return 0;
}