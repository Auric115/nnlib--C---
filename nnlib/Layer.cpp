#include "nnlib.h"

void Layer::SetIntakes(double in[MAXNEURONS])
{
    for (unsigned i = 0; i < numNeurons_u; i++)
    {
        intakes_d[i] = in[i];
    }
}

void Layer::Display()
{
    std::cout << "\tNodes: " << numNeurons_u << std::endl;
    std::cout << std::endl;
}

Layer::Layer(unsigned nodes)
{
    numNeurons_u = nodes;

    if (DEBUG)
    {
        assert(nodes <= MAXNEURONS);
    }
}