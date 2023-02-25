#include "nnlib.h"

void Layer::Display()
{
    std::cout << "\tNodes: " << numNeurons << std::endl;
    std::cout << std::endl;
}

Layer::Layer(unsigned nodes)
{
    numNeurons = nodes;
    for (unsigned n = 0; n < numNeurons; n++)
    {
        nodes_m_cpA[n] = new Neuron();
    }

    if (DEBUG)
    {
        assert(nodes <= MAXNEURONS);
    }
}