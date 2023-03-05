#include "nnlib.h"

void NeuralNetwork::AddLayer(unsigned nodes)
{
    topology_m_uA[numLayers_m_u] = nodes;
    layers_m_cpA[numLayers_m_u - 1]->CreateMesh(nodes);
    layers_m_cpA[numLayers_m_u] = new Layer(nodes, learningRate_m_d);
    layers_m_cpA[numLayers_m_u]->prevLayer_pC = layers_m_cpA[numLayers_m_u - 1];
    numLayers_m_u++;
}

void NeuralNetwork::Display()
{
    std::cout << "------------ Neural Network ------------" << std::fixed << std::setprecision(4) << std::endl;
    std::cout << std::setw(12) << "Topology: ";
    for (unsigned l = 0; l < numLayers_m_u; l++)
    {
        std::cout << topology_m_uA[l] << std::setw(MAXNODES / 10 + 3);
    }
    std::cout << std::endl;
    std::cout << std::setw(12) << "Learning: " << learningRate_m_d << std::setw(MAXNODES / 10 + 8) << std::endl;

    std::cout << std::setw(12) << "Results: ";
    for (unsigned o = 0; o < topology_m_uA[numLayers_m_u - 1]; o++)
    {
        std::cout << results_m_dA[o] << std::setw(MAXNODES / 10 + 8);
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for (unsigned l = 0; l < numLayers_m_u; l++)
    {
        std::cout << "  ------------- Layer: " << l << " -------------  " << std::endl;
        layers_m_cpA[l]->Display();
        std::cout << std::endl;
    }
    std::cout << std::setw(12) << "Outputs: ";
    for (unsigned o = 0; o < topology_m_uA[numLayers_m_u - 1]; o++)
    {
        std::cout << outputs_m_dA[o] << ' ';
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

NeuralNetwork::NeuralNetwork(unsigned inodes, double lr)
{
    topology_m_uA[numLayers_m_u] = inodes;
    layers_m_cpA[numLayers_m_u++] = new Layer(inodes, lr);
    learningRate_m_d = lr;
}