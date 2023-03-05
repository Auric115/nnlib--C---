#include "nnlib.h"

NeuralNetwork::NeuralNetwork(unsigned inodes, double lr)
{
    topology_m_uA[numLayers_m_u] = inodes;
    layers_m_cpA[numLayers_m_u++] = new Layer(inodes, lr);
    learningRate_m_d = lr;
}