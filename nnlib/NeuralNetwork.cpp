#include "nnlib.h"

double squish(double x) { return std::tanh(x); }
double dSquish(double x) { return (1.0 - (x * x)); }

double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

void NeuralNetwork::Display()
{
    std::cout << "----- Neural Network -----" << std::fixed << std::setprecision(4) << std::endl;
    std::cout << "Topology: ";
    for (unsigned l = 0; l < numLayers_m_u; l++)
    {
        std::cout << std::setw(MAXNEURONS / 10 + 3) << topology_m_uA[l];
    }
    std::cout << "\tLearning Rate: " << learningRate_m_d << std::endl;
    for (unsigned l = 0; l < numLayers_m_u; l++)
    {
        std::cout << "\t----- Layer: " << l << " -----" << std::endl;
        layers_m_cpA[l]->Display();

        if (l != numLayers_m_u - 1)
        {
            std::cout << "\t----- Mesh: " << l << " -----" << std::endl;
            mesh_m_cpA[l]->Display();
        }
    }
}

void NeuralNetwork::AddLayer(unsigned nodes)
{
    Layer *layer = new Layer(nodes);
    Mesh *mesh = new Mesh(layers_m_cpA[--numLayers_m_u], layer);
    mesh_m_cpA[numLayers_m_u++] = mesh;
    layers_m_cpA[numLayers_m_u] = layer;
    topology_m_uA[numLayers_m_u++] = nodes;
}

NeuralNetwork::NeuralNetwork(unsigned nodes, double learningRate)
{
    topology_m_uA[numLayers_m_u] = nodes;
    Layer *layer = new Layer(nodes);
    layers_m_cpA[numLayers_m_u++] = layer;
    learningRate_m_d = learningRate;
}