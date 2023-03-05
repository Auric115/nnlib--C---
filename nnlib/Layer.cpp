#include "nnlib.h"

double squish(double x) { return std::tanh(x); }
double dSquish(double x) { return (1.0 - (x * x)); }

double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

void Layer::CreateMesh(unsigned outNodes)
{
    outNodes_m_u = outNodes;
    for (unsigned oc = 0; oc < outNodes_m_u; oc++)
    {
        biases_m_dA[oc] = init_weight();
        for (unsigned ic = 0; ic < numNodes_m_u; ic++)
        {
            weights_m_d2A[ic][oc] = init_weight();
        }
    }
}

void Layer::Display()
{
    std::cout << std::setw(12) << "Nodes: " << numNodes_m_u << std::endl;

    std::cout << std::setw(12) << "Intakes: ";
    for (unsigned n = 0; n < numNodes_m_u; n++)
    {
        std::cout << intakes_dA[n] << ' ';
    }
    std::cout << std::endl;

    if (outNodes_m_u > 0)
    {
        std::cout << std::setw(12) << "Gradient: ";
        for (unsigned n = 0; n < outNodes_m_u; n++)
        {
            std::cout << gradients_dA[n] << ' ';
        }
        std::cout << std::endl;

        std::cout << std::endl;
        for (unsigned ic = 0; ic < numNodes_m_u; ic++)
        {
            for (unsigned oc = 0; oc < outNodes_m_u; oc++)
            {
                std::cout << "\tW" << oc << ": " << weights_m_d2A[ic][oc] << std::endl;
            }
            std::cout << std::endl;
        }
    }
    for (unsigned oc = 0; oc < outNodes_m_u; oc++)
    {
        std::cout << "\tB" << oc << ": " << biases_m_dA[oc] << std::endl;
    }
}

Layer::Layer(unsigned nodes, double lr)
{
    numNodes_m_u = nodes;
    lr_m_d = lr;
}