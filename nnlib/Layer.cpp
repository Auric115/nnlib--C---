#include "nnlib.h"

double squish(double x) { return std::tanh(x); }
double dSquish(double x) { return (1.0 - (x * x)); }

double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

void Layer::SetMesh(double weights[MAXNODES][MAXNODES], double biases[MAXNODES])
{
    for (unsigned ic = 0; ic < numNodes_m_u; ic++)
    {
        for (unsigned oc = 0; oc < outNodes_m_u; oc++)
        {
            weights_m_d2A[ic][oc] = weights[ic][oc];
        }
    }
    for (unsigned oc = 0; oc < outNodes_m_u; oc++)
    {
        biases_m_dA[oc] = biases[oc];
    }
}

void Layer::SaveConnections(std::string filename)
{
    std::ofstream outFile;

    outFile.open(filename, std::ios_base::app);

    if (outNodes_m_u > 0 && outFile)
    {
        outFile << std::fixed << std::setprecision(6);
        for (unsigned ic = 0; ic < numNodes_m_u; ic++)
        {
            for (unsigned oc = 0; oc < outNodes_m_u; oc++)
            {
                outFile << weights_m_d2A[ic][oc] << ' ';
            }
            outFile << std::endl;
        }
    }
    for (unsigned oc = 0; oc < outNodes_m_u; oc++)
    {
        outFile << biases_m_dA[oc] << ' ';
    }
    outFile << std::endl;
    outFile.close();
}

void Layer::Update()
{
    for (unsigned oc = 0; oc < outNodes_m_u; oc++)
    {
        biases_m_dA[oc] += (gradients_dA[oc] * lr_m_d);
        for (unsigned ic = 0; ic < numNodes_m_u; ic++)
        {
            weights_m_d2A[ic][oc] += (intakes_dA[ic] * gradients_dA[oc] * lr_m_d);
        }
    }
}

void Layer::BackProp()
{
    double error;
    for (unsigned ic = 0; ic < numNodes_m_u; ic++)
    {
        error = 0;
        for (unsigned oc = 0; oc < outNodes_m_u; oc++)
        {
            error += gradients_dA[oc] * weights_m_d2A[ic][oc];
        }
        prevLayer_pC->gradients_dA[ic] = error * dSquish(prevLayer_pC->results_dA[ic]);
    }
}

void Layer::BackProp(double outputs[MAXNODES])
{
    double error;
    for (unsigned o = 0; o < numNodes_m_u; o++)
    {
        prevLayer_pC->gradients_dA[o] = ((outputs[o] - intakes_dA[o]) * dSquish(intakes_dA[o]));
    }
}

void Layer::FeedForward()
{
    if (prevLayer_pC != nullptr)
    {
        for (unsigned i = 0; i < numNodes_m_u; i++)
        {
            intakes_dA[i] = prevLayer_pC->results_dA[i];
        }
    }

    double activation;
    for (unsigned oc = 0; oc < outNodes_m_u; oc++)
    {
        activation = 0;
        for (unsigned ic = 0; ic < numNodes_m_u; ic++)
        {
            activation += intakes_dA[ic] * weights_m_d2A[ic][oc];
        }
        activation += biases_m_dA[oc];
        activation = squish(activation);
        results_dA[oc] = activation;
    }
}

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