#include "nnlib.h"

void Mesh::FeedForward()
{
    double activation;
    for (unsigned o = 0; o < layerOut_m_cP->numNeurons_u; o++)
    {
        activation = 0;
        for (unsigned i = 0; i < layerIn_m_cP->numNeurons_u; i++)
        {
            activation += layerIn_m_cP->intakes_d[i] * weights_m_d2A[i][o];
        }
        activation += bias_m_d[o];
        activation = squish(activation);
        layerOut_m_cP->intakes_d[o] = activation;
    }
}

void Mesh::Display()
{
    for (unsigned ic = 0; ic < nodesIn_m_u; ic++)
    {
        for (unsigned oc = 0; oc < nodesOut_m_u; oc++)
        {
            std::cout << "\tW" << oc << ": " << weights_m_d2A[ic][oc] << std::endl;
        }
        std::cout << std::endl;
    }

    for (unsigned oc = 0; oc < nodesOut_m_u; oc++)
    {
        std::cout << "\tB" << oc << ": " << bias_m_d[oc] << std::endl;
    }
    std::cout << std::endl;
}

Mesh::Mesh(Layer *layerIn, Layer *layerOut)
{
    layerIn_m_cP = layerIn;
    layerOut_m_cP = layerOut;
    nodesIn_m_u = layerIn_m_cP->GetNum();
    nodesOut_m_u = layerOut_m_cP->GetNum();
    for (unsigned ic = 0; ic < layerIn_m_cP->GetNum(); ic++)
    {
        for (unsigned oc = 0; oc < layerOut_m_cP->GetNum(); oc++)
        {
            weights_m_d2A[ic][oc] = init_weight();
            bias_m_d[oc] = init_weight();
        }
    }
}