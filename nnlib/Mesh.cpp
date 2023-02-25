#include "nnlib.h"

void Mesh::Display()
{
    for (unsigned ic = 0; ic < nodesIn_m_u; ic++)
    {
        for (unsigned oc = 0; oc < nodesOut_m_u; oc++)
        {
            std::cout << "\tW" << oc << ": " << connects_m_cpA[ic][oc]->weight_d << std::endl;
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
            connects_m_cpA[ic][oc] = new Connection();
            bias_m_d[oc] = init_weight();
        }
    }
}