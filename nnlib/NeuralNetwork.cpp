#include "nnlib.h"

void NeuralNetwork::SetLayer(unsigned l, double weights[MAXNODES][MAXNODES], double biases[MAXNODES])
{
    layers_m_cpA[l]->SetMesh(weights, biases);
}

void NeuralNetwork::SaveNetwork(std::string filename)
{
    std::ofstream outFile(filename);

    if (!outFile)
    {
        std::cout << std::endl;
        std::cout << "********************************" << std::endl;
        std::cout << "   Error! Could not open file" << std::endl;
        std::cout << "********************************" << std::endl;
        std::cout << std::endl;
    }
    else
    {
        outFile << numLayers_m_u << ' ' << learningRate_m_d << ' ';
        for (unsigned l = 0; l < numLayers_m_u; l++)
        {
            outFile << topology_m_uA[l] << ' ';
        }
        outFile << std::endl;
        outFile.close();
        for (unsigned l = 0; l < numLayers_m_u - 1; l++)
        {
            layers_m_cpA[l]->SaveConnections(filename);
        }
    }
}

double *NeuralNetwork::Train(double inputs[MAXNODES], double outputs[MAXNODES])
{
    Test(inputs);

    for (unsigned o = 0; o < topology_m_uA[numLayers_m_u - 1]; o++)
    {
        outputs_m_dA[o] = outputs[o];
    }

    layers_m_cpA[numLayers_m_u - 1]->BackProp(outputs);
    for (unsigned l = numLayers_m_u - 2; l > 0; l--)
    {
        layers_m_cpA[l]->BackProp();
    }

    for (unsigned l = 0; l < numLayers_m_u - 1; l++)
    {
        layers_m_cpA[l]->Update();
    }

    double *results = results_m_dA;
    double err = 0.0;
    for (unsigned r = 0; r < topology_m_uA[numLayers_m_u - 1]; r++)
    {
        err += abs(results[r] - outputs[r]);
    }
    totalError_m_d += (err / topology_m_uA[numLayers_m_u - 1]);
    timesTrained_m_u++;

    return results;
}

void NeuralNetwork::Test(double inputs[MAXNODES])
{
    for (unsigned i = 0; i < topology_m_uA[0]; i++)
    {
        layers_m_cpA[0]->intakes_dA[i] = inputs[i];
        inputs_m_dA[i] = inputs_m_dA[i];
    }

    for (unsigned l = 0; l < numLayers_m_u; l++)
    {
        layers_m_cpA[l]->FeedForward();
    }

    for (unsigned o = 0; o < topology_m_uA[numLayers_m_u - 1]; o++)
    {
        results_m_dA[o] = layers_m_cpA[numLayers_m_u - 1]->intakes_dA[o];
    }
}

void NeuralNetwork::AddLayer(unsigned nodes)
{
    topology_m_uA[numLayers_m_u] = nodes;
    layers_m_cpA[numLayers_m_u - 1]->CreateMesh(nodes);
    layers_m_cpA[numLayers_m_u] = new Layer(nodes, learningRate_m_d);
    layers_m_cpA[numLayers_m_u]->prevLayer_pC = layers_m_cpA[numLayers_m_u - 1];
    numLayers_m_u++;
}

void NeuralNetwork::Display(bool displayLayers /*= true*/, int precision /*= 6*/)
{
    std::cout << "------------ Neural Network ------------" << std::fixed << std::setprecision(precision) << std::endl;
    std::cout << std::setw(12) << "Topology: ";
    for (unsigned l = 0; l < numLayers_m_u; l++)
    {
        std::cout << topology_m_uA[l] << std::setw(MAXNODES / 10 + 3);
    }
    std::cout << std::endl;
    std::cout << std::setw(12) << "Learning: " << learningRate_m_d << std::setw(MAXNODES / 10 + 8) << std::endl;
    std::cout << std::setw(12) << "RA Error: " << ((timesTrained_m_u > 0) ? (totalError_m_d / (double)timesTrained_m_u) : (0.0)) << std::setw(MAXNODES / 10 + 8) << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(12) << "Inputs: ";
    for (unsigned i = 0; i < topology_m_uA[0]; i++)
    {
        std::cout << inputs_m_dA[i] << ' ' << std::setw(MAXNODES / 10 + 8);
    }
    std::cout << std::endl;
    std::cout << std::setw(12) << "Outputs: ";
    for (unsigned o = 0; o < topology_m_uA[numLayers_m_u - 1]; o++)
    {
        std::cout << outputs_m_dA[o] << ' ' << std::setw(MAXNODES / 10 + 8);
    }
    std::cout << std::endl;
    std::cout << std::setw(12) << "Results: ";
    for (unsigned r = 0; r < topology_m_uA[numLayers_m_u - 1]; r++)
    {
        std::cout << results_m_dA[r] << ' ' << std::setw(MAXNODES / 10 + 8);
    }
    std::cout << std::endl;
    std::cout << std::endl;

    if (displayLayers)
    {

        for (unsigned l = 0; l < numLayers_m_u; l++)
        {
            std::cout << "  ------------- Layer: " << l << " -------------  " << std::endl;
            layers_m_cpA[l]->Display();
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

NeuralNetwork::NeuralNetwork(unsigned inodes, double lr)
{
    topology_m_uA[numLayers_m_u] = inodes;
    layers_m_cpA[numLayers_m_u++] = new Layer(inodes, lr);
    learningRate_m_d = lr;
}