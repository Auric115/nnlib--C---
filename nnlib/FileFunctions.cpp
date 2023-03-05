#include "nnlib.h"

NeuralNetwork *readNetFile(std::string filename)
{
    std::ifstream netFile(filename);

    if (!netFile)
    {
        std::cout << std::endl;
        std::cout << "********************************" << std::endl;
        std::cout << "   Error! Could not open file: " << filename << std::endl;
        std::cout << "********************************" << std::endl;
        std::cout << std::endl;
        return nullptr;
    }
    else
    {
        unsigned num, n = 0;
        double lr;
        unsigned t[MAXLAYERS];
        netFile >> num >> lr >> t[n];
        NeuralNetwork *newNet = new NeuralNetwork(t[n++], lr);
        for (unsigned l = 1; l < num; l++)
        {
            netFile >> t[n];
            newNet->AddLayer(t[n++]);
        }
        for (unsigned l = 0; l < num - 1; l++)
        {
            double w[MAXNODES][MAXNODES] = {0.0};
            double b[MAXNODES] = {0.0};
            for (unsigned i = 0; i < t[l]; i++)
            {
                for (unsigned o = 0; o < t[l + 1]; o++)
                {
                    netFile >> w[i][o];
                }
            }
            for (unsigned o = 0; o < t[l + 1]; o++)
            {
                netFile >> b[o];
            }
            newNet->SetLayer(l, w, b);
        }
        return newNet;
    }
}