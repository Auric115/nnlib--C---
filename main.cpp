#include "nnlib/nnlib.h"
#include <ctime>
#include <sstream>

int main()
{
    srand(time(0));

    NeuralNetwork myNet(2, 0.7);
    myNet.AddLayer(4);
    myNet.AddLayer(1);
    myNet.Display();

    std::string line;
    double ins[MAXNODES], outs[MAXNODES];
    int i = 0;

    std::ifstream data("data.txt");

    if (data)
    {
        while (std::getline(data, line) && (i <= 1000000))
        {
            std::stringstream ss(line);

            ss >> ins[0] >> ins[1] >> outs[0];

            if ((ins[0] == ins[1] && !outs[0]) || (ins[0] != ins[1] && outs[0]))
            {
                myNet.Train(ins, outs);

                if (i++ % 2000 == 0)
                {
                    std::cout << "Trial: " << i << std::endl;
                    myNet.Display(false, 10);
                }
            }
        }
    }

    myNet.Display();
    std::cout << "Total Trials: " << i << std::endl;

    return 0;
}