#ifndef _NNLIB_H_
#define _NNLIB_H_

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

#define MAXLAYERS 4
#define MAXNODES 4

class Layer;
class NeuralNetwork;

NeuralNetwork *readNetFile(std::string filename);

class Layer
{
public:
    Layer(unsigned nodes, double lr); // initialize learning rate, number of nodes

    void Display();                          // display all values to the console
    void CreateMesh(unsigned outNodes);      // initialize weights & biases & outnodes, grads = 0;
    void FeedForward();                      // take results of previous layer and apply this layer's weights and biases
    void BackProp();                         // calculate the gradient for the prev layer
    void BackProp(double outputs[MAXNODES]); // calculate the gradient for the last hidden layer (to be ran by output layer)
    void Update();                           // adjust this layer's weights and biases by the gradient
    void SaveConnections(std::string filename);
    void SetMesh(double weights[MAXNODES][MAXNODES], double biases[MAXNODES]);

    Layer *prevLayer_pC = nullptr; // the preceding layer in the network

    double intakes_dA[MAXNODES] = {0.0};   // the values that enter this layer [numNodes]
    double results_dA[MAXNODES] = {0.0};   // the values that leave this layer [outNodes]
    double gradients_dA[MAXNODES] = {0.0}; // the step by which each weight should be adjusted [outNodes]

private:
    double weights_m_d2A[MAXNODES][MAXNODES]; // the mesh weights [numNodes][outNodes]
    double biases_m_dA[MAXNODES];             // the biases applied to the values [outNodes]

    unsigned numNodes_m_u = 0; // the number of nodes in this layer
    unsigned outNodes_m_u = 0; // the number of nodes in the next layer
    double lr_m_d = 0;         // the learning rate of the network
};

class NeuralNetwork
{
public:
    NeuralNetwork(unsigned inodes, double lr); // initialize the first layer of the network, learning rate

    void Display();                     // shows the values in the network and display each layer
    void AddLayer(unsigned nodes);      // add a layer to the network, adjust topology
    void Test(double inputs[MAXNODES]); // pass a set of inputs to the network and get an output
    double *Train(                      // given ins & outs, test, backprop and apply to get the network to learn
        double inputs[MAXNODES],
        double outputs[MAXNODES]);
    void SaveNetwork(std::string filename);
    void SetLayer(unsigned l, double weights[MAXNODES][MAXNODES], double biases[MAXNODES]);

private:
    Layer *layers_m_cpA[MAXLAYERS];

    unsigned topology_m_uA[MAXLAYERS];     // the number of nodes in each layer
    double inputs_m_dA[MAXNODES] = {0.0};  // the most recent inputs to the network
    double results_m_dA[MAXNODES] = {0.0}; // the output of the network given most recent inputs
    double outputs_m_dA[MAXNODES] = {0.0}; // the expected result given most recent input

    unsigned numLayers_m_u = 0;    // the number of layers
    double learningRate_m_d = 0.0; // the rate at which the network learns
};

#endif