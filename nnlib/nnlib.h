#ifndef _NNLIB_H_
#define _NNLIB_H_

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

#define DEBUG true

#define MAXLAYERS 4
#define MAXNEURONS 4

class Connection;
class Neuron;
class Mesh;
class Layer;
class NeuralNetwork;

double squish(double x);
double dSquish(double x);

double init_weight();

class Mesh
{
public:
    Mesh(Layer *layerIn, Layer *layerOut);

    void Display();
    void FeedForward();

private:
    Layer *layerIn_m_cP;
    Layer *layerOut_m_cP;
    double weights_m_d2A[MAXNEURONS][MAXNEURONS];

    double bias_m_d[MAXNEURONS];

    unsigned nodesIn_m_u;
    unsigned nodesOut_m_u;
};

class Layer
{
public:
    Layer(unsigned nodes);

    void Display();
    unsigned GetNum() { return numNeurons_u; }
    void SetIntakes(double in[MAXNEURONS]);

    double intakes_d[MAXNEURONS];
    double delta_m_d[MAXNEURONS];
    unsigned numNeurons_u;
};

class NeuralNetwork
{
public:
    NeuralNetwork(unsigned nodes, double learningRate);

    void AddLayer(unsigned nodes);
    void Display();
    void Test(double inputs[MAXNEURONS]);
    void DisplayResults();
    void Train(double inputs[MAXNEURONS], double outputs[MAXNEURONS]);

    double inputs_dA[MAXNEURONS];
    double results_dA[MAXNEURONS];
    double outputs_dA[MAXNEURONS];

private:
    Layer *layers_m_cpA[MAXLAYERS];
    Mesh *mesh_m_cpA[MAXLAYERS - 1];

    unsigned topology_m_uA[MAXLAYERS];

    unsigned numLayers_m_u = 0;
    double learningRate_m_d;
};

#endif