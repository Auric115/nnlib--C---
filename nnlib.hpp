#ifndef _NNLIB_HPP_
#define _NNLIB_HPP_

#include <iostream>
#include <cmath>
#include <cassert>

#define MAXLAYERS 4
#define MAXNEURONS 4

class Connection;
class Neuron;
class Mesh;
class Layer;
class NeuralNetwork;

double squish(double x) { return std::tanh(x); }
double dSquish(double x) { return (1.0 - (x * x)); }

double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

class Mesh
{
public:
    Mesh();

private:
    Layer *layerIn_c;
    Layer *layerOut_c;
    Connection *connects_m_cpA[MAXNEURONS][MAXNEURONS];
    double bias_m_d[MAXNEURONS][MAXNEURONS];
    unsigned numInputs = 0;
    unsigned numOutputs = 0;
};

class Layer
{
public:
    Layer(unsigned nodes);
    void Display();

private:
    Neuron *nodes_m_cpA[MAXNEURONS];
    double intakes_d[MAXNEURONS];
    double results_d[MAXNEURONS];
    double delta_m_d[MAXNEURONS];
    unsigned numNeurons;
};

class NeuralNetwork
{
public:
    NeuralNetwork(unsigned nodes, double learningRate);
    void AddLayer(unsigned nodes);
    void Display();

private:
    Layer *layers_m_cpA[MAXLAYERS];
    Mesh *mesh_m_cpA[MAXLAYERS - 1];
    unsigned topology_m_uA[MAXLAYERS];
    double inputs_dA[MAXNEURONS];
    double outputs_dA[MAXNEURONS];
    unsigned numLayers_m_u;
    double learningRate_m_d;
};

#endif