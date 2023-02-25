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

class Connection
{
public:
    Connection();

    double weight_d;

private:
};

class Neuron
{
public:
    Neuron();

private:
};

class Mesh
{
public:
    Mesh(Layer *layerIn, Layer *layerOut);

    void Display();

private:
    Layer *layerIn_m_cP;
    Layer *layerOut_m_cP;
    Connection *connects_m_cpA[MAXNEURONS][MAXNEURONS];

    double bias_m_d[MAXNEURONS];

    unsigned nodesIn_m_u;
    unsigned nodesOut_m_u;
};

class Layer
{
public:
    Layer(unsigned nodes);

    void Display();
    unsigned GetNum() { return numNeurons; }

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

    unsigned numLayers_m_u = 0;
    double learningRate_m_d;
};

#endif