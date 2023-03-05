#include "nnlib.h"

double squish(double x) { return std::tanh(x); }
double dSquish(double x) { return (1.0 - (x * x)); }

double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

Layer::Layer(unsigned nodes, double lr)
{
    numNodes_m_u = nodes;
    lr_m_d = lr;
}