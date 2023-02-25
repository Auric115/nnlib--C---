#include "nnlib.h"

Connection::Connection()
{
    weight_d = init_weight();
    if (DEBUG)
    {
        assert(weight_d <= 1 && weight_d >= 0);
    }
}