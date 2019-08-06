#ifndef LAYER_MAKER_H
#define LAYER_MAKER_H

#include "nn_layer_base.h"

#include <memory>

namespace sklearn::transformer::nn {

enum class LayerType : int32_t
{
    DENSE = 0,
};

class SKLEARN_API LayerMaker
{
public:
    static std::unique_ptr<NNLayerBase> makeLayer(LayerType type);
};

}
#endif // LAYER_MAKER_H
