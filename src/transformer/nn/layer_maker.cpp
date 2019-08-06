#include <sklearn/transformer/nn/layer_maker.h>

#include <unordered_map>


#include <sklearn/transformer/nn/dense_layer.h>


namespace sklearn::transformer::nn {

std::unique_ptr<NNLayerBase> LayerMaker::makeLayer(LayerType type)
{
    const static std::unordered_map<LayerType, std::function<std::unique_ptr<NNLayerBase>()>> maker{
        { LayerType::DENSE,   []() { return std::make_unique<DenseLayer>(); }  }
    };

    if (maker.count(type))
        return maker.at(type)();

    return nullptr;
}

}
