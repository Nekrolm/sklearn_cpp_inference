#include <sklearn/classifier/nn/nn_classifier.h>

#include <fstream>

#include <utils/binary_read_helper.h>

#include <sklearn/transformer/nn/layer_maker.h>

namespace sklearn::classifier::nn {

NNClassifier::NNClassifier()
{}

bool NNClassifier::init(const std::string& filename)
{
    std::ifstream input(filename, std::ios_base::in | std::ios_base::binary);
    std::vector<char> data(std::istreambuf_iterator<char>(input), (std::istreambuf_iterator<char>()));

    return init(data.data(), data.size());
}

int NNClassifier::init(const char* raw_data, int raw_data_size)
{
    layers_.clear();

    if (raw_data_size < sizeof(int32_t))
        return 0;


    int32_t layers_cnt;

    int read_bytes = 0;

    layers_cnt  = *reinterpret_cast<const int32_t*>(raw_data);
    read_bytes += sizeof(int32_t);

    for (int i = 0; i < layers_cnt; ++i)
    {
        layers_.emplace_back(transformer::nn::LayerMaker::makeLayer(transformer::nn::LayerType::DENSE));
        read_bytes += layers_.back()->init(raw_data + read_bytes, raw_data_size - read_bytes);
    }

    return read_bytes;
}

void NNClassifier::predict_proba(const std::vector<feature_t>& feat, std::vector<proba_t>* proba)
{
    std::vector<feature_t> x = feat;

    for (auto& layer : layers_)
        x = layer->transform(x);
    *proba = x;
}

void NNClassifier::predict_proba(const feature_t& feat, proba_t* proba)
{
    feature_t x = feat;

    for (auto& layer : layers_)
        x = layer->transform(x);
    *proba = x;
}

}
