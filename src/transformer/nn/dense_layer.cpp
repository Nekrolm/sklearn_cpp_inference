#include <sklearn/transformer/nn/dense_layer.h>

#include <utils/binary_read_helper.h>

#include <algorithm>
#include <numeric>



namespace sklearn::activation {

enum class ActivationType : int32_t
{
    RELU = 0,
    SOFTMAX
};

feature_t relu(const feature_t& f)
{
    feature_t ret(0);

    std::transform(f.begin(), f.end(), back_inserter(ret),
                   [](feature_t::value_type x) { return std::max(x, 0.); });
    return ret;
}

feature_t softmax(const feature_t& f)
{
    feature_t ret(0);

    std::transform(f.begin(), f.end(), back_inserter(ret),
                   [](feature_t::value_type x) { return std::exp(x); });

    feature_t::value_type sum_v = std::accumulate(ret.begin(), ret.end(), feature_t::value_type(0));

    for (auto& v : ret)
        v /= sum_v;

    return ret;
}

feature_t none(const feature_t& f)
{
    return f;
}

std::function<feature_t(const feature_t&)> make_activation(ActivationType type)
{
    static const std::unordered_map<ActivationType, std::function<feature_t(feature_t)>> selector
    {
        { ActivationType::RELU, relu },
        { ActivationType::SOFTMAX, softmax }
    };

    if (selector.count(type))
        return selector.at(type);

    return none;
}

}

namespace sklearn::transformer::nn {

DenseLayer::DenseLayer()
{}

int DenseLayer::init(const char* raw_data, int raw_data_size)
{
    std::stringstream s(std::string(raw_data, raw_data_size));

    using namespace boost::numeric::ublas;



    int ofs = 0;

    int32_t n, m, activation_type;

    ofs += utils::binary::readVal<int32_t>(s, n);
    ofs += utils::binary::readVal<int32_t>(s, m);
    ofs += utils::binary::readVal<int32_t>(s, activation_type);

    std::vector<double> matr_raw(n * m);
    ofs += utils::binary::readVec(s, matr_raw);


    weights_.resize(n, m);
    std::copy(matr_raw.begin(), matr_raw.end(), weights_.data().begin());

    activation_ = activation::make_activation(static_cast<activation::ActivationType>(activation_type));

    std::vector<double> bias_raw(m);
    ofs += utils::binary::readVec(s, bias_raw);

    bias_.resize(m);
    std::copy(bias_raw.begin(), bias_raw.end(), bias_.begin());

    return ofs;
}

void DenseLayer::transform(const feature_t& feat, feature_t* output)
{
    *output = transform(feat);
}

feature_t DenseLayer::transform(const feature_t& feat)
{
    using namespace boost::numeric::ublas;
    vector<feature_t::value_type> input(feat.size());
    std::copy(feat.begin(), feat.end(), input.begin());

    vector<feature_t::value_type> transformed = prod(input, weights_) + bias_;

    feature_t ret(0);
    std::copy(transformed.begin(), transformed.end(), back_inserter(ret));

    return activation_(ret);
}

std::vector<feature_t> DenseLayer::transform(const std::vector<feature_t>& feat)
{
    return Transformer::transform(feat);
}

}
