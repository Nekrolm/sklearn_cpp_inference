#include "stdscaler.h"

#include <fstream>
#include <assert.h>
#include <cmath>

StdScaler::StdScaler()
{

}


void StdScaler::init(const std::string& filename)
{

    std::ifstream input(filename, std::ios_base::in | std::ios_base::binary);
    std::vector<char> data(std::istreambuf_iterator<char>(input), (std::istreambuf_iterator<char>()));

    size_t N = data.size() / sizeof (double) / 2;

    double* raw  = reinterpret_cast<double*>(data.data());

    mean_ = feature_t(raw, raw + N);
    vars_ = feature_t(raw + N, raw + 2*N);
}


void StdScaler::transform(const feature_t& features, feature_t* transformed_features)
{
    assert(transformed_features != nullptr);

    transformed_features->clear();

    for (size_t i = 0; i < features.size(); ++i){
        transformed_features->push_back( (features[i] - mean_[i]) / std::sqrt(vars_[i])  );
    }
}

Transformer::feature_t StdScaler::transform(const Transformer::feature_t &features)
{
    feature_t ret(0);
    transform(features, &ret);
    return  ret;
}
