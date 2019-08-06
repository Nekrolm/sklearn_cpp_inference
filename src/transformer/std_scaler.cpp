#include <sklearn/transformer/std_scaler.h>

#include <assert.h>
#include <cmath>
#include <cstring>

namespace sklearn::transformer {

StdScaler::StdScaler()
{}

int StdScaler::init(const char* raw_data, int raw_data_size)
{
    if (sizeof(int) > raw_data_size)
        return 0;

    int N = 0;
    memcpy(&N, raw_data, sizeof(int));

    raw_data += sizeof(int);

    if (2 * N * sizeof(double) + sizeof(int) > raw_data_size)
        return 0;

    mean_.resize(N);
    memcpy(mean_.data(), raw_data, N * sizeof(double));
    raw_data += N * sizeof(double);
    vars_.resize(N);
    memcpy(vars_.data(), raw_data, N * sizeof(double));

    return sizeof(int) + 2 * N * sizeof(double);
}

void StdScaler::transform(const feature_t& features, feature_t* transformed_features)
{
    assert(transformed_features != nullptr);

    transformed_features->clear();

    for (size_t i = 0; i < features.size(); ++i)
    {
        transformed_features->push_back((features[i] - mean_[i]) / std::sqrt(vars_[i]));
    }
}

feature_t StdScaler::transform(const feature_t& features)
{
    feature_t ret(0);

    transform(features, &ret);
    return ret;
}

}
