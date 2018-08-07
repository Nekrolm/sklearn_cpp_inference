#include "stdscaler.h"

#include <assert.h>
#include <cmath>

StdScaler::StdScaler()
{

}


int StdStaler::init(const char* raw_data, int raw_data_size)
{
    int cnt = 0;
    if (cnt + sizeof(int) > raw_data_size){
        return 0;
    }
    int N = *reinterpret_cast<int*>(raw_data);
    cnt += sizeof(int);
    
    if (2 * N * sizeof(double) + cnt > raw_data_size)
        return 0;
        
    double* raw  = reinterpret_cast<double*>(raw_data + cnt);

    
    mean_ = feature_t(raw, raw + N);
    vars_ = feature_t(raw + N, raw + 2*N);
    
    return cnt + 2 * N * sizeof(double);
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
