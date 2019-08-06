#pragma once

#include "transformer.h"

namespace sklearn::transformer {

class SKLEARN_API StdScaler : public Transformer
{
public:
    StdScaler();
    ~StdScaler() override {}

    /*
       Binary file format:

       int features_dim;
       double[features_dim] mean;
       double[features_dim] variance;
     */
    int init(const char* raw_data, int raw_data_size) override;

    void transform(const feature_t& features, feature_t* transformed_features) override;
    feature_t transform(const feature_t& features) override;

private:
    feature_t mean_;
    feature_t vars_;
};

}
