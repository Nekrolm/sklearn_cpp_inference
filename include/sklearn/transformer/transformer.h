#pragma once

#include <string>
#include <vector>

#include <sklearn/dll.h>
#include <sklearn/types.h>

namespace sklearn::transformer {

class SKLEARN_API Transformer
{
public:
    virtual ~Transformer() {}
    virtual bool init(const std::string& filename);

    virtual int init(const char* raw_data, int raw_data_size)                          = 0;
    virtual void transform(const feature_t& features, feature_t* transformed_features) = 0;
    virtual feature_t transform(const feature_t& features)                             = 0;
    virtual std::vector<feature_t> transform(const std::vector<feature_t>& features);
};

}
