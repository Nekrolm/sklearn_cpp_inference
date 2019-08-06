#pragma once


#include <vector>
#include <string>

#include <sklearn/types.h>
#include <sklearn/dll.h>

namespace sklearn::classifier {

class SKLEARN_API Classifier
{
public:

    virtual ~Classifier() {}
    virtual void predict_proba(const feature_t& features, proba_t* proba) = 0;
    virtual void predict_proba(const std::vector<feature_t>& features, std::vector<proba_t>* proba);
    virtual bool init(const std::string& file_name)           = 0;
    virtual int init(const char* raw_data, int raw_data_size) = 0;
};

}
