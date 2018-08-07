#ifndef NN_CLASSIFIER_H
#define NN_CLASSIFIER_H

#include "../classifier.h"

#include "../../transformer/nn/nn_layer_base.h"

#include <memory>

class NNClassifier : public Classifier
{
public:
    NNClassifier();
    ~NNClassifier() override {}

    virtual bool init(const std::string& filename) override;
    virtual int init(const char* raw_data, int raw_data_size) override;

    virtual void predict_proba(const std::vector<features_t>& feat, std::vector<proba_t>* proba) override;

    virtual void predict_proba(const features_t& feat, proba_t* proba) override;


private:
    std::vector<std::unique_ptr<NNLayerBase>> layers_;
};

#endif // NN_CLASSIFIER_H
