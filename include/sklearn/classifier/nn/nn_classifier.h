#ifndef NN_CLASSIFIER_H
#define NN_CLASSIFIER_H

#include <sklearn/classifier/classifier.h>

#include <sklearn/transformer/nn/nn_layer_base.h>

#include <memory>

namespace sklearn::classifier::nn {

class NNClassifier : public Classifier
{
public:
    NNClassifier();
    ~NNClassifier() override {}

    bool init(const std::string& filename) override;

    /*
       Binary file format:

       int N_layers;
       NNLayers[N_layers] layers;
     */
    int init(const char* raw_data, int raw_data_size) override;

    void predict_proba(const std::vector<feature_t>& feat, std::vector<proba_t>* proba) override;

    void predict_proba(const feature_t& feat, proba_t* proba) override;

private:
    std::vector<std::unique_ptr<transformer::nn::NNLayerBase>> layers_;
};

}
#endif // NN_CLASSIFIER_H
