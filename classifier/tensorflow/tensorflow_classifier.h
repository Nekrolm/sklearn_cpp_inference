#ifndef TENSORFLOW_CLASSIFIER_H
#define TENSORFLOW_CLASSIFIER_H

#include "../classifier.h"

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>


class TensorFlowClassifier : public Classifier
{
public:
    TensorFlowClassifier();
    virtual ~TensorFlowClassifier() override;

    virtual int init (const char* raw_data, int data_size) override {return 0;}
    virtual bool init (const std::string& dir) override;

    virtual void predict_proba (const std::vector<features_t>& feat, std::vector<proba_t>* proba) override;

    virtual void predict_proba (const features_t& feat, proba_t* proba) override;



private:

    void predict_internal (const std::vector<features_t>& feat, tensorflow::Tensor* out);


    tensorflow::SavedModelBundle bundle_;
    const std::string X_NAME;
    const std::string SCORES_NAME;
};

#endif // TENSORFLOW_CLASSIFIER_H
