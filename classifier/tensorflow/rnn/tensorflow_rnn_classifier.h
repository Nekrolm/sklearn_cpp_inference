#ifndef TENSORFLOW_RNN_CLASSIFIER_H
#define TENSORFLOW_RNN_CLASSIFIER_H

#include "classifier.h"

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>



class TensorFlowRNNClassifier : public Classifier
{
public:
    TensorFlowRNNClassifier();
    virtual ~TensorFlowRNNClassifier() override {};

    virtual int init (const char* raw_data, int data_size) override {return 0;}
    virtual bool init (const std::string& dir) override;
    virtual void predict_proba (const features_t& feat, proba_t* proba) override;


    virtual void predict_proba (const std::vector<features_t>& feat, std::vector<proba_t>* proba) override;

private:

    void predict_internal (const std::vector<features_t>& feat, tensorflow::Tensor* output);

    tensorflow::MetaGraphDef graph_def_;
    std::unique_ptr<tensorflow::Session> session_;

    const std::string  GRAPH = "model.ckpt.meta";
    const std::string CHECKPOINT = "model.ckpt";

};

#endif // TENSORFLOW_RNN_CLASSIFIER_H
