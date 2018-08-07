#include "tensorflow_classifier.h"

#include <iostream>

namespace tf = tensorflow;

TensorFlowClassifier::TensorFlowClassifier() :
    X_NAME ("Placeholder:0"),
    SCORES_NAME ("dnn/head/predictions/probabilities:0")
{

}

TensorFlowClassifier::~TensorFlowClassifier()
{

}

bool TensorFlowClassifier::init (const std::string& dir)
{
    std::cerr << "Start init DNN" << std::endl;

    auto status = tf::LoadSavedModel (tf::SessionOptions(), tf::RunOptions(), dir, {"serve"}, &bundle_);

    if (!status.ok())
    {
        std::cerr << status.error_message() << std::endl;
    }

    return status.ok();
}


void TensorFlowClassifier::predict_internal (const std::vector<features_t>& feat, tf::Tensor* out)
{
    auto x = tf::Tensor (tf::DT_FLOAT, tf::TensorShape ({int (feat.size()), int (feat[0].size()) }));
    auto matr = x.matrix<float>();
    for (int i = 0; i < feat.size(); ++i)
        for (int j = 0; j < feat[0].size(); ++j)
            matr (i, j) = feat[i][j];


    std::vector<std::pair<std::string, tf::Tensor>> inputs = {{ X_NAME, x }};
    std::vector<tf::Tensor> outputs;

    auto run_status = bundle_.session->Run (inputs, {SCORES_NAME}, {}, &outputs);

    if (!run_status.ok())
    {
        return;
    }

    *out = outputs.front();
}


void TensorFlowClassifier::predict_proba (const std::vector<Classifier::features_t>& feat, std::vector<Classifier::proba_t>* proba)
{
    proba->clear();

    tf::Tensor output;
    predict_internal (feat, &output);


    auto out = output.matrix<float>();

    for (int i = 0; i < feat.size(); ++i)
    {
        proba->emplace_back();
        for (int j = 0; j < out.dimensions().at (1); ++j)
            (proba->back()).push_back (out (i, j));
    }
}

void TensorFlowClassifier::predict_proba (const Classifier::features_t& feat, Classifier::proba_t* proba)
{
    proba->clear();

    tf::Tensor output;
    predict_internal ({feat}, &output);

    auto out = output.matrix<float>();

    for (int i = 0; i < out.dimensions().at (1); ++i)
    {
        proba->push_back (out (0, i));
    }

}
