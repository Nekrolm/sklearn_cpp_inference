#include "tensorflow_rnn_classifier.h"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
namespace tf = tensorflow;

TensorFlowRNNClassifier::TensorFlowRNNClassifier()
{

}


bool TensorFlowRNNClassifier::init (const std::string& dir)
{
    auto graph_path = (fs::path (dir) / GRAPH).string();
    auto chkpt_path = (fs::path (dir) / CHECKPOINT).string();


    session_.reset (tf::NewSession (tf::SessionOptions()));

    if (!session_) return false;

    auto status = tf::ReadBinaryProto (tf::Env::Default(), graph_path, &graph_def_);

    if (!status.ok())
        return false;


    status = session_->Create (graph_def_.graph_def());

    if (!status.ok())
        return false;

    tf::Tensor ckptTensor (tf::DT_STRING, tf::TensorShape());
    ckptTensor.scalar<std::string>() () = chkpt_path;

    status = session_->Run (
    {{ graph_def_.saver_def().filename_tensor_name(), ckptTensor },},
    {},
    {graph_def_.saver_def().restore_op_name() },
    nullptr);

    if (!status.ok())
        return  false;

    return true;

}


void TensorFlowRNNClassifier::predict_internal (const std::vector<features_t>& feat, tf::Tensor* output)
{
    auto x = tf::Tensor (tf::DT_FLOAT, tf::TensorShape ({1, int (feat.size()), int (feat[0].size()) }));
    auto matr = x.tensor<float, 3>();
    for (int i = 0; i < feat.size(); ++i)
        for (int j = 0; j < feat[0].size(); ++j)
            matr (0, i, j) = feat[i][j];

    std::vector<std::pair<std::string, tf::Tensor>> inputs = {{ "input_data", x }};

    std::vector<tf::Tensor> outputs;

    auto status = session_->Run (inputs, {"prediction"}, {}, &outputs);


    if (!status.ok())
        return;

    *output = outputs.front();

}


void TensorFlowRNNClassifier::predict_proba (const std::vector<features_t>& feat, std::vector<proba_t>* proba)
{
    proba->clear();


    tf::Tensor output;
    predict_internal (feat, &output);

    auto out = output.matrix<float>();

    proba->emplace_back();

    for (int i = 0; i < out.dimensions().at (1); ++i)
    {
        (proba->back()).push_back (out (0, i));
    }

}


void TensorFlowRNNClassifier::predict_proba (const features_t& feat, proba_t* proba)
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
