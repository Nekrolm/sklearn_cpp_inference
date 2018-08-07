#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn_layer_base.h"

#include <functional>


#include <boost/numeric/ublas/matrix.hpp>

class DenseLayer : public NNLayerBase
{
public:    
    DenseLayer();
    ~DenseLayer() override {}

    void init(const std::string& filename) override {}
    int init(const char* raw_data, int raw_data_size) override;

    virtual void transform(const feature_t& feat, feature_t* output) override;
    virtual feature_t transform(const feature_t& feat) override;
    virtual std::vector<feature_t> transform(const std::vector<feature_t>& feat) override;

private:
    std::function<feature_t(const feature_t&)> activation_;

    boost::numeric::ublas::matrix<feature_t::value_type> weights_;
    boost::numeric::ublas::vector<feature_t::value_type> bias_;

};

#endif // NN_LAYER_H
