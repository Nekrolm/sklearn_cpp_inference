#pragma once

#include "classifier.h"

#include <boost/numeric/ublas/matrix.hpp>

namespace sklearn::classifier{

class LogisticRegression : public Classifier
{
public:

    LogisticRegression();

    virtual ~LogisticRegression() {}
    //TODO
    bool init(const std::string &file_name) { return false; };
    virtual void predict_proba(const feature_t& features, proba_t* proba);
    virtual int init(const char* raw_data, int raw_data_size);


private:
    boost::numeric::ublas::matrix<feature_t::value_type> weights_;
    boost::numeric::ublas::vector<feature_t::value_type> bias_;


    // Classifier interface

};

}

