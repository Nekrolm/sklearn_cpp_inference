#pragma once

#include "ensemble_classifier.h"
#include <memory>

namespace sklearn::classifier::ensemble {

class SKLEARN_API RandomForestClassifier : public EnsembleClassifier
{
public:
    RandomForestClassifier(AlgoType algo = WEIGHTED_MEAN) :
        EnsembleClassifier(algo)
    {}

    virtual ~RandomForestClassifier() override {}

    /*
       Binary file format:

       int N_estimators;
       TreeClassifier[N] estimators;
     */
    virtual int init(const char* raw_data, int raw_data_size) override;

    virtual bool init(const std::string& fname) override;
};

}
