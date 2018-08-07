#pragma once
#ifndef RANDOM_FOREST_CLASSIFIER_H
#define RANDOM_FOREST_CLASSIFIER_H

#include "ensemble_classifier.h"
#include <memory>



class RandomForestClassifier : public EnsembleClassifier
{
public:
    RandomForestClassifier(AlgoType algo = WEIGHTED_MEAN) :
        EnsembleClassifier(algo)
    {}

    virtual ~RandomForestClassifier() override {}

    virtual int init(const char* raw_data, int raw_data_size) override;

    virtual bool init(const std::string& fname) override;


};




#endif //RANDOM_FOREST_CLASSIFIER_H
