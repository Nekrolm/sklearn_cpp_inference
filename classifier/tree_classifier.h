#pragma once

#ifndef TREE_CLASSIFIER_H
#define TREE_CLASSIFIER_H


#include "classifier.h"



class TreeClassifier : public Classifier
{
public:
    TreeClassifier();
    ~TreeClassifier() override;

    virtual void predict_proba (const features_t& features, proba_t* proba) override;
    virtual bool init (const std::string& file_name) override { return false; }
    virtual int init (const char* raw_data, int raw_data_size) override;


private:
    std::vector<int> left_children_, right_children_;
    std::vector<double> threshold_;
    std::vector<int> feature_id_;
    std::vector<proba_t> value_;


    void predict_internal (const features_t& features, proba_t* proba);


    static const int NO_NODE = -1;

};


#endif //TREE_CLASSIFIER_H
