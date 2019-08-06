#pragma once

#ifndef TREE_CLASSIFIER_H
# define TREE_CLASSIFIER_H


# include "classifier.h"

namespace sklearn::classifier {

class SKLEARN_API TreeClassifier : public Classifier
{
public:
    TreeClassifier();
    ~TreeClassifier() override;

    virtual void predict_proba(const feature_t& features, proba_t* proba) override;
    virtual bool init(const std::string& file_name) override { return false; }


    /*
       Binary file format:

       int N_nodes; int N_classes;
       double[N_nodes] left_childrens;
       double[N_nodes] right_childrens;
       int[N_nodes] feature_in_node;
       double[N_nodes] threshold_in_node;
       double[N_nodes][N_classes] probabilities
     */
    virtual int init(const char* raw_data, int raw_data_size) override;

private:
    std::vector<int> left_children_, right_children_;
    std::vector<double> threshold_;
    std::vector<int> feature_id_;
    std::vector<proba_t> value_;


    void predict_internal(const feature_t& features, proba_t* proba);


    static const int NO_NODE = -1;
};

}
#endif // TREE_CLASSIFIER_H
