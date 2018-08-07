#pragma once

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <string>

class Classifier{
public:
    using features_t = std::vector<double>;
    using proba_t = std::vector<double>;

    virtual ~Classifier() {}
    virtual void predict_proba(const features_t& features, proba_t* proba) = 0;
    virtual void predict_proba(const std::vector<features_t>& features, std::vector<proba_t>* proba);
    virtual bool init(const std::string& file_name) = 0;
    virtual int init(const char* raw_data, int raw_data_size) = 0;
};



#endif //CLASSIFIER_H
