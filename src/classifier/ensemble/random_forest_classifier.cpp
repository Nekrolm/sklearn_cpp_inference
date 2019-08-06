
#include <sklearn/classifier/ensemble/random_forest_classifier.h>
#include <sklearn/classifier/tree_classifier.h>

#include <fstream>

namespace sklearn::classifier::ensemble {

int RandomForestClassifier::init(const char* raw_data, int raw_data_size)
{
    if (raw_data_size < sizeof(int))
        throw std::runtime_error("Error! can't read n_estimators");


    auto beg = raw_data;

    int n_estimators = *reinterpret_cast<const int*>(raw_data);
    raw_data += sizeof(int);

    raw_data_size -= sizeof(int);



    for (int i = 0; i < n_estimators; ++i)
    {
        auto tree     = std::make_shared<TreeClassifier>();
        int  read_cnt = tree->init(raw_data, raw_data_size);

        if (read_cnt > 0)
        {
            addClassifier(tree, 1.);
            raw_data      += read_cnt;
            raw_data_size -= read_cnt;
        }
        else
        {
            throw std::runtime_error("Error! can't read estimator");
        }
    }

    return raw_data - beg;
}

bool RandomForestClassifier::init(const std::string& fname)
{
    std::ifstream input(fname, std::ios_base::in | std::ios_base::binary);
    std::vector<char> data(std::istreambuf_iterator<char>(input), (std::istreambuf_iterator<char>()));

    return init(data.data(), data.size());
}

}
