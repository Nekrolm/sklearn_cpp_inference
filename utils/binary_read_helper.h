#pragma once

#include <sstream>
#include <vector>


namespace  binary{

inline bool readData(std::stringstream& s, void* dst, int cnt)
{
    return bool(s.read(reinterpret_cast<char*>(dst), cnt));
}

template<class T>
inline long readVec(std::stringstream& s, std::vector<T>& data)
{
    readData(s, data.data(), data.size() * sizeof(T));
    return s.gcount();
}

template<class T>
inline long readVal(std::stringstream& s, T& v)
{
    readData(s, &v, sizeof(T));
    return  s.gcount();
}

}
