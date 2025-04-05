#ifndef __DAT_OUTPUT_H__
#define __DAT_OUTPUT_H__
    
#include <string>
#include <vector>
#include <ostream>
#include <fstream>

namespace detail
{

template<class Array>
void write_out_dat_file_scal_3D(const std::string filename, const Array& U)
{

    using T = typename Array::value_type;

    std::ofstream os(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    
    if (!os)
    {
        throw std::runtime_error("error opening file: " + filename);
    }

    for (auto idx : U.rect_nd())
    {
        T val = U(idx);
        if (!os.write(reinterpret_cast<const char *>(&val), sizeof(T)))
        {
            throw std::runtime_error("error writing file: " + filename);
        }
    }

    os.close();

    std::cout << filename << " output done." << std::endl;


}

}

#endif