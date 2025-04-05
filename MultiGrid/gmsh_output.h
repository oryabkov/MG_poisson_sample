#ifndef __GMSH_OUTPUT_H__
#define __GMSH_OUTPUT_H__
    
#include<string>
#include<vector>
#include<cstdio>

namespace detail
{

void touch_file(const std::string filename)
{
    FILE *stream = std::fopen( filename.c_str(), "w" );
    if(stream == NULL)
    {
        throw std::runtime_error("error creating file: " + filename);
    }
    fclose(stream);
}

template<class Vec, class Array>
void append_view_pos_file_scal_3D_ord0(const std::string filename, int iter, Vec grid_step, const Array& U)
{

    using T = typename Array::value_type;

    FILE *stream=std::fopen(filename.c_str(), "a" );
    if(stream == NULL)
    {
        throw std::runtime_error("error opening file: " + filename);
    }

    fprintf( stream, "View");
    fprintf( stream, " '");
    fprintf( stream, "%s %i", filename.c_str(), iter);
    fprintf( stream, "' {\n");
    fprintf( stream, "TIME{0};\n");

    auto r = U.rect_nd();

    for (auto idx : r)
    {
        Vec p0,p1;
        for (int j = 0;j < Vec::dim;++j)
        {
            p0[j] = grid_step[j]*idx[j];
            p1[j] = grid_step[j]*(idx[j]+1);
        }
        auto xm = p0[0];
        auto xp = p1[0];
        auto ym = p0[1];
        auto yp = p1[1];
        auto zm = p0[2];
        auto zp = p1[2];

        T par_x_mmm=0.0;
        T par_x_pmm=0.0;
        T par_x_ppm=0.0;
        T par_x_ppp=0.0;
        T par_x_mpp=0.0;
        T par_x_mmp=0.0;
        T par_x_pmp=0.0;
        T par_x_mpm=0.0;


        par_x_mmm=U(idx);
        par_x_pmm=U(idx);
        par_x_ppm=U(idx);
        par_x_ppp=U(idx);
        par_x_mpp=U(idx);
        par_x_mmp=U(idx);
        par_x_pmp=U(idx);
        par_x_mpm=U(idx);
                    



        fprintf( stream, "SH(%e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e)",
        xm, ym, zm,  
        xp, ym, zm,  
        xp, yp, zm,  
        xm, yp, zm, 
        xm, ym, zp, 
        xp, ym, zp, 
        xp, yp, zp,
        xm, yp, zp);


        fprintf( stream,"{");
        fprintf(stream, "%e,",par_x_mmm);
        fprintf(stream, "%e,",par_x_pmm);
        fprintf(stream, "%e,",par_x_ppm);
        fprintf(stream, "%e,",par_x_mpm);
        fprintf(stream, "%e,",par_x_mmp);
        fprintf(stream, "%e,",par_x_pmp);
        fprintf(stream, "%e,",par_x_ppp);
        fprintf(stream, "%e",par_x_mpp);
        fprintf(stream, "};\n");
    }

    fflush(stream);  //flush all to disk

    fprintf( stream, "};");

    fclose( stream );

    std::cout << filename << " output done." << std::endl;


}

template<class Vec, class Array>
void write_out_pos_file_scal_3D_ord0(const std::string filename, Vec grid_step, const Array& U)
{
    touch_file(filename);

    append_view_pos_file_scal_3D_ord0(filename, 0, grid_step, U);
}

}



#endif