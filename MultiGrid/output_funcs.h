#ifndef __OUTPUT_FUNCS_H__
#define __OUTPUT_FUNCS_H__
    
#include<string>
#include<vector>
#include<cstdio>

#include "types.h"
#include "gmsh_output.h"
#include "dat_output.h"

namespace detail
{

class output_funcs : public monitor_funcs_t
{

public:
    output_funcs(std::string pos_vis_prefix,std::string dat_vis_prefix,int vis_freq, int max_vis_iters, grid_step_type grid_step) :
      pos_vis_prefix_(pos_vis_prefix),dat_vis_prefix_(dat_vis_prefix),vis_freq_(vis_freq),max_vis_iters_(max_vis_iters),grid_step_(grid_step)
    {
        detail::touch_file(pos_vis_prefix_ + "x" + ".pos");
        detail::touch_file(pos_vis_prefix_ + "r" + ".pos");
    }

    virtual void check_finished(int iters_performed, const vector_t& x, const vector_t& r)
    {
        if (iters_performed > max_vis_iters_) return;
        if (iters_performed%vis_freq_ != 0) return;
        if ((pos_vis_prefix_ == "none")&&(dat_vis_prefix_ == "none")) return;

        auto x_view = x.create_view(true);
        auto r_view = r.create_view(true);

        if (pos_vis_prefix_ != "none")
        {
            detail::append_view_pos_file_scal_3D_ord0(pos_vis_prefix_ + "x" + ".pos", iters_performed, grid_step_, x_view);
            detail::append_view_pos_file_scal_3D_ord0(pos_vis_prefix_ + "r" + ".pos", iters_performed, grid_step_, r_view);
        }
        if (dat_vis_prefix_ != "none")
        {
            detail::write_out_dat_file_scal_3D(dat_vis_prefix_ + "x" + "_iter_" + std::to_string(iters_performed) + ".dat", x_view);
            detail::write_out_dat_file_scal_3D(dat_vis_prefix_ + "r" + "_iter_" + std::to_string(iters_performed) + ".dat", r_view);
        }

        x_view.release(false);
        r_view.release(false);
    }
private:
    std::string    pos_vis_prefix_,dat_vis_prefix_;
    int            vis_freq_, max_vis_iters_;
    grid_step_type grid_step_;
};

}



#endif