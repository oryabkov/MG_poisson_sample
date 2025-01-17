#ifndef __NMFD_RESIDUAL_REGULATIZATION_TEST_H__
#define __NMFD_RESIDUAL_REGULATIZATION_TEST_H__


namespace nmfd {
namespace solvers {
namespace detail {

template<class VectorOperations, class Log>
class residual_regularization_test
{
public:
    using scalar_type =  typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;    

    residual_regularization_test(std::shared_ptr<VectorOperations> vec_ops, Log *log = nullptr):
    vec_ops_(vec_ops),
    log_(log)
    {}
    ~residual_regularization_test()
    {}
    
    template<class VecX>
    void apply(VecX &x) const
    {
        auto res = vec_ops_->sum(x);
        if(log_!=nullptr)
        {
            log_->info_f("residual_regularization_test: before: %e", res);
        }
        auto sz = vec_ops_->size();
        vec_ops_->add_mul_scalar(-res/sz, 1, x);
        if(log_!=nullptr)
        {
            res = vec_ops_->sum(x);
            log_->info_f("residual_regularization_test: after: %e", res);
        }
    }

private:
    std::shared_ptr<VectorOperations> vec_ops_;
    Log *log_;


};

}
}
}


#endif