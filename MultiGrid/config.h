#ifndef __CONFIG_H__
#define __CONFIG_H__

#define SCFD_ARRAYS_ENABLE_INDEX_SHIFT

constexpr int dim =     3;
#ifndef USE_DOUBLE_PRECISION 
using scalar      = float;
#else
using scalar      = double;
#endif

#endif
