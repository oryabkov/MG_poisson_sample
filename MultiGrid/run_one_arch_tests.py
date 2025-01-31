
#import numpy as np
#from matplotlib import pyplot as plt
import sys
#import json
#import GPUtil
#from datetime import datetime
import subprocess
import os
#from datetime import datetime
#import random
#import glob
#import time
import logging as lg

def run_arch(arch_name,arch_device_name):
    for solver in ['jacobi','gmres']:
        for prec in ['diag','mg']:
            for float_type in ['f','d']:
                for size in [64,128,256,512]:
                    lg.info("execution: solver = %s, prec = %s, size = %i", solver, prec, size)
                    try:
                        exec_ok = True
                        cmd_out = subprocess.check_output( ['./test_'+solver+'_'+arch_name+'_'+float_type+'.bin', prec, str(size), arch_device_name ] )
                        out_str = cmd_out.decode('utf-8')
                        # lg.info("execution: proc_num = %i, batch = %i, number_in_batch = %i, config_file_name = %s, outstr:\n%s", process_number, btch, it, config_file_name, out_str)
                        print(out_str)

                    #except subprocess.CalledProcessError as e:
                    except Exception as e:
                        exec_ok = False
                        #lg.error(e.output)
                        lg.error(e)
                        lg.error("execution: solver = %s, prec = %s, size = %i failed", solver, prec, size)
                    
                    if exec_ok:   
                        lg.info("execution: solver = %s, prec = %s, size = %i success", solver, prec, size)
                    else:
                        break


def main():
    
    lg.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=lg.INFO)

    if len(sys.argv) != 3:
        print("Usage:" + sys.argv[0] + " arch_name arch_device_name")  
        return

    with open('data/times.dat', 'w') as file_handle:
        file_handle.write("solver,prec,arch,float_type,size,time(ms),iters_n,reduction_rate\n")

    run_arch(sys.argv[1],sys.argv[2])

if __name__ == '__main__':
    main()