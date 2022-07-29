#!/usr/bin/env python
import os

for i in range(1,6):
    print('python brf_16cl_v2.py' + str(i) + ' &> log_brf' + str(i) + '_261120.txt')
    print('python ada_16cl_v2.py' + str(i) + ' &> log_ada' + str(i) + '_261120.txt')
    
