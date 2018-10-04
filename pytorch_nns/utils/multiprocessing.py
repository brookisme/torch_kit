import itertools
from multiprocessing import Process, cpu_count
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


#
# CONFIG
#
MAX_POOL_PROCESSES=cpu_count()-1
MAX_THREADPOOL_PROCESSES=16



#
# METHODS
#
""" MAP METHODS

  Args:
    * map_function <function>: 
      a function to map over args list. the function should take a single argument.
      if multiple arguments are needed accept them as a single list or tuple
    * args_list <list>: the list of arguments to map over
    * max_process <int>: number of processes
      - for max_with_pool defaults to the number of cpus minus 1
      - for max_with_threadpool defaults to 16
      - map_sequential ignores this argument as its doesn't actually do 
        any multiprocesssing 

  Return:
    List of return values from map_function

  Notes:
    map_sequential does NOT multiprocess.  it can be used as a sequential drop-in 
    replacement for map_with_pool/threadpool.  this is useful for:
      - development 
      - debugging
      - benchmarking 

"""
def map_with_pool(map_function,args_list,max_processes=MAX_POOL_PROCESSES):
  pool=Pool(processes=min(len(args_list),max_processes))
  return _run_pool(pool,map_function,args_list)


def map_with_threadpool(map_function,args_list,max_processes=MAX_THREADPOOL_PROCESSES):
  pool=ThreadPool(processes=min(len(args_list),max_processes))
  return _run_pool(pool,map_function,args_list)


def map_sequential(map_function,args_list,print_args=False,noisy=False,**dummy_kwargs):
  if noisy:
    print('multiprocessing(test):')
  out=[]
  for i,args in enumerate(args_list):
      if noisy: 
        print('\t{}...'.format(i))
      if print_args:
        print('\t{}'.format(args))
      out.append(map_function(args))
  if noisy: 
    print('-'*25)
  return out





""" simple: vanilla multiprocessing
  Args:
    * function <function>: function. function can take multiple arguments 
    * args_list <list>: the list of argument lists
    * join <bool[True]>: join processes before return

  Return: 
    List of processes 
"""
def simple(function,args_list,join=True):
  procs=[]
  for args in args_list:
      proc=Process(
          target=function, 
          args=args)
      procs.append(proc)
      proc.start()
  if join:
    for proc in procs:
        proc.join()
  return procs




#
# INTERNAL METHODS
#
def _stop_pool(pool,success=True):
  pool.close()
  pool.join()
  return success


def _map_async(pool,map_func,objects):
  try:
    return pool.map_async(map_func,objects)
  except KeyboardInterrupt:
    print("Caught KeyboardInterrupt, terminating workers")
    pool.terminate()
    return False
  else:
    print("Failure")
    return _stop_pool(pool,False)


def _run_pool(pool,map_function,args_list):
  out=_map_async(pool,map_function,args_list)
  _stop_pool(pool)
  return out.get()


