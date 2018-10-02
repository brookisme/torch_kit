import itertools
from multiprocessing import Process, cpu_count
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

MAX_PROCESSES=cpu_count()-1


def map_with_pool(data_load_func,jobs_list,max_processes=MAX_PROCESSES):
  pool=Pool(processes=min(len(jobs_list),max_processes))
  dfs=_run_procs(pool,data_load_func,jobs_list)
  _stop_pool(pool)
  return dfs.get()


def map_with_threadpool(data_load_func,jobs_list,max_processes=MAX_PROCESSES):
  pool=ThreadPool(processes=min(len(jobs_list),max_processes))
  dfs=_run_procs(pool,data_load_func,jobs_list)
  _stop_pool(pool)
  return dfs.get()


def _stop_pool(pool,success=True):
  pool.close()
  pool.join()
  return success


def _run_procs(pool,map_func,objects):
  try:
    return pool.map_async(map_func,objects)
  except KeyboardInterrupt:
    print("Caught KeyboardInterrupt, terminating workers")
    pool.terminate()
    return False
  else:
    print("Failure")
    return _stop_pool(pool,False)


def simple(target,args_list,join=True):
  procs=[]
  for args in args_list:
      proc=Process(
          target=target, 
          args=args)
      procs.append(proc)
      proc.start()
  if join:
    for proc in procs:
        proc.join()
  return procs


def map_sequential(target,args_list,print_args=False,noisy=False,**dummy_kwargs):
  if noisy:
    print('multiprocessing(test):')
  out=[]
  for i,args in enumerate(args_list):
      if noisy: 
        print('\t{}...'.format(i))
      if print_args:
        print('\t{}'.format(args))
      out.append(target(args))
  if noisy: 
    print('-'*25)
  return out

