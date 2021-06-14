from pathos.pools import ProcessPool


def run_experiments_parallel(runners, processes):
    processes = min(len(runners), processes)
    print(f"Multiprocessing of {len(runners)} runners with {processes} processes.")
    pool = ProcessPool(processes)
    pending_results = pool.amap(lambda runner: runner.run(), runners)
    results = pending_results.get()
    pool.close()
    pool.join()
    return results
