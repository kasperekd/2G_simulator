# import numpy as np

# rng = np.random.default_rng(42)

# rng_root1, rng_root2 = rng.spawn(2)
# los_phase1 = rng_root1.uniform(0, 2*np.pi)
# los_phase2 = rng_root2.uniform(0, 2*np.pi)
# print(f"1) {los_phase1}\n2) {los_phase2}")

# rng3, rng4 = rng_root1.spawn(2)
# los_phase3 = rng3.uniform(0, 2*np.pi)
# los_phase4 = rng4.uniform(0, 2*np.pi)
# print(f"3) {los_phase3}\n4) {los_phase4}")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

def run_simulation_step(child_rng, proc_id):
    # Внутри процесса мы используем ТОЛЬКО переданный child_rng
    # Порождаем еще два уровня для внутренних нужд процесса
    rng_sub1, rng_sub2 = child_rng.spawn(2)
    
    val1 = rng_sub1.uniform(0, 2*np.pi)
    val2 = rng_sub2.uniform(0, 2*np.pi)
    
    return f"Proc {proc_id}: {val1:.5f} | {val2:.5f}"

if __name__ == "__main__":
    root_seed = 42
    root_rng = np.random.default_rng(root_seed)
    
    num_processes = 4
    # Генерируем 4 независимых RNG-потомка для каждого процесса
    process_rngs = root_rng.spawn(num_processes)
    
    print(f"Запуск {num_processes} параллельных реализаций...\n")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Передаем каждому процессу его персональный RNG
        futures = [
            executor.submit(run_simulation_step, process_rngs[i], i) 
            for i in range(num_processes)
        ]
        
        for f in futures:
            print(f.result())
