from multiprocessing import Pool, cpu_count
from core.iterations import single_burst_iteration
import itertools

def calculate_ber(pool, base_args, num_bursts, target_ratio_db):
    def run_batch(count):
        if count <= 0:
            return 0, 0
        
        chunk_size = max(1, count // (cpu_count() * 4))
        
        results = pool.map(single_burst_iteration, itertools.repeat(base_args, count), chunksize=chunk_size)
        
        batch_errors = sum(r[0] for r in results)
        batch_bits = sum(r[1] for r in results)
        return batch_errors, batch_bits

    # 1. Pilot run
    test_iterations = 25
    total_errors, total_bits = run_batch(test_iterations)
    
    current_ber = total_errors / total_bits if total_bits > 0 else 0.5

    # 2. Determine how many more bursts to calculate
    target_bursts = num_bursts
    
    # User logic: if BER < 10e-5, increase the number of bursts
    if current_ber < 10e-5:
        # print(f"DEBUG: Low BER detected ({current_ber}), increasing precision.")
        plus_burst = 500
        target_bursts = num_bursts + plus_burst

    # 3. Calculate the remaining bursts (Main run)
    # We do not discard the test results but add the missing ones
    remaining_bursts = target_bursts - test_iterations
    
    if remaining_bursts > 0:
        add_errors, add_bits = run_batch(remaining_bursts)
        total_errors += add_errors
        total_bits += add_bits

    final_ber = total_errors / total_bits if total_bits > 0 else 0.5
    return final_ber