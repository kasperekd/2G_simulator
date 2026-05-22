from multiprocessing import Pool, cpu_count
from core.iterations import single_burst_iteration
import itertools

def calculate_ber(pool, base_args, num_bursts, target_ratio_db):
    def run_batch(count):
        if count <= 0:
            return 0, 0, 0.0, 0.0, 0.0
        
        chunk_size = max(1, count // (cpu_count() * 4))
        
        results = pool.map(single_burst_iteration, itertools.repeat(base_args, count), chunksize=chunk_size)
        
        batch_errors = sum(r[0] for r in results)
        batch_bits = sum(r[1] for r in results)
        
        batch_sum_mse_ls = sum(r[2] for r in results)
        batch_sum_mse_lmmse = sum(r[3] for r in results)
        
        batch_sum_measured_sinr_total = sum(r[4] for r in results)

        batch_sum_target_sinr_total = sum(r[5] for r in results)

        return batch_errors, batch_bits, batch_sum_mse_ls, batch_sum_mse_lmmse, batch_sum_measured_sinr_total, batch_sum_target_sinr_total

    # 1. Pilot run
    test_iterations = 25
    total_errors, total_bits, total_mse_ls, total_mse_lmmse, measured_sinr_total, target_sinr_total = run_batch(test_iterations)
    
    current_ber = total_errors / total_bits if total_bits > 0 else 0.5

    # 2. Determine target bursts
    target_bursts = num_bursts
    
    # User logic: if BER < 10e-5, increase the number of bursts
    if current_ber < 10e-5:
        target_bursts = num_bursts + 500

    # 3. Main run
    remaining_bursts = target_bursts - test_iterations

    if remaining_bursts > 0:
        add_err, add_bits, add_mse_ls, add_mse_lmmse, add_measured_sinr_total, add_target_sinr_total = run_batch(remaining_bursts)
        total_errors += add_err
        total_bits += add_bits
        total_mse_ls += add_mse_ls
        total_mse_lmmse += add_mse_lmmse
        measured_sinr_total += add_measured_sinr_total
        target_sinr_total += add_target_sinr_total


    final_ber = total_errors / total_bits if total_bits > 0 else 0.5
    
    final_avg_mse_ls = total_mse_ls / target_bursts
    final_avg_mse_lmmse = total_mse_lmmse / target_bursts

    final_avg_measured_sinr_total = measured_sinr_total / target_bursts
    final_avg_target_sinr_total = target_sinr_total / target_bursts

    return final_ber, final_avg_mse_ls, final_avg_mse_lmmse, final_avg_measured_sinr_total, final_avg_target_sinr_total