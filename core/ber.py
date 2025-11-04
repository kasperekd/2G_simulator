from multiprocessing import Pool, cpu_count
from core.iterations import single_burst_iteration

def calculate_ber(base_args, num_bursts, target_ratio_db):
    def run_pool(n_bursts):
        args_list = [base_args] * n_bursts
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(single_burst_iteration, args_list)
        total_errors = sum(r[0] for r in results)
        total_bits = sum(r[1] for r in results)
        return total_errors, total_bits
    # 1. caulculation ber with num_bursts=5
    test_iterations = 25 
    total_errors, total_bits = run_pool(test_iterations)
    ber = total_errors / total_bits if total_bits > 0 else 0.5

    # 2. If ber < 10e-5, bursts + plus_burst
    if ber < 10e-5:
        print(f"BER {ber} < 10e-5, increasing bursts + plus_burst for target_ratio_db={target_ratio_db}")
        plus_burst = 500
        total_errors, total_bits = run_pool(num_bursts + plus_burst)
        ber = total_errors / total_bits if total_bits > 0 else 0.5
    else:
        # 3. Иначе рассчитать BER с заданным num_bursts
        total_errors, total_bits = run_pool(num_bursts)
        ber = total_errors / total_bits if total_bits > 0 else 0.5
        

    return ber
