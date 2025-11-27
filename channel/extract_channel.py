def extract_cir(a):
    """
    Извлекает канальные матрицы для каждой пары антенн из массива CIR 'a'.
    
    Параметры:
    ----------
    a : np.ndarray
        Массив CIR с формой:
        [batch, num_rx_groups, num_rx_ant, num_tx_groups, num_tx_ant, num_paths, num_time_steps]
        
    Возвращает:
    -----------
    channel_dict : dict
        Словарь, где ключи — это названия каналов ('h11', 'h12', 'h21', 'h22' и т.д.),
        а значения — матрицы CIR для этой пары антенн.
        Размер каждой матрицы: [num_paths, num_time_steps]
    """
    # Определяем количество антенн из формы массива
    num_rx_ant = a.shape[2]
    num_tx_ant = a.shape[4]
    
    channel_dict = {}
    
    for r in range(num_rx_ant):
        for t in range(num_tx_ant):
            key = f"h{r+1}{t+1}"
            
            # Извлекаем матрицу: [num_paths, num_time_steps]
            # Фиксируем batch=0, rx_group=0, tx_group=0
            channel_matrix = a[0, 0, r, 0, t, :, :]
            channel_dict[key] = channel_matrix
            
    return channel_dict
