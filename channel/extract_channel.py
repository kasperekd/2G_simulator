def extract_cir(H_list, num_rx_ant=2, num_tx_ant=2):
    """
    Разбивает список CIR на каналы h11, h12, h21, h22.

    Parameters
    ----------
    H_list : list[np.ndarray]

        Каждый элемент:
        H.shape = [num_time_steps, num_paths]

    num_rx_ant : int
    num_tx_ant : int

    Returns
    -------
    channel_dict : dict

        {
            'h11': [num_paths, num_time_steps],
            'h12': [num_paths, num_time_steps],
            'h21': [num_paths, num_time_steps],
            'h22': [num_paths, num_time_steps]
        }
    """

    if len(H_list) != num_rx_ant * num_tx_ant:
        raise ValueError(
            "Количество CIR не соответствует числу каналов"
        )

    channel_dict = {}
    idx = 0

    for r in range(num_rx_ant):
        for t in range(num_tx_ant):

            key = f"h{r+1}{t+1}"

            # транспонируем как в исходной функции
            channel_dict[key] = H_list[idx].T

            idx += 1

    return channel_dict