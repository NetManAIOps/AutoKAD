
epoch_range = [10, 1000]
lr_range = [1e-5, 1e-2]
win_len_range = [5, 200]

parameters = [
        {
            "name": "model",
            "type": "choice",
            "values": ["Donut", "LSTM", "HW"],
            "dependents": {
                "Donut": ["lr_Donut", "win_len_Donut", "h_dim_Donut", "z_dim_Donut", "batch_size_Donut", "epoch_cnt_Donut"],
                "LSTM": ["lr_LSTM", "win_len_LSTM", "z_dim_LSTM", "batch_size_LSTM", "epoch_cnt_LSTM"],
                "HW": ["trend", "seasonal", "damped_trend", "seasonal_periods", "initialization_method"]
            },
        },
        {
            'name': "lr_Donut",
            'type': "range",
            'bounds': lr_range,
        },
        {
            'name': "win_len_Donut",
            'type': "range",
            'bounds': win_len_range,
        },
        {
            'name': "h_dim_Donut",
            'type': "range",
            'bounds': [20, 1000],
        },
        {
            'name': "z_dim_Donut",
            'type': "range",
            'bounds': [4, 200],
        },
        {
            'name': "batch_size_Donut",
            'type': "range",
            'bounds': [128, 25600],
        },
        {
            'name': "epoch_cnt_Donut",
            'type': "range",
            'bounds': epoch_range,
        },
        {
            'name': "lr_LSTM",
            'type': "range",
            'bounds': lr_range,
        },
        {
            'name': "win_len_LSTM",
            'type': "range",
            'bounds': win_len_range,
        },
        {
            'name': "z_dim_LSTM",
            'type': "range",
            'bounds': [4, 200],
        },
        {
            'name': "batch_size_LSTM",
            'type': "range",
            'bounds': [128, 2560],
        },
        {
            'name': "epoch_cnt_LSTM",
            'type': "range",
            'bounds': epoch_range,
        },
        {
            'name': "trend",
            'type': "choice",
            'values': ["add", "mul", "additive", "multiplicative", 'None'],

        },
        {
            'name': "seasonal",
            'type': "choice",
            'values': ["add", "mul", "additive", "multiplicative", 'None']

        },
        {
            'name':"damped_trend",
            'type': "choice",
            'values': [True, False]

        },
        {
            'name': "seasonal_periods",
            'type': "choice",
            'values': [60, 3600, 7200, 10800, 25200]

        },
        {
            'name': "initialization_method",
            'type': "choice",
            'values': ['None', 'estimated', 'heuristic', 'legacy-heuristic', 'known']
        },

]