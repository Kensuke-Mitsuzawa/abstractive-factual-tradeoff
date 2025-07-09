def get_extractive_penalty_fct(penalty_command: str) -> str:
    dict_commnad2ep = dict(
        lambda4 = 'log_exp(2,4.804488)',  # lambda4
        lambda2 = 'log_exp(2,2.402244)',  # lambda2
        lambda1 = 'log_exp(2,1.201122)',  # lambda1
        none = 'none()',
        linear = 'linear()',
    )
    dict_commnad2ep['1/lambda2'] = 'log_exp(2,0.416277447)'  # 1/lambda2, log_exp(2, 1 / (1.20112 * 2))
    dict_commnad2ep['1/lambda1'] = 'log_exp(2,0.832556281)'  # 1/lambda1, log_exp(2, 1 / 1.20112)

    assert penalty_command in dict_commnad2ep

    return dict_commnad2ep[penalty_command]
