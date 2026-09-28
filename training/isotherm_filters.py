isotherm_filter_ooms = [
    'https://doi.org/10.1007/s11356-023-26390-x', # < 1 oom
    'https://doi.org/10.1016/j.marpolbul.2021.113086', # < 1 oom
    'https://doi.org/10.1016/j.envpol.2023.122573', # < 1 oom
    'https://doi.org/10.1007/s10653-024-01961-0', # < 1 oom
]
isotherm_linear_fit = [
    'https://doi.org/10.1016/j.envint.2022.107459',
    'https://doi.org/10.1016/j.ecoenv.2023.114533',
    'https://doi.org/10.1016/j.scitotenv.2021.145451',
    'https://doi.org/10.1016/j.scitotenv.2022.160786',
    'https://doi.org/10.1021/es071737s',
    'https://doi.org/10.1021/es405721v',
    'https://doi.org/10.1016/j.scitotenv.2019.07.176'
]


def has_enough_ooms(row):
    return row['doi'] not in isotherm_filter_ooms


def is_linear_fit(row):
    return row['doi'] in isotherm_linear_fit


def is_non_linear_fit(row):
    return not is_linear_fit(row)
