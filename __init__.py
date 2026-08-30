# mlBridge package
#
# Lazy re-exports (PEP 562): importing mlBridge (or any submodule) must stay
# cheap. Heavy dependencies (sklearn, sqlalchemy, torch, ...) are only paid
# when a symbol from the owning module is first touched. This lets every
# consumer use canonical package imports (`from mlBridge import mlBridgeFFLib`)
# instead of putting the package directory itself on sys.path.

import importlib

# symbol -> owning submodule
_LAZY_EXPORTS = {
    # mlBridgeLib
    "pd_options_display": "mlBridgeLib",
    "Direction_to_NESW_d": "mlBridgeLib",
    "brs_to_pbn": "mlBridgeLib",
    "Vulnerability_to_Vul_d": "mlBridgeLib",
    "json_to_sql_walk": "mlBridgeLib",
    "CreateSqlFile": "mlBridgeLib",
    "NESW": "mlBridgeLib",
    "SHDC": "mlBridgeLib",
    "CDHS": "mlBridgeLib",
    "CDHSN": "mlBridgeLib",
    "NS_EW": "mlBridgeLib",
    "NSHDC": "mlBridgeLib",
    "HandsToBin": "mlBridgeLib",
    "BoardNumberToVul": "mlBridgeLib",
    "validate_brs": "mlBridgeLib",
    "seats": "mlBridgeLib",
    "vul_syms": "mlBridgeLib",
    "ranked_suit": "mlBridgeLib",
    "PlayerDirectionToPairDirection": "mlBridgeLib",
    "NextPosition": "mlBridgeLib",
    "PairDirectionToOpponentPairDirection": "mlBridgeLib",
    "declarer_direction_to_pair_direction": "mlBridgeLib",
    "BoardNumberToDealer": "mlBridgeLib",
    "ContractToScores": "mlBridgeLib",
    "DirectionToVul": "mlBridgeLib",
    "hands_to_brs": "mlBridgeLib",
    "hrs_to_brss": "mlBridgeLib",
    "pbn_to_hands": "mlBridgeLib",
    "score": "mlBridgeLib",
    "ContractTypeFromContract": "mlBridgeLib",
    "ContractType": "mlBridgeLib",
    "HandsToHCP": "mlBridgeLib",
    "HandsToQT": "mlBridgeLib",
    "HandsToSuitLengths": "mlBridgeLib",
    "HandsToDistributionPoints": "mlBridgeLib",
    "LoTT_SHDC": "mlBridgeLib",
    "CategorifyContractTypeBySuit": "mlBridgeLib",
    "CategorifyContractTypeByDirection": "mlBridgeLib",
    "MatchPointScoreUpdate": "mlBridgeLib",
    "show_estimated_memory_usage": "mlBridgeLib",
    "CATEGORICAL_SCHEMAS": "mlBridgeLib",
    # mlBridgeAcblLib
    "get_club_results_from_acbl_number": "mlBridgeAcblLib",
    "get_tournament_sessions_from_acbl_number": "mlBridgeAcblLib",
    "get_tournament_session_results": "mlBridgeAcblLib",
    "get_club_results_details_data": "mlBridgeAcblLib",
    "create_club_dfs": "mlBridgeAcblLib",
    "merge_clean_augment_club_dfs": "mlBridgeAcblLib",
    "merge_clean_augment_tournament_dfs": "mlBridgeAcblLib",
    # mlBridgeAcblPostmortemLib
    "augment_postmortem_dataframe": "mlBridgeAcblPostmortemLib",
    "build_club_postmortem": "mlBridgeAcblPostmortemLib",
    "build_tournament_postmortem": "mlBridgeAcblPostmortemLib",
    "tournament_section_for_player": "mlBridgeAcblPostmortemLib",
    # logging_config
    "setup_logger": "logging_config",
    "get_logger": "logging_config",
    "log_print": "logging_config",
    "init_project_logging": "logging_config",
    "print_started": "logging_config",
    "print_ended": "logging_config",
}

# Pure-literal constants, defined here directly (these previously shadowed the
# same names imported from mlBridgeLib, so behavior is unchanged).

# List of all possible contract strings
contract_classes = [f"{level}{strain}{dbl}" for level in range(1,8) for strain in ['C','D','H','S','N'] for dbl in ['','X','XX']] + ['Pass']

# List of all possible strains
strain_classes = ['C', 'D', 'H', 'S', 'N']

# List of all possible bid levels
level_classes = list(range(1,8))

# List of all possible double states
dbl_classes = ['', 'X', 'XX']

# List of all possible directions
direction_classes = ['N', 'E', 'S', 'W']

__all__ = sorted(
    set(_LAZY_EXPORTS)
    | {"contract_classes", "strain_classes", "level_classes", "dbl_classes", "direction_classes"}
)


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    else:
        # Attribute-style submodule access (`mlBridge.mlBridgeLib.foo`) worked
        # under the old eager __init__; keep it working lazily.
        try:
            value = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            if e.name != f"{__name__}.{name}":
                raise  # submodule exists but its own dependency is missing
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
