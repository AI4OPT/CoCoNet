"""Dataset registry: one entry per real dataset kept from the original learnscore repo.

Each `DatasetSpec` records what its loader (see loaders.py) needs to build (X, Y) from the
files under data/<name>/ (or a shared directory for datasets that are nested views of one
source table). `kind` selects which function in loaders.py handles it -- several datasets
share a kind because they are the same source format (e.g. every Mulan/OpenML multi-target
dataset uses "openml_csv_multi").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetSpec:
    name: str
    output_dim: int
    kind: str
    # Relative paths under the repo's data/ directory. Most datasets read one file named
    # after themselves (data/<name>/<file>); a few (argoverse2_d2/d4, qm9_d*, w300_d*) point
    # at a shared file outside their own dataset directory to avoid copying a big CSV N times.
    files: dict[str, str] = field(default_factory=dict)
    target_column: str | None = None
    target_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    log1p_target: bool = False
    # kind-specific odds and ends that don't deserve their own dataclass field (e.g. naval
    # propulsion's column indices, qm8's eigenvalue count).
    extra: dict = field(default_factory=dict)


REGISTRY: dict[str, DatasetSpec] = {

    # ---------------------------------------------------------------- d = 1 ----------
    "concrete": DatasetSpec(
        name="concrete", output_dim=1, kind="uci_xls",
        files={"xls": "concrete/Concrete_Data.xls"},
        target_column="Concrete compressive strength(MPa, megapascals) ",
    ),
    "kin8nm": DatasetSpec(
        name="kin8nm", output_dim=1, kind="flat_csv_single",
        files={"csv": "kin8nm/kin8nm.csv"},
        target_column="y",
    ),
    "naval_propulsion": DatasetSpec(
        name="naval_propulsion", output_dim=1, kind="naval_propulsion",
        files={"txt": "naval_propulsion/naval_propulsion.txt"},
        # GT Turbine decay state coefficient (index 17 of 18 whitespace columns); see
        # loaders.py for the full column layout comment.
        extra={"target_column_index": 17, "feature_column_indices": list(range(16))},
    ),
    "protein": DatasetSpec(
        name="protein", output_dim=1, kind="flat_csv_single",
        files={"csv": "protein/CASP.csv"},
        target_column="RMSD",
    ),

    # ---------------------------------------------------------------- d = 2 ----------
    "enb": DatasetSpec(
        name="enb", output_dim=2, kind="flat_csv_multi",
        files={"csv": "enb/enb.csv"},
        target_columns=["Y1", "Y2"],
    ),
    "geographical_music": DatasetSpec(
        name="geographical_music", output_dim=2, kind="geographical_music",
        files={"txt": "geographical_music/default_features_1059_tracks.txt"},
        target_columns=["latitude", "longitude"],
    ),
    "argoverse2_motion_forecasting": DatasetSpec(
        name="argoverse2_motion_forecasting", output_dim=2, kind="argoverse2",
        files={"csv": "argoverse2_motion_forecasting/argoverse2_motion_forecasting.csv"},
        target_columns=["future_dx", "future_dy"],
        drop_columns=["scenario_id"],
        categorical_columns=["object_type", "city"],
    ),
    "argoverse2_d2": DatasetSpec(
        name="argoverse2_d2", output_dim=2, kind="argoverse2",
        files={"csv": "argoverse2_multidim/argoverse2_multidim.csv"},
        target_columns=["future_dx_s60", "future_dy_s60"],
        drop_columns=["scenario_id"],
        categorical_columns=["object_type", "city"],
    ),
    "taxi": DatasetSpec(
        name="taxi", output_dim=2, kind="taxi",
        files={"csv": "taxi/taxi.csv"},
        extra={"expected_rows": 61286, "expected_x_dim": 6},
    ),
    "casp": DatasetSpec(
        # NOT a duplicate of `protein`: this is the MVCS authors' own bivariate packaging of
        # the same underlying CASP data (2 targets instead of 1, their own preprocessing).
        name="casp", output_dim=2, kind="npz_xy",
        files={"npz": "casp/CASP.npz"},
    ),

    # ---------------------------------------------------------------- d = 4 ----------
    "argoverse2_d4": DatasetSpec(
        name="argoverse2_d4", output_dim=4, kind="argoverse2",
        files={"csv": "argoverse2_multidim/argoverse2_multidim.csv"},
        target_columns=["future_dx_s30", "future_dy_s30", "future_dx_s60", "future_dy_s60"],
        drop_columns=["scenario_id"],
        categorical_columns=["object_type", "city"],
    ),
    "sgemm": DatasetSpec(
        name="sgemm", output_dim=4, kind="flat_csv_multi",
        files={"csv": "sgemm/sgemm_product.csv"},
        target_columns=["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"],
        categorical_columns=["STRM", "STRN", "SA", "SB"],
    ),
    "qm8_excited_states": DatasetSpec(
        name="qm8_excited_states", output_dim=4, kind="qm8",
        files={
            "spectra": "qm8_excited_states/qm8_gdb8_22k_elec_spec.txt",
            "xyz": "qm8_excited_states/qm8_XYZ_B3LYP_631G2dfp.xyz",
        },
        target_columns=["CC2_E1", "CC2_E2", "CC2_f1", "CC2_f2"],
        extra={
            "all_target_columns": [
                "CC2_E1", "CC2_E2", "CC2_f1", "CC2_f2",
                "PBE0svp_E1", "PBE0svp_E2", "PBE0svp_f1", "PBE0svp_f2",
                "PBE0tzvp_E1", "PBE0tzvp_E2", "PBE0tzvp_f1", "PBE0tzvp_f2",
                "CAM_E1", "CAM_E2", "CAM_f1", "CAM_f2",
            ],
            "n_eigenvalues": 8,
        },
    ),

    # ---------------------------------------------------------------- d = 3 ----------
    "sf1": DatasetSpec(
        name="sf1", output_dim=3, kind="flat_csv_multi",
        files={"csv": "sf1/sf1.csv"},
        target_columns=["c_class", "m_class", "x_class"],
        categorical_columns=["mod_zurich_class", "largest_spot_size", "spot_distribution",
                              "activity", "evolution", "previous_day_activity", "hist_complex",
                              "become_hist_complex", "area", "area_largest"],
    ),
    "sf2": DatasetSpec(
        name="sf2", output_dim=3, kind="flat_csv_multi",
        files={"csv": "sf2/sf2.csv"},
        target_columns=["c_class", "m_class", "x_class"],
        categorical_columns=["mod_zurich_class", "largest_spot_size", "spot_distribution",
                              "activity", "evolution", "previous_day_activity", "hist_complex",
                              "become_hist_complex", "area", "area_largest"],
    ),
    "scpf": DatasetSpec(
        name="scpf", output_dim=3, kind="flat_csv_multi",
        files={"csv": "scpf/scpf.csv"},
        target_columns=["num_views", "num_votes", "num_comments"],
    ),

    # ---------------------------------------------------------------- d = 6 ----------
    "atp1d": DatasetSpec(
        name="atp1d", output_dim=6, kind="flat_csv_multi",
        files={"csv": "atp1d/atp1d.csv"},
        target_columns=["LBL_ALLminpA_fut_001", "LBL_ALLminp0_fut_001", "LBL_aDLminpA_fut_001",
                         "LBL_aCOminpA_fut_001", "LBL_aFLminpA_fut_001", "LBL_aUAminpA_fut_001"],
    ),
    "communities_crime_unnormalized": DatasetSpec(
        name="communities_crime_unnormalized", output_dim=6, kind="communities_crime_unnormalized",
        files={"txt": "communities_crime_unnormalized/CommViolPredUnnormalizedData.txt"},
        target_columns=["murdPerPop", "rapesPerPop", "robbbPerPop", "assaultPerPop",
                         "burglPerPop", "larcPerPop"],
        drop_columns=["communityname", "State", "countyCode", "communityCode", "fold",
                      "murders", "rapes", "robberies", "assaults", "burglaries", "larcenies",
                      "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop",
                      "violentPerPop", "nonViolPerPop"],
    ),

    # ---------------------------------------------------------------- d = 7 ----------
    "sarcos": DatasetSpec(
        name="sarcos", output_dim=7, kind="sarcos",
        files={"train": "sarcos/sarcos_inv.mat", "test": "sarcos/sarcos_inv_test.mat"},
        extra={"n_input_columns": 21, "target_indices": [21, 22, 23, 24, 25, 26, 27]},
    ),

    # ---------------------------------------------------------------- d = 8 ----------
    "rf1": DatasetSpec(
        name="rf1", output_dim=8, kind="flat_csv_multi",
        files={"csv": "rf1/rf1.csv"},
        target_columns=["CHSI2_48H__0", "NASI2_48H__0", "EADM7_48H__0", "SCLM7_48H__0",
                         "CLKM7_48H__0", "VALI2_48H__0", "NAPM7_48H__0", "DLDI4_48H__0"],
    ),
    "rf2": DatasetSpec(
        name="rf2", output_dim=8, kind="flat_csv_multi",
        files={"csv": "rf2/rf2.csv"},
        target_columns=["CHSI2_48H__0", "NASI2_48H__0", "EADM7_48H__0", "SCLM7_48H__0",
                         "CLKM7_48H__0", "VALI2_48H__0", "NAPM7_48H__0", "DLDI4_48H__0"],
    ),

    # ---------------------------------------------------------------- d = 12 ---------
    "osales": DatasetSpec(
        name="osales", output_dim=12, kind="flat_csv_multi",
        files={"csv": "osales/osales.csv"},
        target_columns=[f"Outcome_M{i}" for i in range(1, 13)],
    ),

    # ---------------------------------------------------------------- d = 14 ---------
    "wq": DatasetSpec(
        name="wq", output_dim=14, kind="flat_csv_multi",
        files={"csv": "wq/wq.csv"},
        target_columns=["x25400", "x29600", "x30400", "x33400", "x17300", "x19400", "x34500",
                         "x38100", "x49700", "x50390", "x55800", "x57500", "x59300", "x37880"],
    ),

    # ---------------------------------------------------------------- d = 16 ---------
    "scm1d": DatasetSpec(
        name="scm1d", output_dim=16, kind="flat_csv_multi",
        files={"csv": "scm1d/scm1d.csv"},
        target_columns=["LBL", "MTLp2", "MTLp3", "MTLp4", "MTLp5", "MTLp6", "MTLp7", "MTLp8",
                         "MTLp9", "MTLp10", "MTLp11", "MTLp12", "MTLp13", "MTLp14", "MTLp15",
                         "MTLp16"],
    ),
    "scm20d": DatasetSpec(
        name="scm20d", output_dim=16, kind="flat_csv_multi",
        files={"csv": "scm20d/scm20d.csv"},
        target_columns=["LBL", "MTLp2A", "MTLp3A", "MTLp4A", "MTLp5A", "MTLp6A", "MTLp7A",
                         "MTLp8A", "MTLp9A", "MTLp10A", "MTLp11A", "MTLp12A", "MTLp13A",
                         "MTLp14A", "MTLp15A", "MTLp16A"],
    ),
    "oes10": DatasetSpec(
        name="oes10", output_dim=16, kind="oes_arff",
        files={"arff": "oes10/oes10.arff"},
        target_columns=[
            "x513021_Butchers_and_Meat_Cutters", "x292071_Medical_Records_and_Health_Information_Technicians",
            "x392021_Nonfarm_Animal_Caretakers", "x151131_Computer_Programmers", "x151141_Database_Administrators",
            "x291069_Physicians_and_Surgeons__All_Other", "x119032_Education_Administrators__Elementary_and_Secondary_School",
            "x432011_Switchboard_Operators__Including_Answering_Service", "x419022_Real_Estate_Sales_Agents",
            "x292037_Radiologic_Technologists_and_Technicians_", "x519061_Inspectors__Testers__Sorters__Samplers__and_Weighers",
            "x291051_Pharmacists", "x172141_Mechanical_Engineers",
            "x431011_First_Line_Supervisors_of_Office_and_Administrative_Support_Workers",
            "x291127_Speech_Language_Pathologists", "x412021_Counter_and_Rental_Clerks",
        ],
    ),
    "oes97": DatasetSpec(
        name="oes97", output_dim=16, kind="oes_arff",
        files={"arff": "oes97/oes97.arff"},
        target_columns=[
            "x58028_Shipping__Receiving__and_Traffic_Clerks", "x15014_Industrial_Production_Managers",
            "x32511_Physician_Assistants", "x15017_Construction_Managers", "x98502_Machine_Feeders_and_Offbearers",
            "x92965_Crushing__Grinding__Mixing__and_Blending_Machine_Operators_and_Tenders",
            "x32314_Speech_Language_Pathologists_and_Audiologists", "x13008_Purchasing_Managers",
            "x21114_Accountants_and_Auditors", "x85110_Machinery_Maintenance_Mechanics",
            "x27311_Recreation_Workers", "x98902_Hand_Packers_and_Packagers", "x65032_Cooks__Fast_Food",
            "x92998_All_Other_Machine_Operators_and_Tenders", "x27108_Psychologists",
            "x53905_Teacher_Aides_and_Educational_Assistants__Clerical",
        ],
    ),

    # ------------------------------------------------- caveated (kept per instructions) -----
    "bike_sharing": DatasetSpec(
        name="bike_sharing", output_dim=1, kind="flat_csv_single",
        files={"csv": "bike_sharing/hour.csv"},
        target_column="cnt",
        drop_columns=["instant", "casual", "registered", "dteday"],
        categorical_columns=["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                              "weathersit"],
    ),
    "bias_correction_temperature": DatasetSpec(
        name="bias_correction_temperature", output_dim=2, kind="bias_correction",
        files={"csv": "bias_correction_temperature/Bias_correction_ucl.csv"},
        target_columns=["Next_Tmax", "Next_Tmin"],
        categorical_columns=["station"],
        extra={"date_column": "Date"},
    ),
    "online_retail_ii": DatasetSpec(
        name="online_retail_ii", output_dim=2, kind="online_retail_ii",
        files={"xlsx": "online_retail_ii/online_retail_II.xlsx"},
        target_columns=["Quantity", "Price"],
        drop_columns=["Invoice", "StockCode", "Description", "Customer ID", "InvoiceDate"],
        categorical_columns=["Country"],
        extra={"sheet_names": ["Year 2009-2010", "Year 2010-2011"]},
    ),
    "household_power_consumption": DatasetSpec(
        name="household_power_consumption", output_dim=3, kind="household_power",
        files={"gz": "household_power_consumption/household_power_consumption.txt.gz"},
        target_columns=["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        drop_columns=["Date", "Time"],
    ),

    # ------------------------------------------------- QM9 nested d = 2/4/6/8 ---------------
    "qm9_d2": DatasetSpec(
        name="qm9_d2", output_dim=2, kind="qm9",
        files={"csv": "qm9/qm9_tabular.csv"}, extra={"target_dim": 2},
    ),
    "qm9_d4": DatasetSpec(
        name="qm9_d4", output_dim=4, kind="qm9",
        files={"csv": "qm9/qm9_tabular.csv"}, extra={"target_dim": 4},
    ),
    "qm9_d6": DatasetSpec(
        name="qm9_d6", output_dim=6, kind="qm9",
        files={"csv": "qm9/qm9_tabular.csv"}, extra={"target_dim": 6},
    ),
    "qm9_d8": DatasetSpec(
        name="qm9_d8", output_dim=8, kind="qm9",
        files={"csv": "qm9/qm9_tabular.csv"}, extra={"target_dim": 8},
    ),

    # ------------------------------------------------- 300-W nested d = 2/4/8/16 ------------
    "w300_d2": DatasetSpec(
        name="w300_d2", output_dim=2, kind="w300",
        files={"csv": "300w/w300_tabular.csv"}, extra={"target_dim": 2},
    ),
    "w300_d4": DatasetSpec(
        name="w300_d4", output_dim=4, kind="w300",
        files={"csv": "300w/w300_tabular.csv"}, extra={"target_dim": 4},
    ),
    "w300_d8": DatasetSpec(
        name="w300_d8", output_dim=8, kind="w300",
        files={"csv": "300w/w300_tabular.csv"}, extra={"target_dim": 8},
    ),
    "w300_d16": DatasetSpec(
        name="w300_d16", output_dim=16, kind="w300",
        files={"csv": "300w/w300_tabular.csv"}, extra={"target_dim": 16},
    ),
}

# QM9's fixed canonical target ordering; qm9_dN takes the first N/... entries. mu/alpha/homo/
# lumo/r2/zpve/Cv/U0 are the 8 physically distinct quantities the kept variants (d<=8) span;
# gap/U/H/G (near-redundant with the first eight) are excluded from this repo's kept set.
QM9_CANONICAL_ORDER = ["mu", "alpha", "homo", "lumo", "r2", "zpve", "Cv", "U0"]

# 300-W's fixed canonical landmark ordering (nose tip, chin, eye corners, ...); w300_dN takes
# the first N/2 landmarks' (x, y).
W300_LANDMARK_RANKS = 8  # d16 needs 8 landmarks x 2 coords


def get(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Known datasets: {sorted(REGISTRY)}")
    return REGISTRY[name]
