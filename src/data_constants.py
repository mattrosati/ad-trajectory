# Constants for the data processing pipeline
# TODO: consider moving these to a config file/yaml file

# unit conversion ratios
unit_conversion = {
    "VSHEIGHT": 0.0254,  # inches to cm
    "VSWEIGHT": 0.453592,  # pounds to kg
}

raw_data_drop_columns = [
    "SITE",
    "FDG",
    "AV45",
    "FBB",  # no idea what field this is, not described in the data dictionary
    "PTAU",
    "CDRSB",
    "ADASQ4",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "LDELTOTAL",
    "DIGITSCOR",
    "TRABSCOR",
    "FAQ",
    "EcogPtMem",
    "EcogPtLang",
    "EcogPtVisspat",
    "EcogPtPlan",
    "EcogPtOrgan",
    "EcogPtDivatt",
    "EcogPtTotal",
    "EcogSPMem",
    "EcogSPLang",
    "EcogSPVisspat",
    "EcogSPPlan",
    "EcogSPOrgan",
    "EcogSPDivatt",
    "EcogSPTotal",
    "FLDSTRENG",
    "FSVERSION",
    "IMAGEUID",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
    "mPACCdigit",
    "mPACCtrailsB",
    "CDRSB_bl",
    "ADASQ4_bl",
    "RAVLT_immediate_bl",
    "RAVLT_learning_bl",
    "RAVLT_forgetting_bl",
    "RAVLT_perc_forgetting_bl",
    "LDELTOTAL_BL",
    "DIGITSCOR_bl",
    "TRABSCOR_bl",
    "FAQ_bl",
    "mPACCdigit_bl",
    "mPACCtrailsB_bl",
    "FLDSTRENG_bl",
    "FSVERSION_bl",
    "IMAGEUID_bl",
    "WholeBrain_bl",
    "Entorhinal_bl",
    "Fusiform_bl",
    "MidTemp_bl",
    "ICV_bl",
    "EcogPtMem_bl",
    "EcogPtLang_bl",
    "EcogPtVisspat_bl",
    "EcogPtPlan_bl",
    "EcogPtOrgan_bl",
    "EcogPtDivatt_bl",
    "EcogPtTotal_bl",
    "EcogSPMem_bl",
    "EcogSPLang_bl",
    "EcogSPVisspat_bl",
    "EcogSPPlan_bl",
    "EcogSPOrgan_bl",
    "EcogSPDivatt_bl",
    "EcogSPTotal_bl",
    "PTAU_bl",
    "FDG_bl",
    "AV45_bl",
    "FBB_bl",
    "update_stamp",
    "M",
    "Month",
    "PIB",
    "PIB_bl",
]

vitals_drop_columns = [
    "ID",
    "SITEID",
    "USERDATE",
    "USERDATE2",
    "DD_CRF_VERSION_LABEL",
    "LANGUAGE_CODE",
    "HAS_QC_ERROR",
    "update_stamp",
    "VISCODE",
    "VSHGTSC",
    "VSHEIGHT",
    "VSHTUNIT",
    "RID",
    "PHASE",
]

dtypes = {
    "RID": "string",
    "COLPROT": "string",
    "APOE4": "string",
    "ORIGPROT": "string",
    "PTID": "string",
    "VISCODE": "string",
    "VISCODE2": "string",
    "DX_BL": "string",
    "PTGENDER": "string",
    "PTEDUCAT": "float64",
    "PTETHCAT": "string",
    "PTRACCAT": "string",
    "PTMARRY": "string",
    "ABETA": "string",  #'>1700' giving problems
    "TAU": "string",  #'>1300' giving problems
    "PTAU": "string",  #'>120' giving problems
    "FSVERSION": "string",
    "DX": "string",
    "DX_bl": "string",
    "FLDSTRENG_bl": "string",
    "FSVERSION_bl": "string",
    "ABETA_bl": "string",  #'>1700' giving problems
    "TAU_bl": "string",  #'>1300' giving problems
    "PTAU_bl": "string",  #'>120' giving problems,
    "PHASE": "string",
    "VSWTUNIT": "string",
    "VSHTUNIT": "string",
    "VSTMPUNT": "string",
    "VSTMPSRC": "string",
}

fields_to_transform = [
    "ABETA",
    "TAU",
    "ADAS11",
    "ADAS13",
    "MMSE",
    "MOCA",
    "Ventricles",
    "ADAS11_bl",
    "ADAS13_bl",
    "MMSE_bl",
    "MOCA_bl",
    "Ventricles_bl",
    "ABETA_bl",
    "TAU_bl",
]

targets = ["MMSE", "MOCA", "ADAS11", "ADAS13", "Ventricles", "DX"]

id_columns = [
    "RID",
    "VISCODE",
    "COLPROT",
    "ORIGPROT",
    "PTID",
]

final_drop_columns = ["VISCODE2", "VISDATE"]

categorical_features = [
    "DX_bl",
    "PTGENDER",
    "PTETHCAT",
    "PTRACCAT",
    "PTMARRY",
    "DX",
    "APOE4",
    "VSWTUNIT",
    "VSTMPSRC",
    "VSTMPUNT",
]
