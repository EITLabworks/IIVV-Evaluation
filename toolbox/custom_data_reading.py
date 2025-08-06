import numpy as np
import pandas as pd
from . import load_data, Encoder, z_score, visualize_Category_hist


def custom_data_reading(
    Cutpats_threshold,
    normalization,
    test_size,  # unused here
    visualize_Category=False,
    prints=True,
    lesion_path="Full_Data_lesion_info.csv",
    data_path="data",
):
    if prints:
        print("\n1. read data")

    lesion_info = pd.read_csv(lesion_path)
    lesion_info.keys()

    LesionCase = lesion_info["Case"]
    LesionLoc = lesion_info[" LesionLoc"]
    LesionAssessment = lesion_info[" LesionAssessment"]

    load_config = {
        "py_idx": False,
        "Zdata": True,
        "patient_no": True,
        "patient_sample":True,
        "Bx_Case": True,
        "Case": True,
        "Avg_Cutpats": True,
        "Sample": True,
        "Category": True,
        "print_info": prints,
    }
    loaded = load_data(path="data", **load_config)

    Zdata = loaded["Zdata"]
    
    patient_no = loaded["patient_no"]
    patient_sample = loaded["patient_sample"]
    
    Case = loaded["Case"]
    Bx_Case = loaded["Bx_Case"]
    Avg_Cutpats = loaded["Avg_Cutpats"]
    Category = loaded["Category"]
    Sample = loaded["Sample"]

    #assert (Case == LesionCase).all()
    if prints:
        print(Zdata.shape, Avg_Cutpats.shape, Category.shape, Sample.shape)

    if prints:
        print(f"\n2. select only data where Avg_Cutpats <{Cutpats_threshold}%")
    dropout = np.where(Avg_Cutpats < Cutpats_threshold)[0]

    LesionCase = LesionCase[dropout]
    LesionLoc = LesionLoc[dropout]
    LesionAssessment = LesionAssessment[dropout]
    # Zdata = Zdata
    Zdata = np.abs(Zdata[dropout])  # get abs value
    
    patient_no = patient_no[dropout]
    patient_sample = patient_sample[dropout]
    
    Case = Case[dropout]
    print(Case)
    print(LesionCase)
    #assert (Case == LesionCase).all()
    Bx_Case = Bx_Case[dropout]
    Avg_Cutpats = Avg_Cutpats[dropout]
    Category = Category[dropout]
    Sample = Sample[dropout]
    if prints:
        print(Zdata.shape, Avg_Cutpats.shape, Category.shape, Sample.shape)

    if prints:
        print("\n3. Category -> only take [[0,1,1],[0,1,0]]")
    cat_select = list()
    for i, cat in enumerate(Category):
        if list(cat) == [0, 1, 1] or list(cat) == [0, 1, 0]:
            cat_select.append(i)
    cat_select = np.array(cat_select)

    Category = Category[cat_select]
    Bx_Case = Bx_Case[cat_select]
    
    patient_no = patient_no[cat_select]
    patient_sample = patient_sample[cat_select]

    assert Category.shape[0] == Bx_Case.shape[0]
    LesionAssessment = np.array(LesionAssessment)[cat_select]
    Zdata = Zdata[cat_select]

    if visualize_Category:
        visualize_Category_hist(Category)

    Bx_Case_encoder = Encoder(Bx_Case, print_info=True)
    patients_labels = np.array(
        Bx_Case_encoder.encoded
    )  # numbering of individual patient

    LA_enc = Encoder(LesionAssessment, print_info=True)
    healthy_labels = LA_enc.encoded
    if prints:
        print(
            "\n4. healthy is everything except 'Invasive carcinoma' -> True ... healthy, False ... cancer.\n"
        )
    # sel all mappings to 1 except "Invasive carcinoma" [True->healthy, False->cancer]
    healthy_labels = np.array(
        [1 if i != LA_enc.mapping["Invasive carcinoma"] else 0 for i in healthy_labels]
    )
    if prints:
        print(f"{Zdata.shape=}, {healthy_labels.shape=}, {patients_labels.shape=}")

    if prints:
        print("\nnormalize EIT measurements")
    X = z_score(
        Zdata, along=normalization, print_info=False
    )  # global / select single frequency...

    if prints:
        print(f"{X.shape=}, {healthy_labels.shape=}, {patients_labels.shape=}")

    return X, Case, healthy_labels, patients_labels, Zdata, patient_no, patient_sample, Bx_Case
