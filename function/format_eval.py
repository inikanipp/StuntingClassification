import pandas as pd
import streamlit as st

def format_eval(report) :

    # list untuk masing-masing kolom
    precision = []
    recall = []
    f1_score = []
    support = []
    index = []

    # ambil kelas 0,1,...
    for key in report:
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            precision.append(report[key]['precision'])
            recall.append(report[key]['recall'])
            f1_score.append(report[key]['f1-score'])
            support.append(report[key]['support'])
            index.append(f"Kelas {key}")

    # accuracy (satu nilai saja, support diisi None atau 0)
    precision.append(None)
    recall.append(None)
    f1_score.append(report['accuracy'])
    support.append(None)
    index.append("Accuracy")

    # macro avg
    precision.append(report['macro avg']['precision'])
    recall.append(report['macro avg']['recall'])
    f1_score.append(report['macro avg']['f1-score'])
    support.append(report['macro avg']['support'])
    index.append("Macro Avg")

    # weighted avg
    precision.append(report['weighted avg']['precision'])
    recall.append(report['weighted avg']['recall'])
    f1_score.append(report['weighted avg']['f1-score'])
    support.append(report['weighted avg']['support'])
    index.append("Weighted Avg")

    # buat dataframe
    report_data = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
        "support": support
    }

    return precision, recall, f1_score, support, index
