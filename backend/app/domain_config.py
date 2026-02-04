"""Domain configuration for multi-dataset PRISM support.

Each domain defines:
- Dataset schema (features, target, types)
- Human-readable labels for explainability
- Model paths
- Decision labels (what + and - mean)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings


@dataclass
class DomainConfig:
    """Configuration for a specific domain/dataset."""
    
    # Identifiers
    id: str
    name: str
    description: str
    
    # Schema
    feature_cols: list[str]
    target_col: str
    categorical_cols: list[str]
    numeric_cols: list[str]
    
    # Human-readable labels for features
    feature_labels: dict[str, str] = field(default_factory=dict)
    
    # Decision labels
    positive_label: str = "Approved"
    negative_label: str = "Rejected"
    positive_value: Any = 1
    negative_value: Any = 0
    
    # Paths (relative to artifacts dir)
    model_filename: str = "model.joblib"
    preprocessing_filename: str = "preprocessing.joblib"
    
    @property
    def model_path(self) -> Path:
        return settings.artifacts_dir / self.id / self.model_filename
    
    @property
    def preprocessing_path(self) -> Path:
        return settings.artifacts_dir / self.id / self.preprocessing_filename
    
    @property
    def artifacts_dir(self) -> Path:
        return settings.artifacts_dir / self.id


# ============== DOMAIN DEFINITIONS ==============

UCI_CREDIT_APPROVAL = DomainConfig(
    id="uci_credit_approval",
    name="UCI Credit Approval",
    description="Credit card application approval prediction",
    feature_cols=[f"A{i}" for i in range(1, 16)],
    target_col="A16",
    categorical_cols=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"],
    numeric_cols=["A2", "A3", "A8", "A11", "A14", "A15"],
    feature_labels={
        "A1": "Gender",
        "A2": "Age",
        "A3": "Debt",
        "A4": "Marital Status",
        "A5": "Bank Customer",
        "A6": "Education Level",
        "A7": "Ethnicity",
        "A8": "Years Employed",
        "A9": "Prior Default",
        "A10": "Employment Status",
        "A11": "Credit Score",
        "A12": "Drivers License",
        "A13": "Citizenship",
        "A14": "Zip Code",
        "A15": "Income",
    },
    positive_label="Approved",
    negative_label="Rejected",
    positive_value="+",
    negative_value="-",
)

GERMAN_CREDIT = DomainConfig(
    id="german_credit",
    name="German Credit (Statlog)",
    description="Credit risk classification for loan applicants",
    feature_cols=[
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment", "installment_commitment", "personal_status",
        "other_parties", "residence_since", "property_magnitude", "age",
        "other_payment_plans", "housing", "existing_credits", "job",
        "num_dependents", "own_telephone", "foreign_worker"
    ],
    target_col="class",
    categorical_cols=[
        "checking_status", "credit_history", "purpose", "savings_status",
        "employment", "personal_status", "other_parties", "property_magnitude",
        "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
    ],
    numeric_cols=[
        "duration", "credit_amount", "installment_commitment", "residence_since",
        "age", "existing_credits", "num_dependents"
    ],
    feature_labels={
        "checking_status": "Checking Account Status",
        "duration": "Loan Duration (months)",
        "credit_history": "Credit History",
        "purpose": "Loan Purpose",
        "credit_amount": "Credit Amount",
        "savings_status": "Savings Account Status",
        "employment": "Employment Duration",
        "installment_commitment": "Installment Rate (%)",
        "personal_status": "Personal Status & Gender",
        "other_parties": "Other Debtors/Guarantors",
        "residence_since": "Years at Residence",
        "property_magnitude": "Property Type",
        "age": "Age (years)",
        "other_payment_plans": "Other Payment Plans",
        "housing": "Housing Type",
        "existing_credits": "Number of Existing Credits",
        "job": "Job Type",
        "num_dependents": "Number of Dependents",
        "own_telephone": "Has Telephone",
        "foreign_worker": "Foreign Worker",
    },
    positive_label="Good Credit",
    negative_label="Bad Credit",
    positive_value=1,
    negative_value=2,
)

TAIWAN_CREDIT_CARD = DomainConfig(
    id="taiwan_credit_card",
    name="Taiwan Credit Card Default",
    description="Credit card default prediction for next month",
    feature_cols=[
        "limit_bal", "sex", "education", "marriage", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
    ],
    target_col="default_payment_next_month",
    categorical_cols=["sex", "education", "marriage"],
    numeric_cols=[
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"
    ],
    feature_labels={
        "limit_bal": "Credit Limit",
        "sex": "Gender",
        "education": "Education Level",
        "marriage": "Marital Status",
        "age": "Age",
        "pay_0": "Payment Status (Sep)",
        "pay_2": "Payment Status (Aug)",
        "pay_3": "Payment Status (Jul)",
        "pay_4": "Payment Status (Jun)",
        "pay_5": "Payment Status (May)",
        "pay_6": "Payment Status (Apr)",
        "bill_amt1": "Bill Amount (Sep)",
        "bill_amt2": "Bill Amount (Aug)",
        "bill_amt3": "Bill Amount (Jul)",
        "bill_amt4": "Bill Amount (Jun)",
        "bill_amt5": "Bill Amount (May)",
        "bill_amt6": "Bill Amount (Apr)",
        "pay_amt1": "Payment Amount (Sep)",
        "pay_amt2": "Payment Amount (Aug)",
        "pay_amt3": "Payment Amount (Jul)",
        "pay_amt4": "Payment Amount (Jun)",
        "pay_amt5": "Payment Amount (May)",
        "pay_amt6": "Payment Amount (Apr)",
    },
    positive_label="Will Default",
    negative_label="No Default",
    positive_value=1,
    negative_value=0,
)


# ============== HEALTHCARE DOMAINS ==============

HEART_DISEASE = DomainConfig(
    id="heart_disease",
    name="Heart Disease Prediction",
    description="Predict presence of heart disease based on clinical indicators",
    feature_cols=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    target_col="target",
    categorical_cols=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    numeric_cols=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
    feature_labels={
        "age": "Age",
        "sex": "Sex",
        "cp": "Chest Pain Type",
        "trestbps": "Resting Blood Pressure",
        "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120",
        "restecg": "Resting ECG Results",
        "thalach": "Max Heart Rate Achieved",
        "exang": "Exercise Induced Angina",
        "oldpeak": "ST Depression",
        "slope": "Slope of Peak Exercise ST",
        "ca": "Number of Major Vessels",
        "thal": "Thalassemia Type",
    },
    positive_label="Heart Disease Present",
    negative_label="No Heart Disease",
    positive_value=1,
    negative_value=0,
)

DIABETES = DomainConfig(
    id="diabetes",
    name="Diabetes Prediction",
    description="Predict diabetes onset based on diagnostic measurements",
    feature_cols=["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age"],
    target_col="outcome",
    categorical_cols=[],
    numeric_cols=["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age"],
    feature_labels={
        "pregnancies": "Number of Pregnancies",
        "glucose": "Glucose Level",
        "blood_pressure": "Blood Pressure (mm Hg)",
        "skin_thickness": "Skin Thickness (mm)",
        "insulin": "Insulin Level (mu U/ml)",
        "bmi": "BMI",
        "diabetes_pedigree": "Diabetes Pedigree Function",
        "age": "Age",
    },
    positive_label="Diabetes Positive",
    negative_label="Diabetes Negative",
    positive_value=1,
    negative_value=0,
)

BREAST_CANCER = DomainConfig(
    id="breast_cancer",
    name="Breast Cancer Diagnosis",
    description="Classify breast mass as malignant or benign based on cell nuclei features",
    feature_cols=[f'{feat}_{stat}' for feat in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'] for stat in ['mean', 'se', 'worst']],
    target_col="target",
    categorical_cols=[],
    numeric_cols=[f'{feat}_{stat}' for feat in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'] for stat in ['mean', 'se', 'worst']],
    feature_labels={
        "radius_mean": "Mean Radius",
        "texture_mean": "Mean Texture",
        "perimeter_mean": "Mean Perimeter",
        "area_mean": "Mean Area",
        "smoothness_mean": "Mean Smoothness",
        "compactness_mean": "Mean Compactness",
        "concavity_mean": "Mean Concavity",
        "concave_points_mean": "Mean Concave Points",
        "symmetry_mean": "Mean Symmetry",
        "fractal_dimension_mean": "Mean Fractal Dimension",
        "radius_se": "Radius Std Error",
        "texture_se": "Texture Std Error",
        "perimeter_se": "Perimeter Std Error",
        "area_se": "Area Std Error",
        "smoothness_se": "Smoothness Std Error",
        "compactness_se": "Compactness Std Error",
        "concavity_se": "Concavity Std Error",
        "concave_points_se": "Concave Points Std Error",
        "symmetry_se": "Symmetry Std Error",
        "fractal_dimension_se": "Fractal Dimension Std Error",
        "radius_worst": "Worst Radius",
        "texture_worst": "Worst Texture",
        "perimeter_worst": "Worst Perimeter",
        "area_worst": "Worst Area",
        "smoothness_worst": "Worst Smoothness",
        "compactness_worst": "Worst Compactness",
        "concavity_worst": "Worst Concavity",
        "concave_points_worst": "Worst Concave Points",
        "symmetry_worst": "Worst Symmetry",
        "fractal_dimension_worst": "Worst Fractal Dimension",
    },
    positive_label="Malignant",
    negative_label="Benign",
    positive_value=1,
    negative_value=0,
)


# ============== FINANCE DOMAIN ==============

BANK_MARKETING = DomainConfig(
    id="bank_marketing",
    name="Bank Marketing",
    description="Predict if client will subscribe to a term deposit",
    feature_cols=["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
    target_col="target",
    categorical_cols=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"],
    numeric_cols=["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
    feature_labels={
        "age": "Age",
        "job": "Job Type",
        "marital": "Marital Status",
        "education": "Education Level",
        "default": "Has Credit Default",
        "housing": "Has Housing Loan",
        "loan": "Has Personal Loan",
        "contact": "Contact Type",
        "month": "Contact Month",
        "day_of_week": "Contact Day",
        "duration": "Call Duration (sec)",
        "campaign": "Contacts This Campaign",
        "pdays": "Days Since Last Contact",
        "previous": "Previous Contacts",
        "poutcome": "Previous Outcome",
        "emp.var.rate": "Employment Variation Rate",
        "cons.price.idx": "Consumer Price Index",
        "cons.conf.idx": "Consumer Confidence Index",
        "euribor3m": "Euribor 3 Month Rate",
        "nr.employed": "Number of Employees",
    },
    positive_label="Will Subscribe",
    negative_label="Will Not Subscribe",
    positive_value=1,
    negative_value=0,
)


# ============== EDUCATION DOMAIN ==============

STUDENT_PERFORMANCE = DomainConfig(
    id="student_performance",
    name="Student Performance",
    description="Predict if student will pass or fail based on academic and social factors",
    feature_cols=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"],
    target_col="target",
    categorical_cols=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"],
    numeric_cols=["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"],
    feature_labels={
        "school": "School",
        "sex": "Sex",
        "age": "Age",
        "address": "Address Type",
        "famsize": "Family Size",
        "Pstatus": "Parent Cohabitation Status",
        "Medu": "Mother's Education",
        "Fedu": "Father's Education",
        "Mjob": "Mother's Job",
        "Fjob": "Father's Job",
        "reason": "Reason for School Choice",
        "guardian": "Guardian",
        "traveltime": "Travel Time to School",
        "studytime": "Weekly Study Time",
        "failures": "Past Class Failures",
        "schoolsup": "Extra School Support",
        "famsup": "Family Support",
        "paid": "Extra Paid Classes",
        "activities": "Extracurricular Activities",
        "nursery": "Attended Nursery",
        "higher": "Wants Higher Education",
        "internet": "Internet Access",
        "romantic": "In Romantic Relationship",
        "famrel": "Family Relationship Quality",
        "freetime": "Free Time After School",
        "goout": "Going Out with Friends",
        "Dalc": "Weekday Alcohol Consumption",
        "Walc": "Weekend Alcohol Consumption",
        "health": "Health Status",
        "absences": "Number of Absences",
    },
    positive_label="Pass",
    negative_label="Fail",
    positive_value=1,
    negative_value=0,
)


# ============== EMPLOYMENT/HR DOMAIN ==============

HR_ATTRITION = DomainConfig(
    id="hr_attrition",
    name="Employee Attrition",
    description="Predict employee attrition based on HR analytics",
    feature_cols=[
        "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
        "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
        "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
        "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
        "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ],
    target_col="target",
    categorical_cols=[
        "BusinessTravel", "Department", "EducationField", "Gender",
        "JobRole", "MaritalStatus", "OverTime"
    ],
    numeric_cols=[
        "Age", "DailyRate", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel",
        "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
        "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
        "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ],
    feature_labels={
        "Age": "Age",
        "BusinessTravel": "Business Travel Frequency",
        "DailyRate": "Daily Rate",
        "Department": "Department",
        "DistanceFromHome": "Distance From Home (miles)",
        "Education": "Education Level",
        "EducationField": "Education Field",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "Gender": "Gender",
        "HourlyRate": "Hourly Rate",
        "JobInvolvement": "Job Involvement",
        "JobLevel": "Job Level",
        "JobRole": "Job Role",
        "JobSatisfaction": "Job Satisfaction",
        "MaritalStatus": "Marital Status",
        "MonthlyIncome": "Monthly Income",
        "MonthlyRate": "Monthly Rate",
        "NumCompaniesWorked": "Number of Companies Worked",
        "OverTime": "Works Overtime",
        "PercentSalaryHike": "Percent Salary Hike",
        "PerformanceRating": "Performance Rating",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "StockOptionLevel": "Stock Option Level",
        "TotalWorkingYears": "Total Working Years",
        "TrainingTimesLastYear": "Training Times Last Year",
        "WorkLifeBalance": "Work-Life Balance",
        "YearsAtCompany": "Years at Company",
        "YearsInCurrentRole": "Years in Current Role",
        "YearsSinceLastPromotion": "Years Since Last Promotion",
        "YearsWithCurrManager": "Years With Current Manager",
    },
    positive_label="Will Leave",
    negative_label="Will Stay",
    positive_value=1,
    negative_value=0,
)


# ============== INSURANCE DOMAIN ==============

INSURANCE_COIL = DomainConfig(
    id="insurance_coil",
    name="Caravan Insurance",
    description="Predict caravan insurance policy purchase likelihood",
    feature_cols=[
        "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD",
        "MGODRK", "MGODPR", "MGODOV", "MGODGE",
        "MRELGE", "MRELSA", "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND",
        "MOPLHOOG", "MOPLMIDD", "MOPLLAAG", "MBERHOOG", "MBERZELF",
        "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO",
        "MSKA", "MSKB1", "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP",
        "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART",
        "MINKM30", "MINK3045", "MINK4575", "MINK7512", "MINK123M", "MINKGEM",
        "MKOOPHOOG", "MKOOPLPLG", "MKOOPKLA",
        "PWAPART", "PWABEDR", "PWALAND", "PPERSAUT", "PBESAUT", "PMOTSCO",
        "PVRAAUT", "PATEFTT", "PWERKT", "PBROM",
        "PLEVEN", "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL",
        "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND",
        "AWAPART", "AWABEDR", "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO",
        "AVRAAUT", "AATEFTT", "AWERKT", "ABROM",
        "ALEVEN", "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL",
        "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND"
    ],
    target_col="target",
    categorical_cols=[],  # All columns are encoded as integers
    numeric_cols=[
        "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD",
        "MGODRK", "MGODPR", "MGODOV", "MGODGE",
        "MRELGE", "MRELSA", "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND",
        "MOPLHOOG", "MOPLMIDD", "MOPLLAAG", "MBERHOOG", "MBERZELF",
        "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO",
        "MSKA", "MSKB1", "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP",
        "MAUT1", "MAUT2", "MAUT0", "MZFONDS", "MZPART",
        "MINKM30", "MINK3045", "MINK4575", "MINK7512", "MINK123M", "MINKGEM",
        "MKOOPHOOG", "MKOOPLPLG", "MKOOPKLA",
        "PWAPART", "PWABEDR", "PWALAND", "PPERSAUT", "PBESAUT", "PMOTSCO",
        "PVRAAUT", "PATEFTT", "PWERKT", "PBROM",
        "PLEVEN", "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL",
        "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND",
        "AWAPART", "AWABEDR", "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO",
        "AVRAAUT", "AATEFTT", "AWERKT", "ABROM",
        "ALEVEN", "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL",
        "APLEZIER", "AFIETS", "AINBOED", "ABYSTAND"
    ],
    feature_labels={
        "MOSTYPE": "Customer Subtype",
        "MAANTHUI": "Number of Houses",
        "MGEMOMV": "Avg Household Size",
        "MGEMLEEF": "Avg Age",
        "MOSHOOFD": "Customer Main Type",
        "MGODRK": "Roman Catholic",
        "MGODPR": "Protestant",
        "MGODOV": "Other Religion",
        "MGODGE": "No Religion",
        "MRELGE": "Married",
        "MRELSA": "Living Together",
        "MRELOV": "Other Relation",
        "MFALLEEN": "Singles",
        "MFGEKIND": "Household Without Children",
        "MFWEKIND": "Household With Children",
        "MOPLHOOG": "High Level Education",
        "MOPLMIDD": "Medium Level Education",
        "MOPLLAAG": "Low Level Education",
        "MBERHOOG": "High Status",
        "MBERZELF": "Entrepreneur",
        "MBERBOER": "Farmer",
        "MBERMIDD": "Middle Management",
        "MBERARBG": "Skilled Labor",
        "MBERARBO": "Unskilled Labor",
        "MSKA": "Social Class A",
        "MSKB1": "Social Class B1",
        "MSKB2": "Social Class B2",
        "MSKC": "Social Class C",
        "MSKD": "Social Class D",
        "MHHUUR": "Rented House",
        "MHKOOP": "Home Owners",
        "MAUT1": "1 Car",
        "MAUT2": "2 Cars",
        "MAUT0": "No Car",
        "MZFONDS": "National Health Insurance",
        "MZPART": "Private Health Insurance",
        "MINKM30": "Income < 30K",
        "MINK3045": "Income 30-45K",
        "MINK4575": "Income 45-75K",
        "MINK7512": "Income 75-122K",
        "MINK123M": "Income > 123K",
        "MINKGEM": "Average Income",
        "MKOOPHOOG": "High Purchasing Power",
        "MKOOPLPLG": "Low Purchasing Power",
        "MKOOPKLA": "Purchasing Power Class",
        "PWAPART": "Private 3rd Party Ins Contrib",
        "PWABEDR": "3rd Party Ins (Firms) Contrib",
        "PWALAND": "3rd Party Ins (Agriculture) Contrib",
        "PPERSAUT": "Car Insurance Contrib",
        "PBESAUT": "Delivery Van Ins Contrib",
        "PMOTSCO": "Motorcycle/Scooter Ins Contrib",
        "PVRAAUT": "Lorry Insurance Contrib",
        "PATEFTT": "Trailer Insurance Contrib",
        "PWERKT": "Tractor Insurance Contrib",
        "PBROM": "Moped Insurance Contrib",
        "PLEVEN": "Life Insurance Contrib",
        "PPERSONG": "Accident Ins Contrib",
        "PGEZONG": "Family Accident Ins Contrib",
        "PWAOREG": "Disability Insurance Contrib",
        "PBRAND": "Fire Insurance Contrib",
        "PZEILPL": "Surfboard Insurance Contrib",
        "PPLEZIER": "Boat Insurance Contrib",
        "PFIETS": "Bicycle Insurance Contrib",
        "PINBOED": "Property Insurance Contrib",
        "PBYSTAND": "Social Security Contrib",
        "AWAPART": "Private 3rd Party Ins Policies",
        "AWABEDR": "3rd Party Ins (Firms) Policies",
        "AWALAND": "3rd Party Ins (Agriculture) Policies",
        "APERSAUT": "Car Insurance Policies",
        "ABESAUT": "Delivery Van Ins Policies",
        "AMOTSCO": "Motorcycle/Scooter Ins Policies",
        "AVRAAUT": "Lorry Insurance Policies",
        "AATEFTT": "Trailer Insurance Policies",
        "AWERKT": "Tractor Insurance Policies",
        "ABROM": "Moped Insurance Policies",
        "ALEVEN": "Life Insurance Policies",
        "APERSONG": "Accident Ins Policies",
        "AGEZONG": "Family Accident Ins Policies",
        "AWAOREG": "Disability Insurance Policies",
        "ABRAND": "Fire Insurance Policies",
        "AZEILPL": "Surfboard Insurance Policies",
        "APLEZIER": "Boat Insurance Policies",
        "AFIETS": "Bicycle Insurance Policies",
        "AINBOED": "Property Insurance Policies",
        "ABYSTAND": "Social Security Policies",
    },
    positive_label="Will Buy",
    negative_label="Won't Buy",
    positive_value=1,
    negative_value=0,
)


# ============== LEGAL DOMAIN ==============

RECIDIVISM_COMPAS = DomainConfig(
    id="recidivism_compas",
    name="Recidivism Prediction",
    description="Predict likelihood of reoffending within two years (COMPAS dataset)",
    feature_cols=["age", "sex", "race", "juv_fel_count", "juv_misd_count", 
                  "juv_other_count", "priors_count", "c_charge_degree"],
    target_col="target",
    categorical_cols=["sex", "race", "c_charge_degree"],
    numeric_cols=["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"],
    feature_labels={
        "age": "Age",
        "sex": "Sex",
        "race": "Race",
        "juv_fel_count": "Juvenile Felony Count",
        "juv_misd_count": "Juvenile Misdemeanor Count",
        "juv_other_count": "Juvenile Other Count",
        "priors_count": "Prior Offenses Count",
        "c_charge_degree": "Charge Degree",
    },
    positive_label="Will Reoffend",
    negative_label="Won't Reoffend",
    positive_value=1,
    negative_value=0,
)


# ============== DOMAIN REGISTRY ==============

DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    # Credit/Finance
    "uci_credit_approval": UCI_CREDIT_APPROVAL,
    "german_credit": GERMAN_CREDIT,
    "taiwan_credit_card": TAIWAN_CREDIT_CARD,
    "bank_marketing": BANK_MARKETING,
    # Healthcare
    "heart_disease": HEART_DISEASE,
    "diabetes": DIABETES,
    "breast_cancer": BREAST_CANCER,
    # Education
    "student_performance": STUDENT_PERFORMANCE,
    # Employment/HR
    "hr_attrition": HR_ATTRITION,
    # Insurance
    "insurance_coil": INSURANCE_COIL,
    # Legal
    "recidivism_compas": RECIDIVISM_COMPAS,
}


def get_domain(domain_id: str) -> DomainConfig | None:
    """Get domain configuration by ID."""
    return DOMAIN_REGISTRY.get(domain_id)


def list_domains() -> list[DomainConfig]:
    """List all available domains."""
    return list(DOMAIN_REGISTRY.values())


def is_model_trained(domain_id: str) -> bool:
    """Check if a model has been trained for this domain."""
    domain = get_domain(domain_id)
    if not domain:
        return False
    return domain.model_path.exists() and domain.preprocessing_path.exists()
