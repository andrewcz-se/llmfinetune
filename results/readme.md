# Clinical Extraction Model Evaluation Report
**Date:** 2026-01-01 13:22:37

## Loss Curve

![Loss Curve](loss_curve.png "Loss Curve")

## 1. Summary of Results
| Test Case | Model | Status | Detail |
| :--- | :--- | :--- | :--- |
| Standard Case | Base Model (Qwen 2.5 3B) | ❌ FAIL | Invalid JSON |
| Missing Age (Null Check) | Base Model (Qwen 2.5 3B) | ❌ FAIL | Invalid JSON |
| Multi-Medication 1 (Array Check) | Base Model (Qwen 2.5 3B) | ❌ FAIL | Invalid JSON |
| Multi-Medication 2 (Array Check) | Base Model (Qwen 2.5 3B) | ❌ FAIL | Invalid JSON |
| Enhanced Standard Case | Base Model (Qwen 2.5 3B) | ❌ FAIL | Invalid JSON |
| Standard Case | Fine-Tuned (Clinical-Extract) | ✅ PASS | Perfect |
| Missing Age (Null Check) | Fine-Tuned (Clinical-Extract) | ✅ PASS | Perfect |
| Multi-Medication 1 (Array Check) | Fine-Tuned (Clinical-Extract) | ✅ PASS | Perfect |
| Multi-Medication 2 (Array Check) | Fine-Tuned (Clinical-Extract) | ✅ PASS | Perfect |
| Enhanced Standard Case | Fine-Tuned (Clinical-Extract) | ✅ PASS | Perfect |

## 2. Detailed Input/Output Logs

### Test Case: Standard Case
**Input Note:**
> Pt John Doe, 55y, history of T2DM. Meds: Metformin 500mg.

#### Base Model Output
```json
Given the information provided, it appears you want to extract a clinical data record into a strict Solaris-compliant JSON format for a patient named John Doe. Here's how the data can be structured:


{
  "patient": {
    "name": "John Doe",
    "age": 55,
    "gender": "N/A", // Assuming gender is not specified or not applicable
    "diagnosis": "T2DM",
    "medications": [
      {
        "name": "Metformin",
        "dosage": "500mg"
      }
    ]
  }
}


This JSON structure adheres to the requirements and includes all the necessary clinical data in a clear and organized manner. Note that some fields like gender and medications might need more specific details depending on the context and the system's requirements.
```
*Analysis: FAIL - Invalid JSON*

#### Fine-Tuned Model Output
```json
{"resourceType": "SolarisEMRClinicalSummary", "status": "finished", "subject": {"reference": "Patient/JohnDoe", "display": "John Doe", "ageInt": 55}, "reasonCode": [{"text": "T2DM"}], "medicationRequest": [{"medicationCodeableConcept": {"text": "Metformin"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 500, "unit": "mg"}}]}]}]}
```
*Analysis: PASS - Perfect*

---

### Test Case: Missing Age (Null Check)
**Input Note:**
> Patient Sarah Connor presents with severe anxiety. Rx: Diazepam 5mg.

#### Base Model Output
```json
Given the instruction to extract clinical data into a strict Solaris-compliant JSON format, here is the clinical data extracted from the provided scenario:


{
  "patient": {
    "name": "Sarah Connor",
    "diagnosis": "severe anxiety"
  },
  "prescription": {
    "medication": "Diazepam",
    "dosage": "5mg"
  }
}


This JSON structure adheres to the Solaris-compliant format and encapsulates the patient's information along with the prescribed medication in a clear and organized manner.
```
*Analysis: FAIL - Invalid JSON*

#### Fine-Tuned Model Output
```json
{"resourceType": "SolarisEMRClinicalSummary", "status": "finished", "subject": {"reference": "Patient/SarahConnor", "display": "Sarah Connor", "ageInt": null}, "reasonCode": [{"text": "severe anxiety"}], "medicationRequest": [{"medicationCodeableConcept": {"text": "Diazepam"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 5, "unit": "mg"}}]}]}]}
```
*Analysis: PASS - Perfect*

---

### Test Case: Multi-Medication 1 (Array Check)
**Input Note:**
> Pt Paul Bunyon, 48 yrs, reports fever. Prescribe: Ibuprofen 400mg and Paracetamol 500mg.

#### Base Model Output
```json
The clinical data extracted into a strict Solaris-compliant JSON format is as follows:


{
  "patient": {
    "name": "Paul Bunyon",
    "age": 48,
    "symptoms": ["fever"]
  },
  "prescription": {
    "medicines": [
      {
        "name": "Ibuprofen",
        "dosage": "400mg"
      },
      {
        "name": "Paracetamol",
        "dosage": "500mg"
      }
    ]
  }
}


This JSON structure adheres to the Solaris-compliant format while encapsulating the patient's details and the prescribed medications.
```
*Analysis: FAIL - Invalid JSON*

#### Fine-Tuned Model Output
```json
{"resourceType": "SolarisEMRClinicalSummary", "status": "finished", "subject": {"reference": "Patient/PaulBunyon", "display": "Paul Bunyon", "ageInt": 48}, "reasonCode": [{"text": "fever"}], "medicationRequest": [{"medicationCodeableConcept": {"text": "Ibuprofen"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 400, "unit": "mg"}}]}]}, {"medicationCodeableConcept": {"text": "Paracetamol"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 500, "unit": "mg"}}]}]}]}
```
*Analysis: PASS - Perfect*

---

### Test Case: Multi-Medication 2 (Array Check)
**Input Note:**
> Pt Dave Smith, 28y, reports headache. Rx: Codeine 400mg and Paracetamol 500mg.

#### Base Model Output
```json
To extract the clinical data into a strict Solaris-compliant JSON format, we can structure it as follows:


{
  "patient": {
    "name": "Dave Smith",
    "age": 28,
    "gender": "N/A", // Assuming gender is not provided or unknown
    "medical_history": []
  },
  "symptoms": {
    "headache": true
  },
  "medications": [
    {
      "name": "Codeine 400mg",
      "dosage": "as needed"
    },
    {
      "name": "Paracetamol 500mg",
      "dosage": "as needed"
    }
  ]
}


This JSON structure is designed to be Solaris-compliant and includes all the necessary information from the clinical data provided. Note that some fields like gender and medical history are left blank since they were not specified in the input. If more details were available, these could be filled in accordingly.
```
*Analysis: FAIL - Invalid JSON*

#### Fine-Tuned Model Output
```json
{"resourceType": "SolarisEMRClinicalSummary", "status": "finished", "subject": {"reference": "Patient/DaveSmith", "display": "Dave Smith", "ageInt": 28}, "reasonCode": [{"text": "headache"}], "medicationRequest": [{"medicationCodeableConcept": {"text": "Codeine"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 400, "unit": "mg"}}]}]}, {"medicationCodeableConcept": {"text": "Paracetamol"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 500, "unit": "mg"}}]}]}]}
```
*Analysis: PASS - Perfect*

---

### Test Case: Enhanced Standard Case
**Input Note:**
> Pete Jones (68), presents diarrhea. Dx gastroenteritis. Prescribe Loperamide 20mg, Gaviscon 250ml, Omprazole 50mg

#### Base Model Output
```json
It appears that the instruction you've provided is not directly related to extracting clinical data into a strict Solaris-compliant JSON format. However, I can extract the clinical data into a structured JSON format that could be used for various purposes, such as medical records or patient information systems.

Here's an example of how the clinical data could be represented in a Solaris-compliant JSON format:


{
  "patient": {
    "name": "Jones, Pete",
    "age": 68,
    "gender": "N/A", // Assuming gender is not applicable or unknown
    "diagnosis": "Gastroenteritis"
  },
  "prescriptions": [
    {
      "drug_name": "Loperamide",
      "dose": "20mg",
      "route": "Oral",
      "frequency": "As needed"
    },
    {
      "drug_name": "Gaviscon",
      "dose": "250ml",
      "route": "Oral",
      "frequency": "As directed"
    },
    {
      "drug_name": "Omeprazole",
      "dose": "50mg",
      "route": "Oral",
      "frequency": "Once daily"
    }
  ]
}


This JSON structure captures the patient's details and the prescribed medications in a clear and organized manner. Note that some fields like "gender" and "frequency" are left blank
```
*Analysis: FAIL - Invalid JSON*

#### Fine-Tuned Model Output
```json
{"resourceType": "SolarisEMRClinicalSummary", "status": "finished", "subject": {"reference": "Patient/PeteJones", "display": "Pete Jones", "ageInt": 68}, "reasonCode": [{"text": "diarrhea"}], "diagnosis": {"condition": "gastroenteritis", "clinicalStatus": "active"}, "medicationRequest": [{"medicationCodeableConcept": {"text": "Loperamide"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 20, "unit": "mg"}}]}]}, {"medicationCodeableConcept": {"text": "Gaviscon"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 250, "unit": "ml"}}]}]}, {"medicationCodeableConcept": {"text": "Omprazole"}, "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 50, "unit": "mg"}}]}]}]}
```
*Analysis: PASS - Perfect*

---
