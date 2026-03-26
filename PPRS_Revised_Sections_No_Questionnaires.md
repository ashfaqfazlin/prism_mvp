# PPRS Revised Sections: Aligned with Ethical Form (No Questionnaires / No Human Participants)

Use these revised paragraphs in your Project Proposal and Requirements Specification. Each block is ready to drop in where the corresponding section appears. The project is framed as **secondary research (SLR) + technical/system validation only**—no user studies, questionnaires, interviews, or surveys.

---

## 1. Table 3 – Saunders' Research Onion (Chapter 3, ~p. 34)

**Replace the row "Techniques and Procedures" (and any mention of SUS questionnaires).**

**Revised text for "Techniques and Procedures" cell:**

> Data collection using open tabular datasets. No human participants are recruited; evaluation relies on technical validation (model accuracy, SHAP fidelity, counterfactual consistency) and synthesis of existing published research. Quantitative analysis through accuracy measures, explanation stability, and computational performance. Qualitative evaluation is derived from thematic synthesis of the literature and design review against established XAI and HCI principles. Tools include Python, XGBoost, SHAP, React, and standard software testing frameworks. No questionnaires, surveys, or interviews are used.

---

## 2. Section 3.4.2 – User-Centred Design Approach (Chapter 3, ~p. 38)

**Replace the full paragraph.**

**Revised paragraph:**

> A user-centred design (UCD) process will be integrated within the DSR framework to ensure usability and interpretability from the earliest stages. Requirement elicitation will be conducted through literature review and document analysis only; no interviews or surveys are used, in line with the approved ethical application. User needs such as explanation clarity, transparency, and ease of navigation will be inferred from existing XAI and HCI literature and prioritised during interface development. Wireframes and low-fidelity prototypes will be created using Figma to test interaction flow and layout. Refinements to visual hierarchy, terminology, and interactive controls will be informed by design heuristics and literature-based usability principles rather than direct user feedback. This process will ensure that the interface is designed to support exploration of explanations without undue cognitive overload, consistent with findings from the systematic review.

---

## 3. Section 3.4.5 – Iterative Refinement and Evaluation (Chapter 3, ~p. 39)

**Replace the full paragraph.**

**Revised paragraph:**

> Each design iteration will undergo technical validation and analytical evaluation. Early prototypes will be assessed for layout clarity and logical flow against design principles drawn from the literature. Later iterations will be evaluated for explanation fidelity, interface consistency, and alignment with documented best practices in explainable AI. Refinements to visual structure and interface density will be informed by supervisor feedback, internal design review, and literature-based criteria. This iterative improvement process will ensure that design decisions remain grounded in technical validation and evidence from the systematic review, without reliance on human-participant studies.

---

## 4. Section 3.4.6 – Design Validation and Ethical Compliance (Chapter 3, ~p. 39)

**Replace the full paragraph.**

**Revised paragraph:**

> Design validation will follow the ethical principles approved in the PRISM Ethical Clearance Form. As the study involves no human participants—relying exclusively on secondary research (systematic literature review) and technical development—no participant data, consent procedures, or user studies are required. System design will maintain transparency, data protection for any logged interaction data (e.g. anonymised usage logs if applicable), and alignment with institutional research standards. Validation is achieved through technical testing, literature-informed design review, and compliance with the approved ethical application.

---

## 5. Section 3.6.3 – Usability and Human-Centred Testing (Chapter 3, ~p. 42–43)

**Replace the full paragraph.**

**Revised paragraph:**

> In addition to software validation, the PRISM interface will be evaluated for interpretability and usability through technical and design-based methods only. No human participants are recruited; no questionnaires (e.g. SUS, NASA-TLX), surveys, or interviews are used. Evaluation will focus on (1) functional correctness of explanation outputs (SHAP and counterfactual consistency), (2) interface behaviour under defined usage scenarios (e.g. upload, prediction, explanation generation, export), and (3) alignment of design with usability and interpretability principles identified in the systematic literature review. Metrics such as explanation fidelity, response time, and correctness of exported outputs will be recorded. Design clarity and cognitive alignment will be assessed against criteria derived from the literature rather than from participant feedback. The collected technical and design insights will inform subsequent refinements in system functionality and interface design.

---

## 6. Section 3.6.8 – Ethical Compliance in Testing (Chapter 3, ~p. 44)

**Replace the full paragraph.**

**Revised paragraph:**

> All testing activities will be carried out in accordance with the approved PRISM Ethical Clearance Form. As the study involves no human participants—no user studies, questionnaires, or interviews—no participant consent or identifiable personal data is involved. Testing is limited to technical validation (unit, integration, system tests) and evaluation of the artefact against literature-based design criteria. Any system or interaction logs used for debugging or analysis will contain only non-identifiable, technical data and will be handled in compliance with institutional and ethical research standards.

---

## 7. Section 3.7.6 – Testing (Solution Methodology) (Chapter 3, ~p. 46)

**Replace the sentence that refers to "Qualitative user testing".**

**Revised paragraph (full 3.7.6 block):**

> Testing will involve both quantitative and qualitative evaluation at the technical and design level. Quantitative measures such as accuracy, precision, recall, and F1 score will be used to assess predictive performance. SHAP explanations will be validated for consistency across multiple instances to confirm fidelity and stability. Counterfactual outputs will be reviewed to ensure logical reasoning, where small feature changes produce coherent decision variations. Qualitative design evaluation will assess explanation clarity and interface coherence against criteria derived from the systematic literature review and established XAI/HCI principles, without human-participant studies or questionnaires.

---

## 8. Section 3.7.7 – Feedback Loop (Chapter 3, ~p. 46)

**Replace the full paragraph.**

**Revised paragraph:**

> A continuous feedback loop will guide system refinement throughout the project lifecycle. Insights from technical testing, supervisor feedback, and literature-based design review will inform improvements in both the model and the interface design. For example, adjustments will be made to terminology or visual presentation if design review or literature suggests potential confusion. Retraining or parameter tuning will be performed when data transformation or model representation requires enhancement. This iterative process will ensure that the final version of PRISM achieves high interpretability, technical accuracy, and alignment with evidence-based design principles, without reliance on participant-based evaluation.

---

## 9. Section 3.8.1 – Schedule / Table 4 (Chapter 3, ~p. 47)

**Replace the deliverables table rows that refer to Pilot Study and Main Study.**

**Revised table (only the rows that change):**

| Deliverables              | Target Date |
|---------------------------|-------------|
| Proposal + Ethics submission | Week 2   |
| Low-fidelity UI + Requirements Specification | Week 4 |
| Base ML model + SHAP/Counterfactual prototype services | Week 6 |
| UI v1 + Logging + Basic Integration | Week 8 |
| **Prototype validation complete** | Week 10 |
| Revised UI + Analysis plan (technical & literature-based) | Week 11 |
| **Technical evaluation complete** | Week 14 |
| Analysis & Results Chapter Draft | Week 16 |
| Final Thesis, Artefact Pack, Demo Video | Week 18 |

**Optional short note under the table:**

> Evaluation milestones refer to technical validation and literature-based assessment only; no pilot or main user studies are conducted, in line with the approved ethical application.

---

## 10. Research Objective RO6 – Table 2 (Chapter 1, ~p. 9)

**Replace the RO6 row description.**

**Revised RO6 cell (Description column):**

> Evaluate the interactive and static explanation interfaces through technical and literature-based evaluation. Conduct validation of explanation fidelity, task correctness in automated scenarios, and alignment with design principles from the systematic review. No human participants are used; metrics are derived from model performance, explanation stability, and literature-informed usability criteria.

---

## 11. Research Objective RO7 – Table 2 (Chapter 1, ~p. 9)

**Replace the RO7 row description.**

**Revised RO7 cell (Description column):**

> Analyse quantitative and literature-based results to assess system performance and design alignment. Perform statistical or analytical tests on model and explanation metrics where applicable, and synthesise findings from the systematic review to evaluate how the PRISM design addresses documented gaps in interpretability and usability. No participant-based thematic analysis is conducted.

---

## 12. Abstract (front of document, ~p. 2)

**Replace the evaluation sentence in the abstract.**

**Revised sentence (replace the sentence that mentions "qualitative and quantitative analysis of user interactions"):**

> The evaluation involves technical validation and literature-based assessment of the interface design to demonstrate alignment with criteria for understanding, trust, cognitive load, and usability drawn from existing XAI and HCI research, without human-participant studies or questionnaires.

**Or, if you prefer a slightly longer abstract clause:**

> The evaluation involves quantitative validation of model and explanation performance alongside literature-based assessment of the interface against established criteria for interpretability, usability, and trust. No user studies or questionnaires are conducted; findings are grounded in technical metrics and synthesis of the systematic literature review.

---

## 13. Section 1.12.1 – In Scope (Chapter 1, ~p. 10)

**Replace the sentence that refers to "controlled user study".**

**Revised paragraph:**

> This study will focus on publicly accessible tabular datasets (for example, credit-approval data) and employ a single opaque (“black-box”) classifier model. The developed system will incorporate two explanation modalities: SHAP-based feature attribution and counterfactual recourse in interactive what-if scenarios. Two user interface variants will be implemented: one offering a static attribution view and the other supporting interactive exploration via counterfactuals. The evaluation will be restricted to technical validation (model and explanation performance) and literature-based assessment of the design against criteria for comprehension, usability, trust, and task performance identified in the systematic review. No human participants, questionnaires, or user studies are used. Additionally, the system will enable export of decision rationales and interaction logs in both PDF and CSV formats, thereby supporting auditability and traceability of individual decisions.

---

## 14. Chapter 2.5.2 – Qualitative Evaluation (Chapter 2, ~p. 28)

**Add a short clarifying paragraph at the end of 2.5.2 (after the existing content).**

**New paragraph to add:**

> The present PRISM project does not conduct participant-based qualitative evaluation; it relies on secondary research and technical validation only. The instruments and methods described above (e.g. SUS, NASA-TLX, interviews) are cited as part of the literature on XAI evaluation and are not used in this study, in accordance with the approved ethical application.

---

## 15. Chapter 4.1 – SRS Chapter Overview (Chapter 4, ~p. 56)

**Optional: tighten the sentence about how requirements are derived.**

**Replace the sentence:**  
"These requirements are derived through a structured process that involves analysing research objectives, **conducting stakeholder discussions**, reviewing literature, and evaluating feasibility and ethical compliance."

**With:**

> These requirements are derived through a structured process that involves analysing research objectives, reviewing literature and existing systems, and evaluating feasibility and ethical compliance. No stakeholder interviews or surveys are conducted; requirements are informed by the systematic literature review and academic supervision.

---

## Quick checklist

- [ ] Table 3 (Saunders' Onion) – Techniques and Procedures
- [ ] 3.4.2 User-Centred Design
- [ ] 3.4.5 Iterative Refinement and Evaluation
- [ ] 3.4.6 Design Validation and Ethical Compliance
- [ ] 3.6.3 Usability and Human-Centred Testing
- [ ] 3.6.8 Ethical Compliance in Testing
- [ ] 3.7.6 Testing (Solution Methodology)
- [ ] 3.7.7 Feedback Loop
- [ ] Table 4 Schedule (Pilot Study / Main Study)
- [ ] Table 2 RO6 and RO7
- [ ] Abstract
- [ ] 1.12.1 In Scope
- [ ] 2.5.2 Qualitative Evaluation (new paragraph)
- [ ] 4.1 Chapter Overview (optional)

---

*Document generated for PRISM PPRS alignment with Ethical Form (no questionnaires, no human participants).*
