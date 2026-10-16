"""
Generates synthetic clinical speech transcript pairs and knowledge base documents.
Run: python data/generate_synthetic_data.py
"""

import json
import random
import hashlib
from pathlib import Path

random.seed(42)

CATEGORIES = ["NICU", "CVICU", "PICU", "ICU", "HUC", "ED"]

# Clinical acronym definitions for the RAG knowledge base
KB_DOCUMENTS = [
    {
        "id": "kb_001",
        "category": "NICU",
        "title": "NICU — Neonatal Intensive Care Unit",
        "content": (
            "NICU stands for Neonatal Intensive Care Unit. It provides care for premature or critically ill newborns. "
            "Common terms include corrected gestational age (CGA), continuous positive airway pressure (CPAP), "
            "bronchopulmonary dysplasia (BPD), necrotizing enterocolitis (NEC), and retinopathy of prematurity (ROP). "
            "NICU patients may require mechanical ventilation, total parenteral nutrition (TPN), and phototherapy."
        ),
    },
    {
        "id": "kb_002",
        "category": "CVICU",
        "title": "CVICU — Cardiovascular Intensive Care Unit",
        "content": (
            "CVICU stands for Cardiovascular Intensive Care Unit. It handles post-cardiac surgery and severe cardiac "
            "conditions. Key terms: coronary artery bypass graft (CABG), intra-aortic balloon pump (IABP), "
            "extracorporeal membrane oxygenation (ECMO), left ventricular assist device (LVAD), atrial fibrillation (AFib), "
            "ejection fraction (EF), and pulmonary capillary wedge pressure (PCWP)."
        ),
    },
    {
        "id": "kb_003",
        "category": "PICU",
        "title": "PICU — Pediatric Intensive Care Unit",
        "content": (
            "PICU stands for Pediatric Intensive Care Unit. It cares for critically ill children. "
            "Common terms include respiratory syncytial virus (RSV), Glasgow Coma Scale (GCS), "
            "status epilepticus (SE), diabetic ketoacidosis (DKA), septic shock, and "
            "acute respiratory distress syndrome (ARDS). Sedation protocols and weight-based dosing are standard."
        ),
    },
    {
        "id": "kb_004",
        "category": "ICU",
        "title": "ICU — Intensive Care Unit",
        "content": (
            "ICU stands for Intensive Care Unit. General critical care terminology includes "
            "mean arterial pressure (MAP), positive end-expiratory pressure (PEEP), fraction of inspired oxygen (FiO2), "
            "sequential organ failure assessment (SOFA), central venous pressure (CVP), "
            "blood urea nitrogen (BUN), creatinine clearance (CrCl), and vasopressors such as norepinephrine."
        ),
    },
    {
        "id": "kb_005",
        "category": "HUC",
        "title": "HUC — Health Unit Coordinator",
        "content": (
            "HUC stands for Health Unit Coordinator (also called unit clerk or ward clerk). "
            "Responsibilities include managing physician orders, coordinating admissions, discharges and transfers (ADT), "
            "maintaining medical records (MR), processing lab requisitions, and communicating with nursing staff. "
            "HUC works closely with the charge nurse and handles EMR (Electronic Medical Records) entry."
        ),
    },
    {
        "id": "kb_006",
        "category": "ED",
        "title": "ED — Emergency Department",
        "content": (
            "ED stands for Emergency Department. Key acronyms include triage levels (ESI 1–5), "
            "chief complaint (CC), history of present illness (HPI), mechanism of injury (MOI), "
            "point-of-care testing (POCT), rapid sequence intubation (RSI), "
            "emergency medical technician (EMT), and trauma activation. "
            "STEMI (ST-elevation myocardial infarction) and stroke codes are time-critical ED events."
        ),
    },
    {
        "id": "kb_007",
        "category": "NICU",
        "title": "NICU — Respiratory Support Terminology",
        "content": (
            "NICU respiratory support includes high-flow nasal cannula (HFNC), nasal intermittent positive pressure "
            "ventilation (NIPPV), and surfactant replacement therapy. Oxygen saturation targets (SpO2) are typically "
            "91–95% in premature infants. Apnea of prematurity (AOP) is treated with caffeine citrate. "
            "Pulse oximetry and arterial blood gas (ABG) monitoring are standard."
        ),
    },
    {
        "id": "kb_008",
        "category": "CVICU",
        "title": "CVICU — Hemodynamic Monitoring",
        "content": (
            "Hemodynamic monitoring in CVICU includes pulmonary artery catheter (PAC) measurements: "
            "cardiac output (CO), cardiac index (CI), systemic vascular resistance (SVR), and mixed venous oxygen "
            "saturation (SvO2). Near-infrared spectroscopy (NIRS) monitors cerebral and somatic oxygenation. "
            "Vasoactive medications include milrinone, dobutamine, and vasopressin."
        ),
    },
    {
        "id": "kb_009",
        "category": "ED",
        "title": "ED — Trauma and Resuscitation",
        "content": (
            "ED trauma resuscitation follows ATLS (Advanced Trauma Life Support) protocols. "
            "Primary survey: airway, breathing, circulation, disability, exposure (ABCDE). "
            "Secondary survey follows. Damage control resuscitation (DCR) uses balanced blood products. "
            "FAST exam (Focused Assessment with Sonography in Trauma) detects free fluid. "
            "GCS below 8 typically indicates need for airway protection."
        ),
    },
    {
        "id": "kb_010",
        "category": "ICU",
        "title": "ICU — Sepsis and Infection",
        "content": (
            "ICU sepsis management follows Surviving Sepsis Campaign (SSC) bundles. "
            "SIRS criteria: temperature, heart rate, respiratory rate, WBC count. "
            "Septic shock requires vasopressors to maintain MAP ≥65 mmHg and serum lactate >2 mmol/L. "
            "Broad-spectrum antibiotics (BSA) are initiated early. Procalcitonin (PCT) guides antibiotic de-escalation. "
            "Blood cultures (BC) are drawn before antibiotic administration."
        ),
    },
    {
        "id": "kb_011",
        "category": "PICU",
        "title": "PICU — Neurological Assessment",
        "content": (
            "PICU neurological assessment uses the Glasgow Coma Scale (GCS) and Pediatric GCS for younger children. "
            "Intracranial pressure (ICP) monitoring may be used in traumatic brain injury (TBI). "
            "Electroencephalogram (EEG) monitors for subclinical seizures. "
            "Cerebral perfusion pressure (CPP) = MAP - ICP, target >50 mmHg in children. "
            "Hyperosmolar therapy with mannitol or hypertonic saline reduces cerebral edema."
        ),
    },
    {
        "id": "kb_012",
        "category": "HUC",
        "title": "HUC — Documentation and Order Processing",
        "content": (
            "HUC documentation responsibilities include transcribing physician orders into the EMR, "
            "printing discharge instructions, coordinating dietary orders (NPO, clear liquids, etc.), "
            "scheduling imaging (CT, MRI, X-ray) and lab draws. "
            "Kardex maintenance and shift report preparation are standard duties. "
            "HUC must flag STAT orders and critical lab values (CLV) to the nurse immediately."
        ),
    },
    {
        "id": "kb_013",
        "category": "ICU",
        "title": "ICU — Ventilator Management",
        "content": (
            "ICU ventilator management uses lung-protective ventilation: tidal volume (Vt) 6 mL/kg IBW, "
            "plateau pressure (Pplat) <30 cmH2O, and adequate PEEP. "
            "Ventilator modes include assist-control (AC), synchronized intermittent mandatory ventilation (SIMV), "
            "and pressure support ventilation (PSV). Spontaneous breathing trials (SBT) assess readiness for extubation. "
            "Ventilator-associated pneumonia (VAP) bundles are standard of care."
        ),
    },
    {
        "id": "kb_014",
        "category": "CVICU",
        "title": "CVICU — Post-Cardiac Surgery Care",
        "content": (
            "Post-CABG care in CVICU monitors for complications: atrial fibrillation (most common arrhythmia), "
            "low cardiac output syndrome (LCOS), bleeding requiring re-exploration, and pericardial effusion. "
            "Chest tube output is closely monitored. Protamine reversal of heparin is used intraoperatively. "
            "Early mobilisation and respiratory physiotherapy reduce complications. "
            "INR and ACT guide anticoagulation management."
        ),
    },
    {
        "id": "kb_015",
        "category": "NICU",
        "title": "NICU — Infection and Pharmacology",
        "content": (
            "NICU infection prevention includes hand hygiene, central line-associated bloodstream infection (CLABSI) bundles, "
            "and cohorting of infants with RSV or other infections. "
            "Common medications: ampicillin and gentamicin for empiric sepsis coverage, fluconazole prophylaxis in VLBW infants, "
            "indomethacin or ibuprofen for patent ductus arteriosus (PDA) closure. "
            "Total daily fluid (TDF) is calculated by weight in mL/kg/day."
        ),
    },
]

# Templates for generating transcript pairs
# Each tuple: (reference_template, error_types)
CLEAN_TEMPLATES = [
    ("Patient transferred from {cat} following stabilization of {condition}.",
     lambda s: s),
    ("Doctor ordered {med} for patient in {cat} with {condition}.",
     lambda s: s),
    ("Nursing staff documented {cat} admission at zero three hundred hours.",
     lambda s: s),
    ("{cat} team performed morning rounds and updated care plan.",
     lambda s: s),
    ("Lab results reviewed by {cat} attending physician this morning.",
     lambda s: s),
]

CONDITIONS = {
    "NICU": ["bronchopulmonary dysplasia", "necrotizing enterocolitis", "apnea of prematurity", "retinopathy of prematurity"],
    "CVICU": ["atrial fibrillation", "low cardiac output syndrome", "pericardial effusion", "ventricular tachycardia"],
    "PICU": ["status epilepticus", "diabetic ketoacidosis", "acute respiratory distress syndrome", "septic shock"],
    "ICU": ["ventilator associated pneumonia", "septic shock", "acute kidney injury", "multi organ failure"],
    "HUC": ["discharge planning", "admission order processing", "critical lab value notification", "imaging coordination"],
    "ED": ["ST elevation myocardial infarction", "traumatic brain injury", "rapid sequence intubation", "anaphylaxis"],
}

MEDS = {
    "NICU": ["caffeine citrate", "surfactant", "ampicillin", "gentamicin"],
    "CVICU": ["milrinone", "norepinephrine", "vasopressin", "amiodarone"],
    "PICU": ["midazolam", "fentanyl", "mannitol", "hypertonic saline"],
    "ICU": ["norepinephrine", "vancomycin", "piperacillin tazobactam", "propofol"],
    "HUC": ["heparin flush", "normal saline", "insulin infusion", "potassium chloride"],
    "ED": ["epinephrine", "morphine", "alteplase", "adenosine"],
}


def introduce_acronym_error(text: str, category: str) -> str:
    """Replace the category acronym with a common misrecognition."""
    errors = {
        "NICU": ["NICO", "NICK YOU", "NEEKU", "NIK U"],
        "CVICU": ["SEE VIK YOU", "C V I C U", "CIVIC U", "SEE VIKU"],
        "PICU": ["PIKU", "PEE KU", "PIK YOU", "P I C U"],
        "ICU": ["I SEE YOU", "ICE YOU", "I C U"],
        "HUC": ["HUCK", "HOOK", "H U C", "HUK"],
        "ED": ["E D", "EDDY", "E. D."],
    }
    replacement = random.choice(errors.get(category, [category + "_ERR"]))
    return text.replace(category, replacement)


def introduce_partial_error(text: str, category: str) -> str:
    """Replace a condition or medication with a partial misrecognition."""
    words = text.split()
    if len(words) < 4:
        return text
    idx = random.randint(2, len(words) - 2)
    words[idx] = words[idx][:max(1, len(words[idx]) - 2)] + "..."
    return " ".join(words)


def generate_pairs() -> list[dict]:
    pairs = []
    pair_id = 1

    for category in CATEGORIES:
        conditions = CONDITIONS[category]
        meds = MEDS[category]

        # 10 clean pairs per 6 categories = 60 total (split: ~3-4 clean, 3-4 borderline, 3-4 failed)
        for i in range(10):
            condition = random.choice(conditions)
            med = random.choice(meds)
            template_fn, _ = random.choice(CLEAN_TEMPLATES)
            reference = template_fn.format(cat=category, condition=condition, med=med)

            if i < 4:
                # Clean: no errors or very minor
                recognized = reference
                wer_range = "clean"
            elif i < 7:
                # Borderline: acronym partially wrong
                recognized = introduce_acronym_error(reference, category)
                wer_range = "borderline"
            else:
                # Failed: acronym + word errors
                recognized = introduce_partial_error(
                    introduce_acronym_error(reference, category), category
                )
                wer_range = "failed"

            audio_file = f"audio_{category.lower()}_{pair_id:03d}.wav"
            pairs.append({
                "id": f"pair_{pair_id:03d}",
                "audio_file": audio_file,
                "category": category,
                "reference": reference,
                "recognized": recognized,
                "expected_wer_range": wer_range,
            })
            pair_id += 1

    # Introduce one duplicate to test dedup
    duplicate = pairs[0].copy()
    duplicate["id"] = f"pair_{pair_id:03d}"
    pairs.append(duplicate)

    return pairs


def main():
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = generate_pairs()
    kb = KB_DOCUMENTS

    output = {
        "transcript_pairs": pairs,
        "knowledge_base": kb,
        "metadata": {
            "total_pairs": len(pairs),
            "categories": CATEGORIES,
            "kb_documents": len(kb),
        },
    }

    out_path = output_dir / "synthetic_transcripts.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(pairs)} transcript pairs and {len(kb)} KB documents → {out_path}")
    by_range = {}
    for p in pairs:
        by_range.setdefault(p["expected_wer_range"], 0)
        by_range[p["expected_wer_range"]] += 1
    for k, v in by_range.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
