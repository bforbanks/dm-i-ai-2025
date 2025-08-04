# Medical Topic Summarization Prompt

## Task
Summarize the given medical topic document to approximately 30-50% of its original length while retaining ALL critical medical information for accurate diagnosis and treatment.

## Critical Requirements

### MUST RETAIN:
- **All diagnostic criteria** (symptoms, signs, lab values, imaging findings)
- **All treatment options** (medications, procedures, dosages)
- **All differential diagnoses** and key distinguishing features
- **All medical terminology** and synonyms
- **All critical thresholds** (vital signs, lab values, timeframes)
- **All contraindications** and warnings
- **All anatomical locations** and medical procedures

### MUST PRESERVE:
- **Medical accuracy** - no factual errors or omissions
- **Clinical relevance** - information needed for emergency decision-making
- **Terminology consistency** - use standard medical terms
- **Logical flow** - maintain cause-and-effect relationships
- **Completeness** - no topic should be completely removed

### CAN REMOVE:
- **Redundant explanations** of the same concept
- **Excessive background information** not directly relevant to diagnosis/treatment
- **Repetitive examples** of the same symptom
- **Verbose descriptions** that can be stated more concisely
- **Historical context** not essential for current practice

## Format Guidelines

### Structure:
1. **Definition/Overview** (1-2 sentences)
2. **Key Symptoms & Signs** (bullet points)
3. **Diagnostic Criteria** (specific thresholds and findings)
4. **Differential Diagnosis** (key alternatives)
5. **Treatment Options** (immediate and definitive)
6. **Critical Considerations** (warnings, contraindications)

### Style:
- Use **concise, direct language**
- Maintain **medical precision**
- Include **specific values** (not "elevated" but ">1000")
- Use **bullet points** for lists
- Keep **sentences short** but complete

## Example Transformation

### Before (Original):
```
Acute Myocardial Infarction (AMI) represents a critical cardiac emergency characterized by the sudden interruption of blood flow to a portion of the heart muscle, typically resulting from the complete occlusion of a coronary artery due to atherosclerotic plaque rupture and subsequent thrombus formation. This condition manifests through a constellation of clinical symptoms including severe, crushing chest pain that may radiate to the left arm, jaw, or back, often described as a sensation of pressure or heaviness that persists for more than 20 minutes and is not relieved by rest or nitroglycerin administration.
```

### After (Summarized):
```
Acute Myocardial Infarction (AMI): Sudden coronary artery occlusion causing heart muscle damage.

**Key Symptoms:**
- Severe crushing chest pain (>20 min, unrelieved by rest/nitroglycerin)
- Pain radiating to left arm, jaw, back
- Pressure/heaviness sensation

**Diagnostic Criteria:**
- ECG: ST elevation ≥1mm in 2+ contiguous leads
- Troponin: Elevated (I >0.04 ng/mL, T >0.01 ng/mL)
- Symptoms + ECG changes + biomarkers
```

## Quality Checklist

Before finalizing each summary, verify:
- [ ] All diagnostic criteria preserved
- [ ] All treatment options included
- [ ] All critical thresholds maintained
- [ ] Medical terminology consistent
- [ ] No factual errors introduced
- [ ] Length reduced by 30-50%
- [ ] Clinical decision-making information intact

## Special Considerations

### BM25 Optimization:
- Retain diverse medical terminology and synonyms
- Include alternative phrasings for symptoms
- Maintain medical concept relationships

### Consistency:
- Use consistent formatting across all topics
- Maintain similar level of detail
- Follow same structural pattern 