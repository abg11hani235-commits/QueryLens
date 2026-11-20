# SQuAD v1.1 Error Analysis & Evaluation Report

## 1. Overall Performance Summary
- **Total Evaluated Examples**: 10,570
- **Exact Match (EM)**: 79.10%
- **Token-Level F1 Score**: 86.87%

## 2. Prediction Breakdown
- **Exact Match**: 8,361 (79.1%)
- **Partial Match**: 1,397 (13.2%)
- **Complete Miss**: 812 (7.7%)

## 3. Error Analysis by Answer Length
| Answer Length (Words) | Count | Mean EM (%) | Mean F1 (%) |
|---|---|---|---|
| 1-2 words | 6,309 | 82.91% | 87.91% |
| 3-5 words | 3,039 | 78.38% | 87.57% |
| 6-10 words | 871 | 63.49% | 80.76% |
| >10 words | 351 | 55.56% | 77.20% |

## 4. Representative Qualitative Error Samples

### Partial Matches (Correct Concept, Boundary Imperfection)
- **Question**: The name of the NFL championship game is?
  - **Ground Truth**: `Super Bowl | Super Bowl | Super Bowl`
  - **Prediction**: `Super Bowl L` (F1: 80.0%)

- **Question**: Who won the Super Bowl MVP?
  - **Ground Truth**: `Von Miller | The Broncos | Miller`
  - **Prediction**: `Denver linebacker Von Miller` (F1: 66.67%)

- **Question**: Who won the MVP for the Super Bowl?
  - **Ground Truth**: `Von Miller | Von Miller | Von Miller`
  - **Prediction**: `Denver linebacker Von Miller` (F1: 66.67%)

### Complete Misses (Incorrect Span Selected)
- **Question**: What was the theme of Super Bowl 50?
  - **Ground Truth**: `"golden anniversary" | gold-themed | "golden anniversary`
  - **Prediction**: `Arabic numerals`

- **Question**: What was the theme of Super Bowl 50?
  - **Ground Truth**: `"golden anniversary" | gold-themed | gold`
  - **Prediction**: `Arabic numerals`

- **Question**: Who lost to the Broncos in the AFC Championship?
  - **Ground Truth**: `New England Patriots | the New England Patriots | New England Patriots`
  - **Prediction**: `Arizona Cardinals`

