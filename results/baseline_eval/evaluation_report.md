# SQuAD v1.1 Error Analysis & Evaluation Report

## 1. Overall Performance Summary
- **Total Evaluated Examples**: 500
- **Exact Match (EM)**: 85.40%
- **Token-Level F1 Score**: 88.91%

## 2. Prediction Breakdown
- **Exact Match**: 427 (85.4%)
- **Partial Match**: 30 (6.0%)
- **Complete Miss**: 43 (8.6%)

## 3. Error Analysis by Answer Length
| Answer Length (Words) | Count | Mean EM (%) | Mean F1 (%) |
|---|---|---|---|
| 1-2 words | 395 | 86.58% | 89.40% |
| 3-5 words | 100 | 82.00% | 87.20% |
| 6-10 words | 4 | 50.00% | 80.95% |
| >10 words | 1 | 100.00% | 100.00% |

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

