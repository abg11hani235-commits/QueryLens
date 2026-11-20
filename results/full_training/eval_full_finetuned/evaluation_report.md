# SQuAD v1.1 Error Analysis & Evaluation Report

## 1. Overall Performance Summary
- **Total Evaluated Examples**: 10,570
- **Exact Match (EM)**: 77.09%
- **Token-Level F1 Score**: 85.25%

## 2. Prediction Breakdown
- **Exact Match**: 8,148 (77.1%)
- **Partial Match**: 1,520 (14.4%)
- **Complete Miss**: 902 (8.5%)

## 3. Error Analysis by Answer Length
| Answer Length (Words) | Count | Mean EM (%) | Mean F1 (%) |
|---|---|---|---|
| 1-2 words | 6,309 | 80.76% | 86.31% |
| 3-5 words | 3,039 | 76.97% | 86.15% |
| 6-10 words | 871 | 61.65% | 78.94% |
| >10 words | 351 | 50.43% | 74.18% |

## 4. Representative Qualitative Error Samples

### Partial Matches (Correct Concept, Boundary Imperfection)
- **Question**: What city did Super Bowl 50 take place in?
  - **Ground Truth**: `Santa Clara | Santa Clara | Santa Clara`
  - **Prediction**: `Santa Clara, California` (F1: 80.0%)

- **Question**: What city did Super Bowl 50 take place in?
  - **Ground Truth**: `Santa Clara | Santa Clara | Santa Clara`
  - **Prediction**: `Santa Clara, California` (F1: 80.0%)

- **Question**: The name of the NFL championship game is?
  - **Ground Truth**: `Super Bowl | Super Bowl | Super Bowl`
  - **Prediction**: `Super Bowl 50` (F1: 80.0%)

### Complete Misses (Incorrect Span Selected)
- **Question**: Who were the defending Super Bowl champions?
  - **Ground Truth**: `New England Patriots | the New England Patriots | New England Patriots`
  - **Prediction**: `Arizona Cardinals`

- **Question**: How many times have the Panthers been in the Super Bowl?
  - **Ground Truth**: `2 | second | second`
  - **Prediction**: `eight`

- **Question**: Who did Denver beat in the AFC championship?
  - **Ground Truth**: `New England Patriots | the New England Patriots | New England Patriots`
  - **Prediction**: `Arizona Cardinals`

