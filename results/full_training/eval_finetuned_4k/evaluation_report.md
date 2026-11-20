# SQuAD v1.1 Error Analysis & Evaluation Report

## 1. Overall Performance Summary
- **Total Evaluated Examples**: 10,570
- **Exact Match (EM)**: 47.05%
- **Token-Level F1 Score**: 57.42%

## 2. Prediction Breakdown
- **Exact Match**: 4,973 (47.0%)
- **Partial Match**: 2,407 (22.8%)
- **Complete Miss**: 3,190 (30.2%)

## 3. Error Analysis by Answer Length
| Answer Length (Words) | Count | Mean EM (%) | Mean F1 (%) |
|---|---|---|---|
| 1-2 words | 6,309 | 53.23% | 60.55% |
| 3-5 words | 3,039 | 45.41% | 58.71% |
| 6-10 words | 871 | 21.58% | 40.70% |
| >10 words | 351 | 13.39% | 31.60% |

## 4. Representative Qualitative Error Samples

### Partial Matches (Correct Concept, Boundary Imperfection)
- **Question**: The name of the NFL championship game is?
  - **Ground Truth**: `Super Bowl | Super Bowl | Super Bowl`
  - **Prediction**: `Roman numerals (under which the game would have been known as "Super Bowl L` (F1: 26.67%)

- **Question**: What position does Von Miller play for the Denver Broncos?
  - **Ground Truth**: `linebacker | linebacker | linebacker`
  - **Prediction**: `Denver linebacker Von Miller was named Super Bowl MVP` (F1: 20.0%)

- **Question**: What position does Von Miller play?
  - **Ground Truth**: `linebacker | linebacker | linebacker`
  - **Prediction**: `Denver linebacker Von Miller was named Super Bowl MVP` (F1: 20.0%)

### Complete Misses (Incorrect Span Selected)
- **Question**: Where did Super Bowl 50 take place?
  - **Ground Truth**: `Santa Clara, California | Levi's Stadium | Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.`
  - **Prediction**: `Super Bowl L`

- **Question**: Which NFL team won Super Bowl 50?
  - **Ground Truth**: `Denver Broncos | Denver Broncos | Denver Broncos`
  - **Prediction**: `Super Bowl L`

- **Question**: What color was used to emphasize the 50th anniversary of the Super Bowl?
  - **Ground Truth**: `gold | gold | gold`
  - **Prediction**: `Arabic numerals 50`

