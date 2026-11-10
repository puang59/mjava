## Overall

| system | n | ppl | 1st compile | exact | conformance | compile+conform | success@0 | success@1 | success@2 | success@3 | mean attempts | final conformance | final compile+conform | repair compile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retrieval baseline | 600 | - | 85.8% | 4.2% | 28.5% | 7.3% | 85.8% | 85.8% | 85.8% | 85.8% | 1.00 | 28.5% | 7.3% | - |
| supervised | 600 | 1.174 | 83.7% | 69.8% | 93.2% | 71.8% | 83.7% | 86.5% | 87.3% | 87.5% | 1.42 | 93.1% | 73.0% | 82.8% |
| supervised+RL | 600 | 1.147 | 83.7% | 70.0% | 93.0% | 72.0% | 83.7% | 85.5% | 86.3% | 87.2% | 1.44 | 93.1% | 73.2% | 82.0% |

## By task family (compile-and-conformant rate: first attempt / after the correction loop)

| family (n) | retrieval baseline: 1st / final compile+conform | supervised: 1st / final compile+conform | supervised+RL: 1st / final compile+conform |
|---|---:|---:|---:|
| abstract_class (37) | 5.4% / 5.4% | 64.9% / 64.9% | 64.9% / 64.9% |
| class_methods (103) | 0.0% / 0.0% | 59.2% / 60.2% | 57.3% / 60.2% |
| data_class (95) | 0.0% / 0.0% | 72.6% / 73.7% | 72.6% / 72.6% |
| enum (51) | 21.6% / 21.6% | 68.6% / 70.6% | 70.6% / 70.6% |
| inheritance (55) | 0.0% / 0.0% | 60.0% / 63.6% | 60.0% / 63.6% |
| interface_impl (39) | 0.0% / 0.0% | 71.8% / 71.8% | 74.4% / 74.4% |
| main_program (55) | 49.1% / 49.1% | 92.7% / 94.5% | 92.7% / 94.5% |
| multi_instruction (92) | 0.0% / 0.0% | 70.7% / 70.7% | 71.7% / 72.8% |
| static_utils (73) | 5.5% / 5.5% | 89.0% / 90.4% | 89.0% / 89.0% |

## Held-out identifier subset

| subset (n) | retrieval baseline: 1st compile / 1st conform / final compile+conform | supervised: 1st compile / 1st conform / final compile+conform | supervised+RL: 1st compile / 1st conform / final compile+conform |
|---|---:|---:|---:|
| seen identifiers (287) | 87.1% / 30.8% / 9.8% | 97.9% / 98.7% / 92.3% | 98.3% / 98.8% / 93.0% |
| held-out identifiers (313) | 84.7% / 26.4% / 5.1% | 70.6% / 88.1% / 55.3% | 70.3% / 87.7% / 55.0% |

## Notes

- retrieval baseline: no repair ability: the correction block repeats the first attempt; perplexity and repair are n/a
