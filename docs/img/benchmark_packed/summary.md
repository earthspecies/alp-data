| dataset | rows | sequential packed / live (ms) | best packed | best live | speed-up at best |
|---|---|---|---|---|---|
| beans | 62,415 | 105 / 125 | 1029/s (48w, pf 2) | 497/s | 2.1x |
| audioset | 632,257 | 121 / 151 | 431/s (48w, pf 8) | 455/s | 0.9x |
| InfantMarmosetsVox | 72,921 | 88 / 467 | 1393/s (48w, pf 8) | 81/s | 17.1x |
| dclde2026 | 1,336 | 37 / 511 | 27/s (0w, pf 0) | 2/s | 10.8x |

| dataset | workers | prefetch | packed /s | live /s | speed-up |
|---|---|---|---|---|---|
| InfantMarmosetsVox | 0 | 0 | 20 | 3 | 7.49 |
| InfantMarmosetsVox | 4 | 2 | 100 | 8 | 12.91 |
| InfantMarmosetsVox | 4 | 8 | 101 | 8 | 12.97 |
| InfantMarmosetsVox | 16 | 2 | 447 | 37 | 12.25 |
| InfantMarmosetsVox | 16 | 8 | 467 | 34 | 13.85 |
| InfantMarmosetsVox | 48 | 2 | 1330 | 81 | 16.44 |
| InfantMarmosetsVox | 48 | 8 | 1393 | 81 | 17.13 |
| audioset | 0 | 0 | 8 | 8 | 1.09 |
| audioset | 4 | 2 | 28 | 33 | 0.85 |
| audioset | 4 | 8 | 31 | 38 | 0.81 |
| audioset | 16 | 2 | 120 | 135 | 0.89 |
| audioset | 16 | 8 | 125 | 153 | 0.81 |
| audioset | 48 | 2 | 409 | 436 | 0.94 |
| audioset | 48 | 8 | 431 | 455 | 0.95 |
| beans | 0 | 0 | 11 | 9 | 1.20 |
| beans | 4 | 2 | 49 | 32 | 1.52 |
| beans | 4 | 8 | 59 | 38 | 1.55 |
| beans | 16 | 2 | 285 | 145 | 1.96 |
| beans | 16 | 8 | 346 | 173 | 2.00 |
| beans | 48 | 2 | 1029 | 497 | 2.07 |
| beans | 48 | 8 | 893 | 600 | 1.49 |
| dclde2026 | 0 | 0 | 27 | 2 | 10.76 |
