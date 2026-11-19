# GTA VI Trophy Rarity Forecast

Forecast tiap trophy GTA VI (PS5) di Exophase pada 2026-11-26T00:00:00Z — dibuat untuk bounty make.fun gta6-trophy-rarity-forecast-01.

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Maranatha99/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Maranatha99/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

## Cara menjalankan

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, stdlib saja, tanpa network) membaca tiap `<id>.in`
di direktori input dan menulis `<id>.out` berisi lima kuantil (5%, 25%,
50%, 75%, 95%) dari earn rate tiap trophy pada target case.

## Model

Dua model median per trophy, diblend di skala logit:

1. **Tren logit** — least-squares `logit(rate)` terhadap
   `ln(1 + jam sejak 2026-11-19T00:00Z)` atas 48 jam read terakhir,
   slope di-shrink ke median slope tier-nya (platinum/gold/silver/bronze).
2. **Model jumlah earner** — jumlah earner dan jumlah pemain dilacak
   masing-masing difit log-linear lalu diproyeksikan ke target dan dibagi,
   sehingga dilusi pemain baru dan progres pemain lama dipisah. Ini penting
   karena Exophase me-refresh hitungannya dalam lompatan: refresh yang
   melipatgandakan pemain terlacak mendilusi trophy akhir game jauh lebih
   besar daripada trophy pembuka yang sudah dipegang semua orang.

Spread (logit) menggabungkan random walk `0.5·√hari`, residual fit,
sebaran lompatan logit antar-read (refresh shock), dan floor yang tumbuh
dengan horizon — jadi histori launch night yang datar sekalipun tetap
mengakui refresh bisa menggeser semua rate. Kalau histori hanya berisi
satu state (halaman belum pernah refresh), median digeser turun 0,30 logit,
kecuali trophy yang sudah 100%.

## Uji lokal

```sh
python3 ~/workspace/makefun/gta6-trophy-check.py <kit>/input <out-dir>
```

Checker di luar repo ini memvalidasi format output dan menghitung pinball
loss terhadap referensi kewarasan (persistensi last-rate dan dua skenario
refresh sintetis dari total refresh 04:30Z yang tercatat di rules).
Pada practice: loss ~0,07–0,08 vs skenario refresh (baseline house ~0,20–0,22),
coverage 90% = 0,98 (baseline 0,00–0,04). Deterministik (dua run byte-identik),
runtime ~0,1 detik per case, output ~2,8 KB.
