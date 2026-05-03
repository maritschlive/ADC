# Streitplan: Naechste Schritte nach der Run-Analyse

Stand: 2026-04-24

## Ziel
- Erfolgreiche Presets sichern.
- Fehlgeschlagene Presets mit minimalem GPU-Verlust nachziehen.
- Speicherprobleme und Node-Probleme vorab eliminieren.
- Monitoring sauber ueber TensorBoard und SLURM-Logs halten.

## 1) Sofortmassnahmen (heute)
1. Erfolgreiche Runs nicht anfassen:
   - `scratch` (20000)
   - `scratch_unlocked` (10000)
   - `paper_faithful_scratch` (24000)
2. Doppelte Checkpoints in diesen Runs reduzieren:
   - pro Run nur `last.ckpt` behalten
   - `epoch=...-step=...ckpt` loeschen, wenn identischer Trainingsstand
3. Alte Root-Logs nicht weiter vermehren:
   - neue Jobs laufen jetzt nach `logs/*.out` und `logs/*.err`

## 2) Preflight vor jedem neuen Submit
1. Genug freier Speicher sicherstellen (kritisch wegen vorherigem `No space left on device`).
2. `setup` muss erfolgreich sein (uv vorhanden, weights vorhanden).
3. Keine RUN_TAG-Sweeps fuer abhaengige Presets verwenden.
   - Grund: Chain-Resolver sucht `runs/<preset>/*/checkpoints/last.ckpt`.
   - Bei getaggten Ordnern (`runs/<preset>_<tag>/...`) brechen abhaengige Presets mit `FileNotFoundError`.
4. Wenn moeglich problematische Nodes mit altem Treiber vermeiden
   - ein Run ist an `NVIDIA driver ... too old` gescheitert.

## 3) Prioritaetsreihenfolge fuer Retries
1. `scratch_stage2`
   - lief vorher nicht wegen fehlendem Quell-Checkpoint.
   - jetzt nur ohne RUN_TAG starten.
2. `polyp_transfer`
   - zuerst erneut sauber durchlaufen lassen (war wegen Speicherplatz abgebrochen).
3. `polyp_unlocked`
   - erst nach erfolgreichem `polyp_transfer`.
4. `polyp_stage2`
   - parallel erst sinnvoll, wenn `polyp_transfer` stabil abgeschlossen ist.
5. `polyp_stage2_from_unlocked`
   - erst nach erfolgreichem `polyp_unlocked`.
6. `paper_faithful_polyp`
   - zuletzt auf passendem Node ohne Treiberkonflikt wiederholen.

## 4) Abnahmekriterien pro Run
Ein Run gilt nur dann als "fertig", wenn alle Punkte erfuellt sind:
1. `.out`-Summary meldet `Exit: 0`.
2. `.err` enthaelt `Trainer.fit stopped: max_steps=... reached`.
3. Ziel-Run hat `checkpoints/last.ckpt`.
4. Keine harten Fehler in `.err`:
   - `command not found`
   - `FileNotFoundError`
   - `No space left on device`
   - `NVIDIA driver ... too old`

## 5) Monitoring-Plan
1. TensorBoard lokal auf `runs/` verwenden.
2. SLURM-Live-Logs aus `logs/` verfolgen.
3. Nach jedem abgeschlossenen Preset kurz pruefen:
   - Exit-Code
   - Checkpoint vorhanden
   - Disk-Nutzung stabil

## 6) Abschlussziel
Wenn alle Retries durch sind, soll folgender Zustand erreicht sein:
1. Alle benoetigten Presets haben `Exit: 0`.
2. Jede Chain-Abhaengigkeit ist aufloesbar.
3. Keine unnötigen 10-30 GB Duplikat-Checkpoints mehr.
4. TensorBoard ist fuer alle relevanten Runs lokal sichtbar.
