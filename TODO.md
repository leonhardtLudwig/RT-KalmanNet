# TODO 

# robust_kalman.py



Hai già una baseline molto utile: ora faccio una diagnosi strutturata “prestazioni vs costo” e ti elenco priorità concrete (cosa misurare prima, cosa toccare dopo) con riferimenti puntuali ai punti del codice da osservare.

Searched for text `def fnComputeTheta` (`**/CODE/RT_KFNET/RobustKalmanPY/robust_kalman.py`), 5 results

Searched for text `for _ in range(50):` (`**/CODE/RT_KFNET/RobustKalmanPY/robust_kalman.py`), 2 results

Searched for text `torch.linalg.solve` (`**/CODE/RT_KFNET/RobustKalmanPY/robust_kalman.py`), 12 results

Searched for text `torch.linalg.eigvalsh` (`**/CODE/RT_KFNET/RobustKalmanPY/robust_kalman.py`), 2 results

Searched for text `avg_loss.backward` (`**/CODE/RT_KFNET/main_Robust_KNet.ipynb`), 1 result

Ho verificato che il segno in `g_fn` è già corretto e ho localizzato i punti più costosi (`solve`, bisezione, loop per-sequenza). Adesso ti lascio una checklist prioritaria, concreta e ordinata per iniziare a migliorare sia accuracy sia tempo.Perfetto, adesso hai un baseline solido e coerente coi paper.  
I numeri che hai ottenuto dicono chiaramente una cosa: **RT-KNet non sta ancora “usando” bene la parte neural**, e il costo computazionale è dominato dal filtro robusto, non dalla GRU.

## Cosa guardare subito (ordine di priorita)

1. **Ablation “rete spenta vs rete accesa”**
- Obiettivo: capire se la GRU sta davvero imparando.
- Fai 3 run identiche:
  1. REKF con `c` fisso (`1e-3`)
  2. RT-KNet con GRU ma `self.c` forzato a costante (bypass NN)
  3. RT-KNet normale
- Se (2) ≈ (3), la rete non sta portando informazione utile.
- Se (3) migliora ma poco, il problema è capacity/training/features.

2. **Distribuzione temporale di `c_t`**
- Hai già il plot: ora devi misurare statistiche.
- Da estrarre su train/CV/test:
  - media, std, min, max di `c_t`
  - autocorrelazione e varianza nel tempo
  - correlazione tra `c_t` e innovazione $\|y_t-h(\hat x_t)\|$
- Se `c_t` è quasi costante, la rete è collassata su una policy fissa.

3. **Profiling del tempo dentro `fnREKF`**
- Il collo di bottiglia è quasi certamente qui: robust_kalman.py
- Misura tempo medio per step per:
  - Jacobiane (`fnComputeJacobianF/H`)
  - `fnComputeTheta` (bisezione + Newton)
  - update covariance (i vari `torch.linalg.solve`)
- Senza profiling fine, ogni ottimizzazione è “a caso”.

4. **Costo di `fnComputeTheta`: numero iterazioni bisezione**
- Ora hai 50 iterazioni fisse.
- Prova subito:
  - 50 → 30 → 20 iterazioni
  - valuta impatto su MSE e tempo
- Se MSE resta quasi uguale, hai guadagno netto immediato.

5. **Feature relevance per la GRU (`input_feat_mode`)**
- Ora usi mode 3 (`F1,F2,F3,F4`).
- Testa sistematicamente 0,1,2,3 a parità di seed.
- Spesso ridurre feature rumorose migliora sia accuratezza sia stabilità del training.

6. **Loss mismatch rispetto all’obiettivo robusto**
- Stai ottimizzando solo MSE stato finale.
- Ma la rete controlla `c_t` (parametro robustezza): potrebbe non essere ben identificato.
- Da provare:
  - loss con termine regolarizzazione su variazione di `c_t` (smoothness)
  - loss pesata nel tempo (più peso ai transienti)
- Se no, la rete tende a imparare una scorciatoia “quasi costante”.

7. **Stabilità training della GRU**
- Logga ogni epoch:
  - norma gradienti GRU
  - distribuzione output pre-softplus (`c_raw`)
- Possibili sintomi:
  - gradienti molto piccoli → rete non impara
  - `c_raw` troppo negativo → softplus quasi zero
- Contromisure: gradient clipping, lr sweep (`1e-3`, `5e-4`, `1e-4`), wd più basso.

8. **Confronto corretto di costo con KalmanNet**
- È normale che RT-KNet sia molto più lento: risolve online un filtro robusto completo.
- Però 15x è alto: serve separare
  - costo del filtro base
  - costo extra `theta` robusto
  - costo rete
- Così capisci dove intervenire per ridurre da 15x a, ad esempio, 5x.

9. **Riduzione overhead Python nei loop**
- Nel notebook fai moltissimi loop annidati per sequenza.
- Valuta:
  - meno `DataGen` ripetuti
  - meno conversioni/allocazioni nel loop
  - preallocazioni e riuso tensori dove possibile
- Anche senza riscrittura full vectorized, puoi ridurre parecchio overhead.

10. **Target realistico di performance**
- Con la struttura attuale, aspettarsi di battere KalmanNet in tempo puro è improbabile.
- Obiettivo realistico:
  - ridurre gap MSE RT-KNet vs KNet
  - mantenere vantaggio RT-KNet in scenari con outlier/mismatch
- Quindi serve benchmark “stress test robustezza”, non solo dataset nominale.

---

## Piano pratico “prima settimana” (minimo indispensabile)

1. Ablation rete accesa/spenta + statistiche `c_t`.
2. Profiling tempi per blocco in `fnREKF`.
3. Sweep rapido bisezione (50/30/20 iter).
4. Sweep `input_feat_mode` (0/1/2/3).
5. Sweep `lr` + gradient clipping.

Con questi 5 passi saprai esattamente:
- se il collo di bottiglia è matematico o implementativo,
- se la rete sta imparando davvero,
- dove hai il miglior trade-off MSE/tempo.

Se vuoi, nel prossimo step ti preparo una **tabella esperimenti pronta da copiare** (configurazioni, metriche da loggare, criterio decisionale) così parti subito in modo strutturato.



## RT_KalmanNet_nn.py


## main_Robust_KNet.ipynb