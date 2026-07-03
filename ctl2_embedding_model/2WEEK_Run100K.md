# Notas del primer smoke run — NGT Linformer NodesAligned

- **Notebook**: [`NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb`](NGT_Linformer_NodesAligned_FullPrecision_Collide2V_L1features.ipynb)
- **Contexto**: deliverables del [meeting del 29/06](https://codimd.web.cern.ch/s/oEeoJz7QJ#Meeting-2906) — modelo Linformer full-precision alineado con `ctl2_model`, visualización 1D pre/post preprocesado, y entrenamiento + anomaly score Mahalanobis sobre Collide-2V (MinBias, top, QCD, DY).
- **Tipo de run**: smoke test / sanity check local, **no** el run de producción pedido en el meeting.

## 1. Config del run

- `N_EVENTS_PER_CLASS = 1000` (modo "quick local testing" — future run NGT `100_000`).
- 4000 eventos totales (1000 × 4 clases) → split 70/15/15 estratificado: 2800 train / 600 val / 600 eval.
- Modelo: Linformer full-precision, `ff_dim=32`, `latent_dim=12`, `num_heads=2`, `proj_k=8` → 18.6k parámetros.
- 30 ephocs, `AdamW`, `lr=1e-3`, `weight_decay=1e-4`, `batch_size=512`.
- `USE_AUGMENTATION = False` (baseline sin rotación en phi).

## 2. Verificación de alineación con `nodes.py`

- **Arquitectura**: estructuralmente idéntica a [`linformer.py`](ctl2_model/src/ctl2_model/models/linformer.py) — `input_proj → attn1 → +res → attn2 → +res → sum/N → 2×Dense(ReLU) → latent → cls_hidden → logits`, con `QDense→nn.Linear`, `QAdd→+`, `QSum(axes=1)/N→sum(dim=1)/N` como análogos full-precision correctos.
- **Orden de preprocesado**: idéntico a [`data_processing/nodes.py`](ctl2_model/src/ctl2_model/pipelines/data_processing/nodes.py) — split → `compute_normalization_constants(raw[train_idx])` → `build_pid_mapping(raw[train_idx])` → `preprocess_dataset(..., add_pid_ohe=True)`.
- **Hallazgo puntual — `compute_normalization_constants`**: el notebook enmascara `pt>0` antes de tomar min/max; el repo (`preprocessing.py`) no enmascara. En teoría esto podría divergir si hay padding con ceros. Se verificó ejecutando ambas versiones sobre los `.npy` cacheados del run real:
  - `pad_fraction = 0.0` en las 4 clases (Collide-1M siempre trae ≥100 candidatos por evento; el loader trunca a 100, nunca rellena).
  - Con 0% padding, ambas versiones dan **exactamente los mismos 6 valores** (`pt_min/max`, `eta_min/max`, `dxy_min/max`, diff=0.0) y el mismo `pt_norm` en todo el dataset (delta=0.0).
  - **Conclusión**: para este dataset la normalización es numéricamente idéntica a `nodes.py`, no solo estructuralmente similar.
  - **Riesgo latente**: si algún dataset futuro trajera eventos con menos de 100 candidatos válidos, esta divergencia de código sí se activaría (el repo daría `pt_min=0` por el padding; el notebook no). Vale la pena unificar la lógica igualmente para evitar sorpresas.

## 3. Curva de entrenamiento

- Épocas 1–4: estancado en accuracy ≈ 0.25 (nivel random para 4 clases).
- Épocas 5–12: aprendizaje rápido, sube a ≈ 0.63 val_acc.
- Épocas 12–30: plateau en 0.64–0.67 val_acc, sin mejora sostenida.
- Lectura: capacidad/datos agotados para un modelo de 18.6k parámetros con solo 2800 eventos de train — comportamiento esperable en un smoke test, no necesariamente indicativo del run completo.

## 4. Resultados de clasificación (eval, 600 eventos)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| top | 0.84 | 0.93 | **0.88** (mejor) |
| qcd | 0.75 | 0.73 | 0.74 |
| minbias | 0.45 | 0.70 | 0.55 |
| dy | 0.46 | 0.17 | **0.25** (peor) |

- Accuracy global: 0.63.
- Matriz de confusión: **113/150 eventos DY se clasifican como MinBias** — la confusión dominante del run. `top` y `qcd` se separan bien entre sí y del resto.

## 5. Mahalanobis anomaly score (normal class = MinBias, PCA a 12 dims)

| Clase | mean | std | mediana (p50) |
|---|---|---|---|
| minbias | -38.7 | 12.8 | -42.7 |
| top | 3.66M | 1.0M | 3.91M |
| qcd | 786k | 1.4M | 3994 |
| dy | 28k | 212k | -42.6 |

- `top` se separa muy limpiamente de MinBias (score alto y consistente) → el anomaly score funciona bien para esta clase.
- `qcd` y `dy` son bimodales: mediana cercana a MinBias (coherente con la confusión de la sección 4) pero con una cola de eventos con scores enormes que arrastra la media muy por encima de la mediana.
- Posible causa: inestabilidad numérica en la inversión de la covarianza empírica en espacio latente de 12 dims con solo ~700 eventos MinBias de entrenamiento (dimensión relativamente alta respecto al tamaño de muestra). A vigilar al escalar.

## 6. Brecha respecto al deliverable del meeting

Este run confirma que el pipeline funciona **end-to-end** (carga → preprocesado → entrenamiento → embeddings → anomaly score), pero **no** cumple todavía la tarea 3 del meeting: falta ejecutar el run de producción con 100k eventos/clase (400k total) en NGT.

## 7. Aprendizajes / próximos pasos

- Confirmar si la confusión DY↔MinBias persiste con 100x más datos o es artefacto de muestra pequeña.
- Antes de escalar a las categorías adicionales (W+jet, WW, WZ, ttW, ttZ, señales Higgs), validar la separabilidad de las 4 clases actuales con el run completo.
- Vigilar la posible inestabilidad numérica del Mahalanobis score al escalar (revisar condicionamiento de la covarianza empírica con más datos de MinBias).
- Unificar `compute_normalization_constants` con la versión sin máscara del repo por consistencia, aunque no cambie el resultado con los datos actuales.

## 8. Run de producción — 100k eventos/clase (NGT)

- **Run**: `n100000_142733` (2026-07-01), outputs en [`ngt_linformer_nodes_aligned_outputs/runs/2026-07-01/n100000_142733/`](ngt_linformer_nodes_aligned_outputs/runs/2026-07-01/n100000_142733/).
- **Tipo de run**: run de producción pedido en el meeting del 29/06 (tarea 3) — mismo notebook, mismo modelo (18,672 parámetros, `ff_dim=32`, `latent_dim=12`, `num_heads=2`, `proj_k=8`), misma config de entrenamiento (30 épocas, `AdamW`, `lr=1e-3`, `weight_decay=1e-4`, `batch_size=512`, `USE_AUGMENTATION=False`).

### 8.1 Config del run

- `N_EVENTS_PER_CLASS = 100_000` → 400,000 eventos totales (100k × 4 clases) → split 70/15/15 estratificado: 280,000 train / 60,000 val / 60,000 eval (15,000 eval por clase).
- Normalización (`compute_normalization_constants` sobre train): `pt_min=0.051, pt_max=7.168` (log1p), `eta_min=-5.0, eta_max=2.400`, `dxy_min=-51.87, dxy_max=51.24`.
- `pid_mapping` con 19 valores únicos de PID → `X_all` termina con 23 features por partícula (4 cinemáticas + 19 PID one-hot), igual estructura que el run de 1k.

### 8.2 Curva de entrenamiento

- Convergencia mucho más rápida que el smoke test: época 1 ya alcanza `val_acc=0.654` (el run de 1k necesitó ~9 épocas para salir del plateau random de 0.25).
- Épocas 1–18: subida sostenida hasta `val_acc≈0.706`.
- Épocas 18–30: plateau en 0.695–0.706, `val_loss≈0.56–0.57` — comportamiento similar al de 1k (satura), pero en un nivel de accuracy más alto.
- Con más datos el modelo de 18.6k parámetros deja de estar limitado por tamaño de muestra tan temprano como en el smoke test, pero el plateau final sugiere que ahora el techo lo pone la capacidad del modelo (o la separabilidad de clases), no la cantidad de datos.

### 8.3 Resultados de clasificación (eval, 60,000 eventos)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| top | 0.93 | 0.95 | **0.94** (mejor) |
| qcd | 0.83 | 0.83 | 0.83 |
| minbias | 0.50 | 0.86 | 0.63 |
| dy | 0.67 | 0.16 | **0.26** (peor) |

- Accuracy global: 0.70 (macro F1 0.67, weighted F1 0.67).
- Matriz de confusión: **11,674/15,000 eventos DY se clasifican como MinBias** (77.8%) — la misma confusión dominante del smoke test, y en términos relativos incluso un poco más marcada que en 1k (75.3%). La precisión de DY mejora (0.46→0.67) porque hay más eventos DY correctamente aislados en términos absolutos, pero el recall sigue muy bajo (0.16, prácticamente igual al 0.17 del run de 1k).
- `top` y `qcd` siguen separándose bien entre sí y del resto, y mejoran su F1 respecto al smoke test (top 0.88→0.94, qcd 0.74→0.83).
- El PCA 2D de los embeddings (`figures/embedding_pca_eval.png`) muestra visualmente el mismo patrón: `top` forma un lóbulo claramente separado (PCA1 alto), `qcd` ocupa una región intermedia que se solapa parcialmente con `top`, y `minbias`/`dy` quedan superpuestos casi por completo en la misma región (PCA1 bajo) — consistente con la confusión dominante de la matriz.

### 8.4 Mahalanobis anomaly score (normal class = MinBias, PCA a 12 dims)

| Clase | mean | std | mediana (p50) |
|---|---|---|---|
| minbias | -17.6 | 30.2 | -21.6 |
| top | 48,659 | 36,827 | 41,902 |
| qcd | 1,463 | 6,818 | 48.9 |
| dy | 152 | 1,192 | -21.7 |

- `top` se sigue separando muy limpiamente de MinBias.
- `qcd` sigue siendo bimodal (mediana ≈49, muy por debajo de la media 1,463), pero mucho menos extremo que en el smoke test (mean/median ratio bajó de ~197x a ~30x).
- `dy` tiene mediana (-21.7) prácticamente igual a la de MinBias (-21.6) — confirma numéricamente la confusión de la sección 8.3: la mitad "típica" de los eventos DY es indistinguible de MinBias en el espacio latente, y solo una cola minoritaria de eventos DY produce scores altos.
- La escala absoluta de los scores no es comparable 1:1 entre el run de 1k y el de 100k (dominan la covarianza empírica del espacio latente en cada run, que depende del tamaño de muestra usado para ajustarla) — lo relevante es la forma relativa de las distribuciones, no el valor absoluto.
- El histograma de scores (`figures/mahalanobis_scores_eval.png`) sigue siendo difícil de leer directamente por los outliers de `top`/`qcd` que estiran el eje x — sería útil replottear con escala log o recortando percentiles altos en una futura iteración.

### 8.5 Lectura

Este run confirma que el pipeline funciona a escala de producción y cumple la tarea 3 del meeting del 29/06 (400k eventos en NGT). La escala mejora accuracy global (0.63→0.70) y separa mejor `top`/`qcd`, pero **no resuelve** el problema estructural DY↔MinBias identificado en el smoke test — persiste con magnitud muy similar, lo que sugiere que es una limitación de las features/arquitectura actuales (no de tamaño de muestra).

## 9. Comparación 1k vs 100k

| Métrica | 1k/clase (smoke test) | 100k/clase (producción) |
|---|---|---|
| Eventos totales | 4,000 | 400,000 |
| Eventos eval | 600 (150/clase) | 60,000 (15,000/clase) |
| Accuracy global | 0.63 | 0.70 |
| Macro F1 | 0.60 | 0.67 |
| F1 top | 0.88 | 0.94 |
| F1 qcd | 0.74 | 0.83 |
| F1 minbias | 0.55 | 0.63 |
| F1 dy | 0.25 | 0.26 |
| DY→MinBias (confusión) | 113/150 (75.3%) | 11,674/15,000 (77.8%) |
| Mahalanobis qcd mean/median ratio | ~197x | ~30x |

**Respuesta a las preguntas abiertas de la sección 7:**

- *¿La confusión DY↔MinBias persiste con 100x más datos?* Sí, casi sin cambios (75.3%→77.8%). No es un artefacto de muestra pequeña — es un límite real del modelo/features actuales para separar estas dos clases.
- *¿La inestabilidad numérica del Mahalanobis se estabiliza al escalar?* Sí, claramente: el ratio mean/median de `qcd` (el indicador más directo de la bimodalidad/outliers) baja de ~197x a ~30x, y los std relativos son más razonables. Confirma la hipótesis de la sección 5: la inestabilidad del smoke test era en gran parte un artefacto de ajustar la covarianza empírica en 12 dims con solo ~700 eventos MinBias de train.

## 10. Próximos pasos

- **Actualización 02/07:** el notebook ahora incluye un diagnóstico explícito DY↔MinBias con features agregadas por evento (`n_lep`, `n_gamma`, `sum_pt_lep`, `max_ossf_mass`, etc.). En el run local sobre las cachés de 100k, un `HistGradientBoostingClassifier` sobre estas variables alcanza solo **ROC AUC ≈ 0.609** y accuracy ≈0.57, lo que apunta a separabilidad de entrada débil con las L1 features actuales, no solo a falta de capacidad del clasificador.
- Se añadió el preset `dy_minbias_wide_attn_aug`: `latent_dim=24`, `ff_dim=64`, `num_heads=4`, `proj_k=16`, pooling por atención y `USE_AUGMENTATION=True` con concatenación de original + rotación en phi. `baseline_no_aug` reproduce el run anterior.
- `figures/mahalanobis_scores_eval.png` fue regenerado con ventana p95 enfocada en `minbias`/`dy`; el rango completo queda en `figures/mahalanobis_scores_eval_logy_full_range.png`.

- El run de 100k cumple el deliverable de producción del meeting del 29/06; el foco ahora debería pasar de "escalar datos" a **resolver la separabilidad DY↔MinBias** — por ejemplo, revisar si faltan features discriminantes entre estas dos clases, probar `latent_dim`/`ff_dim` mayores, o evaluar si activar `USE_AUGMENTATION` ayuda.
- Antes de sumar las categorías adicionales (W+jet, WW, WZ, ttW, ttZ, señales Higgs), vale la pena resolver o al menos entender mejor la confusión DY↔MinBias en las 4 clases actuales, ya que probablemente se repita con clases de fondo similares.
- Replotear `mahalanobis_scores_eval.png` con escala log o recorte de percentiles altos para que la distribución de `minbias`/`dy` (que hoy queda aplastada contra el eje) sea legible.
- Unificar `compute_normalization_constants` con la versión sin máscara del repo (pendiente desde la sección 2), aunque el hallazgo de padding=0% también se cumple en el dataset de 100k.
