# Balle vidéo : détection RF-DETR puis fusion avec l'ATD

Chaîne complète, de la vidéo tactique à `balle_fusionnee.npz` (utilisé par l'étage 1 et l'étage 2).

Versions figées : étapes 1 et 3 (CPU) `possession/requirements_ball_cpu.txt` ; étape 2 (GPU) `possession/requirements_ball_gpu.txt`.

| Étape | Fichier | Commande |
|---|---|---|
| 1. Balle ATD projetée en pixels (centres des recadrages) | `atd_ball_px.npy` | `python possession/atd_ball_px.py --data data --out atd_ball_px.npy` |
| 2. Détection RF-DETR (GPU) | `rfdetr_parts/part_*.npz` (32 fichiers : `frame, src, u, v, conf`) | voir ci-dessous |
| 3. Fusion vidéo + ATD (CPU, ~20 s) | `balle_fusionnee.npz` | `python possession/video_ball.py --parts outputs/intermediaires/rfdetr_parts --data data --out balle_fusionnee.npz` |

## Ce qui est vérifié, et ce qui ne l'est pas

| Étape | État | Détail |
|---|---|---|
| 1 | **vérifiée** | redonne exactement `atd_ball_px.npy` |
| 2 | **non rejouée** | la détection n'a été exécutée qu'une fois ; les 32 fichiers de `rfdetr_parts/` sont fournis comme fichiers intermédiaires (autorisé par le règlement) et font foi |
| 3 | **vérifiée** | à partir des `rfdetr_parts/` fournis, redonne `balle_fusionnee.npz` : `x`, `y`, `xs`, `ys`, `source`, `video_x`, `video_y`, `video_conf`, `atd_x`, `hotspots`, `blacklist_cells`, `params` identiques |

Limites connues :
- **Étape 2, pas garantie au bit près.** Sur un autre GPU, ou selon la compilation du modèle (`inference(compile=True)`),
  les confiances peuvent varier légèrement, ce qui peut changer le chemin choisi à l'étape 3. Pour retrouver exactement
  nos sorties, partir des `rfdetr_parts/` fournis.
- **Étape 3, écarts d'arrondi sur trois tableaux.** Revérifiée le 2026-11-13 avec les versions de
  `possession/requirements_ball_cpu.txt` : `vx`, `vy` et `atd_y` diffèrent du fichier fourni d'au plus 4·10⁻⁶
  (arrondi flottant, la version de numpy / scipy de l'exécution d'origine n'ayant pas été notée). Tout le reste est identique.

## Étape 2 : détection
- Modèle : **RF-DETR Medium** pré-entraîné COCO (Roboflow, licence Apache-2.0), utilisé tel quel, classe 37 « sports ball ».
  Aucun entraînement ni réglage sur la vidéo fournie.
- Par frame : image entière à 1440 px, plus des recadrages ×2 de 192 px (inférés à 384 px) centrés sur la balle ATD projetée et
  sur les deux meilleurs candidats de l'image entière. Seuls les candidats bruts sont écrits ; suivi, liste noire des marques
  fixes et fusion avec l'ATD sont faits par `video_ball.py`.
- Environnement : image Docker `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` +
  `pip install -r possession/requirements_ball_gpu.txt`
  (`rfdetr==1.10.1 transformers==5.17.0 supervision==0.30.5 opencv-python-headless==5.0.0.93` ;
  sans fixer `transformers`, rfdetr casse : `BackboneConfigMixin` manquant).
- **OpenCV** décode la vidéo et se place sur la première frame (`CAP_PROP_POS_FRAMES`) : une autre version peut décaler
  le décodage. Il n'était pas figé lors de l'exécution d'origine ; `5.0.0.93` est la dernière version publiée au
  2026-11-13 et **reste à confirmer** (`pip freeze` dans l'image du job).
- **Poids** : `rf-detr-medium.pth`, téléchargé à l'exécution par `rfdetr` depuis
  `https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth` (404 992 918 octets).
  MD5 `7223f764a87b863f02eb8d52bf0ce2ee` (celui que `rfdetr==1.10.1` déclare pour ce fichier),
  SHA-256 `749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8` (fichier téléchargé le 2026-11-13).
- Commande (une mi-temps par tâche) :
  ```
  python possession/detect_ball_gpu.py --video BAR-ATH_tactical.mp4 --atd atd_ball_px.npy \
         --start 11185 --end 90652 --out parts_h0 --batch 4
  python possession/detect_ball_gpu.py --video BAR-ATH_tactical.mp4 --atd atd_ball_px.npy \
         --start 110808 --end 190198 --out parts_h1 --batch 4
  ```
- Exécution réelle : Google Cloud Batch, g2-standard-4 + 1 NVIDIA L4 (us-west1), environ 2 h 55 par mi-temps
  (79 467 et 79 390 frames ; ~8 images/s).

## Export pour l'étage 2
L'étage 2 (`etage2/airball/metrica_io.py`) lit la balle fusionnée sous forme de CSV, `data/tactical/detections_balle_BAR-ATH.csv` :
```
python possession/export_ball_csv.py --ball outputs/intermediaires/balle_fusionnee.npz --data data \
       --out data/tactical/detections_balle_BAR-ATH.csv
```
Vérifié le 2026-11-13 : identique octet pour octet au fichier utilisé jusqu'ici par l'étage 2.
