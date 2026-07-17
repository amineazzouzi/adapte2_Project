# Étude : caractérisation et classification de sévérité des anomalies oscillo

Note méthodologique — à valider avant toute implémentation ou calibration sur données réelles.

## 1. Contexte : ce qui existe aujourd'hui

Le pipeline oscillo (`oscillo_analysis.py` → `OscilloPipeline`) détecte déjà des anomalies,
mais de façon **binaire** :

- `filter_anomaly_windows` (`src/signal_processing/windowing.py`) marque une fenêtre comme
  anomalie si l'amplitude moyenne (max−min) sur 10 segments dépasse un seuil (défaut 50).
  → capture uniquement l'**intensité**, rien sur la durée, la forme ou la répétition.
- `track_events_temporal_gpu` regroupe les fenêtres anomalies consécutives et similaires
  (NCC) en **événements**.
- `cluster_events_by_type` regroupe les événements similaires en **types** récurrents
  (`cluster_id`).

Il n'existe aujourd'hui aucune notion de gravité : toutes les anomalies détectées sont
traitées de façon équivalente dans les rapports.

## 2. Qu'est-ce qu'une anomalie, dans ce contexte ?

Une anomalie oscillo est un **écart de forme ou d'amplitude par rapport au comportement
électrique attendu** (sinusoïde 50 Hz stable) sur une fenêtre de ~200 ms. Sur une
installation agricole, ces écarts recouvrent typiquement :

- **Transitoires / pics courts** — démarrage moteur, commutation, arc électrique
- **Dérives continues** — surcharge, dégradation d'isolant, échauffement
- **Distorsions harmoniques** — pollution par charges non linéaires (variateurs, etc.)
- **Instabilités récurrentes** — un même défaut qui se répète dans le temps

La détection actuelle (amplitude seule) capture surtout la première catégorie. Une
caractérisation multi-axe est nécessaire pour distinguer un pic bénin ponctuel d'une
dérive structurelle grave.

## 3. Caractériser une anomalie : 4 axes mesurables

Tous ces axes sont déjà calculables à partir de ce que le pipeline produit par événement
(`build_signal_profile`, `src/analysis/clustering.py:80-92`) — aucune nouvelle mesure
coûteuse n'est nécessaire pour un premier passage.

| Axe | Donnée source | Ce qu'il révèle |
|---|---|---|
| **Intensité** | segment range (max−min), déjà calculé pour la détection | Amplitude du dépassement par rapport au signal nominal |
| **Persistance** | `duration_s`, `window_count` de l'événement | Transitoire ponctuel vs défaut soutenu dans le temps |
| **Contenu fréquentiel** | `dom_freq` (déjà calculé par événement) | Haute fréquence → transitoire/arc ; dérive basse fréquence → charge/thermique |
| **Récurrence** | nombre d'événements partageant le même `cluster_id` sur la période observée | Défaut isolé vs défaut systémique qui se répète |

Un 5e axe possible mais non trivial : l'**écart relatif au comportement normal du signal
lui-même** (ex. z-score de l'amplitude par rapport aux fenêtres non-anomalies du même
boîtier/voie), plutôt qu'un seuil absolu — pertinent car l'amplitude nominale varie d'une
installation à l'autre.

## 4. Classifier en grave / moyenne / faible : deux approches

### A. Data-driven (percentiles sur données réelles) — recommandé

Score composite = combinaison pondérée des 4 axes normalisés, puis seuils grave/moyenne/faible
fixés par percentiles calculés sur les `signal_profile.json` déjà produits (ex. top 10 % =
grave, 30 % suivants = moyenne, reste = faible).

- **Avantage** : s'adapte à chaque boîtier/voie, cohérent avec la façon dont
  `NCC_TYPE_THRESHOLD` est déjà retuné par inspection des données réelles plutôt que fixé
  a priori (voir mémoire projet — les seuils NCC sont réajustés directement dans les
  fichiers après inspection des données).
- **Inconvénient** : seuils relatifs, pas de valeur physique absolue — deux signaux
  différents peuvent avoir des seuils de gravité différents pour la même amplitude brute.

### B. Basé normes (IEC 61000-4-30, courbes magnitude × durée pour sags/swells)

Classification selon des grilles magnitude/durée standardisées utilisées en qualité
électrique.

- **Avantage** : seuils physiques traçables, reproductibles hors du dataset.
- **Inconvénient** : suppose un RMS continu de référence sur la durée du défaut — c'est le
  terrain du pipeline **TRMS** (échantillons 1 s), pas des captures oscillo ponctuelles de
  200 ms qui n'ont pas de continuité temporelle entre fenêtres.

**Recommandation** : approche A pour le pipeline oscillo (200 ms), en calibrant les seuils
après inspection des distributions réelles des 4 axes sur plusieurs signaux déjà traités —
même logique que pour les seuils NCC existants. L'approche B resterait pertinente si un
jour la sévérité est étudiée côté pipeline TRMS plutôt qu'oscillo.

## 5. Étapes suivantes (non engagées ici)

1. Calibration : script d'exploration sur les `signal_profile.json` existants pour observer
   les distributions réelles des 4 axes et proposer des seuils concrets.
2. Formule du score composite (pondération à discuter — intensité et persistance sont
   probablement les axes dominants, récurrence un facteur aggravant).
3. Implémentation : ajout d'un champ `severity_score` / `severity_label` dans
   `build_signal_profile`, propagé dans `signal_profile.json` et affiché dans
   `rapport_enrichi.html`.
