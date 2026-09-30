# Urban Risk-Aware Navigation via VQA-Based Event Maps

Code and dataset tooling for the paper:

> **Urban Risk-Aware Navigation via VQA-Based Event Maps for People with Low Vision**
> Antoni Valls, Jordi Sanchez-Riera — IRI (CSIC-UPC), Barcelona.
> arXiv:[2605.11782](https://arxiv.org/abs/2605.11782) · submitted to IEEE T-ITS.

The system asks a Vision-Language Model (VLM) a **three-level hierarchy of yes/no questions** about each keyframe of a walking sequence (stairs, crossings, construction, obstacles, crowding, vehicles, surface, non-sidewalk). The Level-1 answers are turned into a **weighted risk score** per image, aggregated per street segment, and drawn as a **risk event map** (Safe / Caution / Danger / High Risk) on top of OpenStreetMap.

Four VQA models are benchmarked against a manually annotated dataset of **20 cities, 41 sequences, 820 images and 18,352 answered questions**.

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Setup](#2-setup)
3. [The dataset](#3-the-dataset)
4. [Pipeline overview](#4-pipeline-overview)
5. [**Adding a new Mapillary sequence (step-by-step)**](#5-adding-a-new-mapillary-sequence-step-by-step)
6. [Self-recorded data (Biel Glasses / ZED)](#6-self-recorded-data-biel-glasses--zed)
7. [Running the VQA models](#7-running-the-vqa-models)
8. [Evaluation and risk score](#8-evaluation-and-risk-score)
9. [Event maps](#9-event-maps)
10. [Other utilities](#10-other-utilities)
11. [Gotchas](#11-gotchas)
12. [Citation](#12-citation)

---

## 1. Repository layout

```
VQA-for-event-maps/
├── app/
│   └── runner.py              # Runs the VQA models over sequences (+ evaluation)
├── core/
│   ├── mapillaryRetrieve.py   # Downloads a Mapillary sequence (images + GPS + metadata)
│   ├── gt_maker.py            # Keyframe selection + manual ground-truth labeling (OpenCV GUI)
│   ├── gt_corrector.py        # Web tool to review GT where most models disagree (Flask)
│   ├── vqaModel.py            # Wrapper around ViLT / LLaVA / InstructBLIP / Qwen-VL
│   ├── promptManager.py       # Loads the question hierarchy and resolves follow-ups
│   └── eval.py                # Metrics (Acc, F1, Prec, Rec, Spec) + risk-score MAE
├── inout/
│   ├── vqa_prompts.json       # THE QUESTION HIERARCHY (questions, dependencies, presets)
│   ├── svoExport.py           # Extracts frames from ZED .svo2 recordings (Barcelona data)
│   └── utils.py               # Progress bar, misc
├── utils/
│   ├── unify_gt.py            # Merges all GT files into data/all_GT/all_ground_truth.jsonl
│   ├── model_comparison.py    # Aggregates metrics across cities/models, makes figures
│   └── prompt_utils.py        # CLI to inspect presets / questions
├── viz/
│   └── viz_utils.py           # Risk score per image + interactive risk event map (folium + OSM)
├── data/                      # Dataset (only GT + model answers are in git, see §3)
├── .env.example               # Template for credentials -> copy to .env
└── requirements.txt
```

All scripts are configured by editing the variables in their `if __name__ == "__main__":` block (there is no CLI), and **must be run from inside their own folder** because paths are relative (`../data/...`, `../inout/...`):

```bash
cd core && python gt_maker.py        # correct
python core/gt_maker.py              # WRONG: will not find ../data
```

---

## 2. Setup

Tested on Ubuntu 22.04, Python 3.10, NVIDIA GPU with CUDA 12.x driver.

```bash
sudo apt install python3.10-venv          # only if venv is missing
git clone <repo-url> && cd VQA-for-event-maps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"
```

If you **only download data and label ground truth** (the main student task), you don't need a GPU nor the models. The GT tools only need `opencv-python`, `pillow`, `numpy`, `requests`, `tqdm` and `flask`.

### Credentials: Mapillary access token

Credentials are **never** written in the code. They live in a `.env` file at the repository root, which is git-ignored.

1. Create a (free) Mapillary account and go to <https://www.mapillary.com/dashboard/developers>.
2. *Register application* (any name, e.g. `vqa-event-maps-<yourname>`, read access is enough).
3. Copy the **Client Token** (it looks like `MLY|1234567890|abcdef...`).
4. Create your `.env`:
   ```bash
   cp .env.example .env
   # edit .env ->  MAPILLARY_TOKEN="MLY|...your token..."   (keep the quotes)
   ```

Each person uses **their own token**. Do not commit `.env`, do not paste tokens into scripts, issues or chat. Before committing, `git status` must not show `.env`. Alternatively, `export MAPILLARY_TOKEN=...` in your shell also works.

### VQA models

Downloaded automatically from Hugging Face on first use (to `~/.cache/huggingface`), or in advance:

| Key in code    | Hugging Face checkpoint                  | Type                  | Disk   |
|----------------|------------------------------------------|-----------------------|--------|
| `vilt`         | `dandelin/vilt-b32-finetuned-vqa`        | Classification        | ~0.5 GB|
| `instructblip` | `Salesforce/instructblip-flan-t5-xl`     | Generative            | ~16 GB |
| `llava`        | `llava-hf/llava-v1.6-mistral-7b-hf`      | Generative (chat)     | ~15 GB |
| `qwen-vl`      | `Qwen/Qwen2-VL-7B-Instruct`              | Generative (chat)     | ~16 GB |

```bash
hf download Qwen/Qwen2-VL-7B-Instruct      # repeat for each checkpoint
```

The 7B models need ~16 GB of VRAM in fp16. With less, `device_map="auto"` offloads layers to CPU RAM: it works, but slowly.

---

## 3. The dataset

### What is in git and what is not

| In git                                          | NOT in git (shared separately)                    |
|-------------------------------------------------|---------------------------------------------------|
| `ground_truth_labels.jsonl` (+ `.bak`)          | `images/`, `images_selected/` (jpg/png)           |
| `results/<model>/answers.jsonl`                 | `gps_positions.csv`, `metadata.json`, other `.json`|
| `results/<model>/answer_characteristics.txt`    | `metrics.json`, `metrics_summary.csv`, `*.html` maps |
| `data/all_GT/all_ground_truth.jsonl`            | `.svo2` recordings                                |

Images and metadata are too heavy for git (~11 GB). **Ask Antoni for the data archive** (`VQA-data-backup.tar.gz`) and extract it at the repository root:

```bash
tar -xzf VQA-data-backup.tar.gz     # creates/fills ./data/
```

It is safe to extract on top of a fresh clone: the files that are in git are identical in the archive.

### Folder structure

```
data/<Continent>/<Sequence>/
├── images/                     # all downloaded frames (0000_<mapillaryImageId>.jpg ...)
├── images_selected/            # the 20 keyframes chosen for annotation (copies)
├── gps_positions.csv           # filename, latitude, longitude, captured_at, compass_angle, is_pano, image_type
├── metadata.json               # full Mapillary metadata per image (camera, size, sequence id, ...)
├── ground_truth_labels.jsonl   # manual labels (one line per image x question)
├── ground_truth_labels.jsonl.bak   # automatic backup made by gt_corrector.py
├── interactive_map_gt.html     # risk event map from GT (optional)
└── results/<model>/
    ├── answers.jsonl           # model answers (same structure as GT)
    ├── metrics.json            # full evaluation
    ├── metrics_summary.csv     # overall / per level / per topic metrics
    ├── answer_characteristics.txt  # yes/no balance of predictions vs GT
    └── interactive_map.html    # risk event map from this model (optional)
```

`<Continent>` is one of `Africa`, `America`, `Asia`, `Europe`, `Oceania` (North and South America share `America`). Some older sequences also contain `gps_positions.json` and `detections.json` (Mapillary object detections). They come from an earlier version of the download script and are not used.

### Current content

| Continent | Sequences | Source |
|-----------|-----------|--------|
| Africa    | `Acra`, `Kampala`, `Lusaka`, `Marrakesh` | Mapillary |
| America   | `BuenosAires`, `Chihuahua`, `LaHabana`, `NewYork`, `SanFrancisco`, `Ushuaia` | Mapillary |
| Asia      | `Bombai`, `Singapore`, `Tokio1`, `Tokio2` | Mapillary |
| Europe    | `London1`, `Munich`, `Oslo`, `Soller` | Mapillary |
| Europe    | `00`–`22` (21 sequences, no `13`, `18`) | Self-recorded in Barcelona (Biel Glasses, ZED camera) |
| Oceania   | `Sidney`, `Wellington` | Mapillary |

Every sequence has exactly **20 annotated keyframes** and answers from the 4 models.

### Ground-truth file format

`ground_truth_labels.jsonl`, one JSON object per line:

```json
{"image_path": "../data/America/NewYork/images_selected/0000_1247725302301351.jpg",
 "image_name": "0000_1247725302301351.jpg",
 "image_index": 0,
 "question_id": "q_crossing_signal_present",
 "question": "Is there a pedestrian signal present at the crossing?",
 "answer": "yes",
 "level": 2,
 "parent_question": "q_crossing_nearby",
 "short_label": "Signal Present",
 "timestamp": "2026-01-21T15:33:08.311313"}
```

Model answers (`results/<model>/answers.jsonl`) have the same keys plus `model` and `confidence` (only ViLT gives a confidence). Matching between GT and predictions is done on `(image_name, question_id)`.

### The question hierarchy

Defined in [`inout/vqa_prompts.json`](inout/vqa_prompts.json). Every question is prefixed with the base context:

> *"You are an expert at detecting pedestrian obstacles for people with low vision. Answer only with Yes or No."*

A question with a `dependency` is only asked if its parent was answered **Yes**. There are 8 Level-1 questions, 27 Level-2 and 9 Level-3.

| Category | Level-1 question (`id`) | Risk tier |
|----------|-------------------------|-----------|
| Non-Sidewalk | Is the pedestrian currently outside a sidewalk or pedestrian path/crossing? (`q_pedestrian_not_on_sidewalk`) | Critical (1.0) |
| Crossings | Is there a street crossing within the next 10 meters? (`q_crossing_nearby`) | High (0.6) |
| Stairs | Are stairs visible on the current path? (`q_stairs_visible`) | High (0.6) |
| Construction | Is any construction activity present on the current path? (`q_construction_visible`) | Critical (1.0) |
| Obstacles | Is there any object (not a person) located in the walking path ahead that the pedestrian should be aware of? (`q_obstacle_blocking`) | High (0.6) |
| Crowding | Are multiple pedestrians visible in the walking area? (`q_pedestrians_present`) | Low (0.3) |
| Vehicles | Are there any vehicles within 10 meters of the pathway? (`q_vehicle_nearby`) | Low (0.3) |
| Surface | Does the path surface show abnormal conditions (water, damage, significant slope)? (`q_surface_hazardous`) | Critical (1.0) |

<details>
<summary><b>Full hierarchy (Level 2 and Level 3)</b></summary>

```
q_pedestrian_not_on_sidewalk
├── q_pedestrian_on_road            Is the pedestrian on the roadway or street surface?
└── q_pedestrian_on_unpaved         Is the pedestrian walking on grass, dirt, or an unpaved surface?

q_crossing_nearby
├── q_crossing_signal_present       Is there a pedestrian signal present at the crossing?
│   └── q_crossing_signal_red_green Is the pedestrian signal showing green?
├── q_crossing_vehicles_present     Are vehicles currently in the crossing lanes?
│   └── q_crossing_vehicles_moving  Are the vehicles in the crossing lanes moving?
├── q_crossing_marked               Is there a crossing marked with painted lines?
└── q_crossing_others_waiting       Are there other pedestrians on the street crossing?

q_stairs_visible
├── q_stairs_heading                Are the stairs directly ahead on the path?
│   └── q_stairs_only_option        Are the stairs the only path option visible?
└── q_stairs_handrail               Is there a handrail present next to the stairs?

q_construction_visible
├── q_construction_barriers         Are barriers or cones blocking the path?
├── q_construction_equipment        Is construction equipment visible on or near the path?
└── q_construction_overhead         Does the expected path pass underneath any construction scaffolding or overhead work areas?

q_obstacle_blocking
├── q_obstacle_upcoming_step        Is there an upcoming step or change in ground level on the path (curb, sidewalk transition, tilted manhole cover)?
├── q_obstacle_is_vehicle           Is there a parked vehicle blocking the sidewalk?
├── q_obstacle_movable_objects      Are there movable objects on the path (furniture, planters, or loose items)?
├── q_obstacle_fixed_infrastructure Are there fixed infrastructure obstacles on the path (stairs, railings, traffic signs, streetlights, trees)?
├── q_obstacle_blocks_half          Is more than half the path width blocked?
├── q_obstacle_space_around         Is there clear space to walk around the obstructions?
├── q_obstacle_overhead             Is there a low-hanging branch, awning, or sign overhead?
└── q_obstacle_seating              Is outdoor seating encroaching onto the walkway?

q_pedestrians_present
├── q_crowding_same_direction       Are the majority of pedestrians moving in the same direction?
└── q_crowding_slow_movement        Would normal walking speed need to be reduced due to pedestrians?

q_vehicle_nearby
├── q_vehicle_is_bicycle            Is there a bicycle within 10 meters of the path?
│   ├── q_bicycle_moving            Is the bicycle moving?
│   └── q_bicycle_bike_lane         Is the bicycle on a dedicated bike lane?
├── q_vehicle_is_car_truck_bus      Is there a car, truck or bus within 10 meters of the path?
│   ├── q_motor_vehicle_moving      Is there a car, truck or bus moving?
│   └── q_motor_vehicle_blocking    Is the car, truck or bus on the sidewalk?
└── q_vehicle_is_motorcycle         Is there a motorcycle within 10 meters of the path?
    ├── q_motorcycle_sidewalk       Is the motorcycle on the sidewalk?
    └── q_motorcycle_moving         Is the motorcycle moving?

q_surface_hazardous
├── q_surface_water_ice             Is water or ice visible on the surface?
├── q_surface_sloped                Is the surface noticeably sloped or tilted?
└── q_surface_cracks_holes          Is the pavement visibly damaged (large cracks, holes, or broken sections)?
```

Print it yourself with `cd utils && python prompt_utils.py list-questions --preset full_hierarchical`.
</details>

> ⚠️ Do **not** change question IDs or texts in `vqa_prompts.json` while the dataset is being extended: all GT files and model answers are matched by `question_id`, and the paper numbers depend on the current wording.

---

## 4. Pipeline overview

```
 (1) Download sequence        core/mapillaryRetrieve.py      -> images/, gps_positions.csv, metadata.json
            │                 (or inout/svoExport.py for ZED recordings)
            ▼
 (2) Select 20 keyframes      core/gt_maker.py               -> images_selected/
     + label ground truth                                    -> ground_truth_labels.jsonl
            ▼
 (3) Run VQA models           app/runner.py                  -> results/<model>/answers.jsonl
     + evaluate               (calls core/eval.py)           -> results/<model>/metrics*.{json,csv}
            ▼
 (4) Review suspicious GT     core/gt_corrector.py           -> fixes ground_truth_labels.jsonl
            ▼
 (5) Aggregate                utils/unify_gt.py, utils/model_comparison.py, viz/viz_utils.py
```

Steps (1) and (2) need no GPU. Step (3) needs the models.

---

## 5. Adding a new Mapillary sequence (step-by-step)

This is the main task for extending the dataset. Budget ~15 min for the download and ~45–60 min for labeling 20 images.

### 5.1 Choose a good sequence

Browse <https://www.mapillary.com/app>, zoom into a city and click on the green lines (captured sequences) to preview images.

A good sequence:

- **Is taken from a pedestrian's point of view**: someone walking on the sidewalk, crossing streets, with the camera facing forward. Avoid car/dashcam sequences driving on the road, which bias the *Non-Sidewalk* and *Vehicles* questions.
- **Has enough frames.** The script keeps every second image, and you need 20 good keyframes, so aim for sequences of **≥ 100 images** (existing ones have 45–900 downloaded frames).
- **Is sharp and in daylight**, with images not heavily blurred or covered by a car hood or dashboard.
- **Adds diversity**: a new city or country, or a new kind of environment (market street, unpaved road, stairs, construction works, crowded square...). The paper shows the weakest categories are **Non-Sidewalk, Construction, Stairs and Surface** (few positive examples). Sequences containing those are especially valuable.
- 360° sequences are OK: the script automatically crops the forward-facing 90° view using the compass angle.

Before downloading, check with Antoni that the city or sequence is not already in the dataset (see §3).

### 5.2 Get the sequence ID

When you open an image in the Mapillary viewer, the URL contains `pKey=<IMAGE_ID>`. Ask the API which sequence that image belongs to:

```bash
set -a; source .env; set +a      # loads MAPILLARY_TOKEN into the shell
curl "https://graph.mapillary.com/<IMAGE_ID>?fields=id,sequence,captured_at,camera_type&access_token=$MAPILLARY_TOKEN"
# -> {"id": "...", "sequence": "5xBMc2sYv7nOLRUSoCrw8f", "captured_at": ..., "camera_type": "perspective"}
```

The `sequence` value (a ~22-character string) is the sequence ID. `camera_type` tells you whether it is a normal (`perspective`) or 360° (`spherical`/`equirectangular`) sequence.

### 5.3 Download it

Edit the `todos` list at the bottom of [`core/mapillaryRetrieve.py`](core/mapillaryRetrieve.py):

```python
todos = [
    ("Asia", "Hanoi", "AbCdEf1234567890xyz"),   # (continent, folder name, sequence ID)
]
```

Folder name convention: `CamelCase`, no spaces or accents, city name in English. If a city gets a second sequence, number them (`Tokio1`, `Tokio2`).

```bash
cd core
python mapillaryRetrieve.py
```

What it does:

- Lists all image IDs of the sequence and keeps **every second image, up to 601 images**.
- Downloads the 2048 px version of each one to `data/<Continent>/<City>/images/0000_<imageId>.jpg`, `0001_...` (the prefix keeps the temporal order).
- 360° images are cropped to the forward-facing 90° view.
- Writes `metadata.json` (all fields) and `gps_positions.csv` (used by the event maps).

Then open the `images/` folder and check the result. If the sequence turns out to be bad, delete the folder and pick another one.

### 5.4 Label the ground truth

Labeling uses an OpenCV window, so you need a desktop session (it won't work over plain SSH).

Edit the configuration at the bottom of [`core/gt_maker.py`](core/gt_maker.py):

```python
PROMPT_PRESET = "full_hierarchical"   # always this one for the dataset
CONTINENT = "Asia"
CITY = "Hanoi"
NUM_KEYFRAMES = 20                    # always 20
OVERRIDE_EXISTING = False
SELECT_IMAGES = False                 # selection is triggered automatically if needed
PATCH_QUESTIONS = None
```

```bash
cd core
python gt_maker.py
```

#### Step A: keyframe selection

The first time you run it on a sequence, `images_selected/` doesn't contain 20 images, so the **selection window** opens. The sequence is split into 20 equal segments, and you pick **one image per segment**:

| Key | Action |
|-----|--------|
| `K` | Keep this image (copied to `images_selected/`) and jump to the next segment |
| `S` | Skip, show the next image of the same segment |
| `Q` | Abort selection |

Keep the first image of each segment that is **sharp, facing forward, and representative** of what a pedestrian would see there. Don't keep only "interesting" images: the aim is a uniform sample of the route.

You must end with **exactly 20 images**. If a segment runs out of images without a `K`, or you abort, empty `images_selected/` and run again, otherwise leftover copies will mix with the new selection.

#### Step B: answering the questions

After pressing ENTER, each keyframe is shown with one question at a time. Follow-up (Level 2 and 3) questions appear automatically when you answer **Yes**.

| Key | Action |
|-----|--------|
| `Y` | Yes |
| `N` | No |
| `B` | Back to the previous **Level-1** question (discards its follow-ups). Only works on Level-1 questions |
| `R` | *(first question of an image only)* Repeat **all** answers of the previous image. Use it only for nearly identical consecutive frames |
| `S` | Skip this image: its answers are discarded and it will be asked again next run |
| `Q` | Quit. Finished images are saved, **the image in progress is lost** |

- Answers are written to `ground_truth_labels.jsonl` **when an image is finished**.
- To continue later, just run the script again: images that already have answers are skipped.
- If you make a mistake inside the follow-ups, press `S` to discard the image and label it again at the end.

#### Labeling conventions

Consistency across annotators matters more than anything else. Always answer **from the point of view of the pedestrian holding the camera, looking forward**, using the definitions from the paper:

- **Non-Sidewalk**: the visible path the pedestrian is on lies *outside* a designated sidewalk or pedestrian crossing (road, dirt, grass...).
- **Crossings**: a street crossing exists within ~**10 m** ahead.
- **Stairs**: steps or stairs are visible on the current path.
- **Construction**: construction activity or pathway detours are visible on the path.
- **Obstacles**: any *object* (not a person) in the walking path ahead that requires attention (curbs, poles, parked vehicles, furniture, overhead signs...).
- **Crowding**: multiple pedestrians are visible in the walking area.
- **Vehicles**: any vehicle (car, bus, bike, motorcycle) within ~**10 m** of the path.
- **Surface**: abnormal surface conditions: water/ice, damage (cracks, holes, broken sections) or significant slope.

Answer only what is **visible in the image**, not what you know about the place. If you are unsure about a case, write down the image name and ask. Don't guess differently each time.

#### Fixing mistakes afterwards (patch mode)

To re-ask specific questions for all images of a sequence, set for example:

```python
PATCH_QUESTIONS = {"q_stairs_visible"}
```

This deletes the answers to those questions **and their follow-ups** for every image, then asks them again. Leave `OVERRIDE_EXISTING = False`.

> ⚠️ `OVERRIDE_EXISTING = True` with `PATCH_QUESTIONS = None` **overwrites the whole GT file**. Don't use it unless you really want to start from scratch.

### 5.5 Deliver the sequence

1. Work on your own branch: `git checkout -b data/<yourname>-<city>`.
2. Commit **only** `data/<Continent>/<City>/ground_truth_labels.jsonl`. Images, CSV and JSON are ignored by git on purpose.
3. Compress the whole sequence folder (`images/`, `images_selected/`, `gps_positions.csv`, `metadata.json`, GT) and send it to Antoni (shared drive), named `<Continent>_<City>.tar.gz`.
4. Open a pull request, or tell Antoni the branch name. In the PR, note the Mapillary sequence ID, the city, and anything special about the sequence.

Before pushing, run `git status` and check that `.env` and no images appear.

Steps 7 (running models) and 5.6 (review) are usually done by Antoni on the GPU machine.

### 5.6 Review the GT against the models (after the models have run)

[`core/gt_corrector.py`](core/gt_corrector.py) is a small web app that shows only the GT answers where **at least 3 of the 4 models disagree** with the annotator. Most are model errors, but it catches annotation mistakes quickly.

```bash
cd core
# edit BASE_PATH = Path("../data/<Continent>/<City>/")
python gt_corrector.py        # open http://localhost:5000
```

| Key | Action |
|-----|--------|
| `↑` / `↓` | Set GT to Yes / No (saved immediately) |
| `←` / `→` | Previous / next flagged item |

The first change creates `ground_truth_labels.jsonl.bak` with the original file. Only correct the GT when the annotator was clearly wrong, not because the models say so. Afterwards, re-run the evaluation (§8).

Finally, rebuild the merged GT file: `cd utils && python unify_gt.py`.

---

## 6. Self-recorded data (Biel Glasses / ZED)

The Barcelona sequences `data/Europe/00` … `22` were recorded around IRI with a ZED stereo camera mounted on Biel Glasses smart glasses (`.svo2` files in `data/Europe/Barcelona/svo/`, not in git).

- [`inout/svoExport.py`](inout/svoExport.py) extracts ~100 evenly spaced left-camera frames per recording to `data/Europe/Barcelona/<seq>/images/*.png`. The folders were then moved to `data/Europe/<seq>/`. It needs the ZED SDK and its `pyzed` Python wheel (not in `requirements.txt`).
- `data/Europe/Barcelona/IRI_sequences_GT.txt` contains the planned route of each recording (GPS track points).
- Most Barcelona sequences have **no per-frame GPS** (`gps_positions.csv` only exists for `08`), so event maps can't be generated for them.

From step (2) on, the pipeline is the same as for Mapillary.

---

## 7. Running the VQA models

[`app/runner.py`](app/runner.py) loops over models × sequences. For each sequence it:

1. Loads `images_selected/` and processes one image every `len(images)/20` (i.e. all 20 keyframes).
2. Asks the 8 Level-1 questions, then the follow-ups of every *yes* answer (Level 2 and then Level 3), with the base context prepended.
3. Writes `results/<model>/answers.jsonl` (overwritten on each run).
4. Evaluates against `ground_truth_labels.jsonl` (see §8).

Configure the `models` list and the `continent_city` dict in its `__main__` block. **When adding a new sequence, add it to that dict** (and to the one in `core/eval.py`).

```bash
cd app
python runner.py
```

Answers of generative models are free text. Anything starting with "yes" counts as *yes* and everything else as *no*.

---

## 8. Evaluation and risk score

[`core/eval.py`](core/eval.py) matches predictions and GT on `(image_name, question_id)` and computes accuracy, precision, recall, specificity and F1 overall, per level, per topic (Level-1 category) and per question.

- **Level-2/3 questions are evaluated only where the GT has an answer**, i.e. where the annotator said *yes* to the parent. Follow-ups triggered by a model false positive have no GT and are ignored (see paper §IV-B).
- Running `python eval.py` from `core/` re-evaluates every model and sequence listed in its `__main__` without running any model. Use it after GT corrections.

### Risk score (paper §III-C)

For each image, over the 8 Level-1 questions $Q$:

$$R_{img} = \frac{\max\left(0, \sum_{i \in Q} w_i x_i\right)}{\sum_{i \in Q} w_i}, \qquad x_i = \begin{cases} 1 & \text{Yes} \\ -1/|Q| & \text{No} \\ 0 & \text{unanswered} \end{cases}$$

with $w_i$ = 1.0 (Construction, Surface, Non-Sidewalk), 0.6 (Crossings, Stairs, Obstacles), 0.3 (Crowding, Vehicles). In code this is `HAZARD_CONFIG` and `SAFETY_REWARD_RATIO = 1/8`, in both `core/eval.py` and `viz/viz_utils.py`. Keep the two in sync.

- **Segment risk** = max of the image risks mapped to that OSM street segment.
- **Categories**: Safe ≤ 0.15 < Caution < 0.4 ≤ Danger < 0.7 ≤ High Risk.
- **MAE_R** = mean |R_gt − R_pred| over images (`Risk_MAE` in the outputs).

Per-sequence outputs: `metrics.json`, `metrics_summary.csv`, `answer_characteristics.txt` in `results/<model>/`.
Cross-sequence tables and figures: `cd utils && python model_comparison.py` → `data/model_comparison/`.

---

## 9. Event maps

[`viz/viz_utils.py`](viz/viz_utils.py) builds an interactive HTML map (folium, CartoDB tiles):

- The OSM *drive* network within 1 km of the sequence centre is downloaded with `osmnx` (internet needed).
- Each annotated keyframe is snapped to its nearest street segment, and segments are coloured by their max risk.
- Grey segments have no observations. The dashed blue line is the full GPS track, and markers show the keyframes with their answers.

```bash
cd viz
# edit CONTINENT / CITY / MODEL in __main__
python viz_utils.py     # writes results/<MODEL>/interactive_map.html and interactive_map_gt.html
```

It requires `gps_positions.csv`, whose `filename` column must match the names in `images_selected/`.

---

## 10. Other utilities

| Command (run inside the folder) | What it does |
|---|---|
| `utils/ python prompt_utils.py list-presets` | List question presets (`full_hierarchical`, `level_1_only`, `crossing`, ...) |
| `utils/ python prompt_utils.py list-questions --preset crossing` | List the questions of a preset |
| `utils/ python prompt_utils.py stats` | Question counts |
| `utils/ python unify_gt.py` | Merge all GT files into `data/all_GT/all_ground_truth.jsonl` |
| `utils/ python model_comparison.py` | Global / per-continent / per-topic comparison plots |

---

## 11. Gotchas

- **Run scripts from their own folder** (`cd core && python gt_maker.py`). Paths are relative.
- **`images_selected/` must contain exactly 20 images**, otherwise `gt_maker.py` re-opens the selection window.
- **Quitting (`Q`) during labeling loses the image in progress**, but finished images are saved.
- `runner.py` **overwrites** `results/<model>/answers.jsonl` for the sequences it processes.
- Don't commit personal config edits (city names in `__main__` blocks) unless they are meant to stay. Keep your commits to GT files.
- Never commit tokens: use `.env` (§2).
- Mapillary images are licensed CC BY-SA 4.0. Keep the `metadata.json` (image IDs) so the source can be attributed.

---

## 12. Citation

```bibtex
@article{valls2026urban,
  title   = {Urban Risk-Aware Navigation via VQA-Based Event Maps for People with Low Vision},
  author  = {Valls, Antoni and Sanchez-Riera, Jordi},
  journal = {arXiv preprint arXiv:2605.11782},
  year    = {2026}
}
```

This work builds on the event-map paradigm of Morales *et al.*, "VQA-driven event maps for assistive navigation for people with low vision in urban environments", ICRA 2025.

Project CPP2021-008760 funded by MCIU/AEI/10.13039/501100011033 and by the European Union NextGenerationEU/PRTR.
