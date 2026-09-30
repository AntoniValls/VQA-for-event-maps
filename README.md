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

The code is organized by pipeline stage:

```
VQA-for-event-maps/
├── dataset/                   # 1. BUILDING THE DATASET (no GPU needed)
│   ├── download_mapillary.py  #    Downloads a Mapillary sequence (images + metadata.json)
│   ├── export_svo.py          #    Extracts frames from ZED .svo2 recordings (Barcelona data)
│   ├── label_gt.py            #    Keyframe selection + manual ground-truth labeling (OpenCV GUI)
│   ├── review_gt.py           #    Web tool to review GT where most models disagree (Flask)
│   ├── merge_gt.py            #    Merges all GT files into data/all_GT/all_ground_truth.jsonl
│   └── pack_data.sh           #    Creates the dataset / per-sequence .tar.gz archives
├── vqa/                       # 2. ASKING THE MODELS
│   ├── prompts.json           #    THE QUESTION HIERARCHY (questions, dependencies, presets)
│   ├── prompt_manager.py      #    Loads the hierarchy and resolves follow-up questions
│   ├── prompt_utils.py        #    CLI to inspect presets / questions
│   ├── models.py              #    Wrapper around ViLT / LLaVA / InstructBLIP / Qwen-VL
│   └── run.py                 #    Runs the models over the sequences -> answers.jsonl
├── evaluation/                # 3. EVALUATING AND MAPPING
│   ├── risk.py                #    Risk score R_img and risk categories (single definition)
│   ├── evaluate.py            #    Metrics (Acc, F1, Prec, Rec, Spec) + risk-score MAE
│   ├── compare_models.py      #    Aggregates metrics across cities/models (paper tables/figures)
│   └── event_map.py           #    Interactive risk event maps (folium + OpenStreetMap)
├── common/
│   ├── paths.py               # Where things are: repo root, data folder, prompts, .env settings
│   └── utils.py               # Progress bar
├── data/                      # The dataset. NOT in git, see §3
├── .env.example               # Template for your settings -> copy to .env
└── requirements.txt
```

**Git contains only code. The dataset is distributed as `.tar.gz` archives** (§3).

All scripts have a command-line interface (`--help`) and can be run from any directory, for example `python dataset/label_gt.py --continent Asia --city Hanoi`.

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
cp .env.example .env                      # then fill it in (see below)
```

If you **only download data and label ground truth** (the main student task), you need neither a GPU nor the models.

### Settings and credentials (`.env`)

Personal settings and credentials live in a `.env` file at the repository root. It is **git-ignored and must never be committed**. Tokens are never written in the code.

| Variable | Needed for | Default |
|---|---|---|
| `MAPILLARY_TOKEN` | Downloading Mapillary sequences | — |
| `VQA_DATA_DIR` | Keeping the dataset outside the repository | `<repo>/data` |

**Getting a Mapillary token:**

1. Create a (free) Mapillary account and go to <https://www.mapillary.com/dashboard/developers>.
2. *Register application* (any name, e.g. `vqa-event-maps-<yourname>`, read access is enough).
3. Copy the **Client Token** (it looks like `MLY|1234567890|abcdef...`) into `.env`:
   ```bash
   MAPILLARY_TOKEN="MLY|...your token..."     # keep the quotes
   ```

Each person uses **their own token**. Don't paste tokens into scripts, issues or chat. Before every commit, `git status` must not show `.env`. Variables exported in the shell take precedence over `.env`.

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

### Distribution

The dataset is **not in git**. It is shared as archives whose paths start with `data/`, so they are always extracted at the repository root:

| Archive | Content | Who needs it |
|---|---|---|
| `VQA-dataset-<date>.tar.gz` | The whole dataset: images, GT, GPS, metadata, model results | Everyone |
| `VQA-raw-svo.tar.gz` | Raw ZED recordings of Barcelona (`data/Europe/Barcelona/svo/`, 4.3 GB) | Only to re-extract Barcelona frames |
| `<Continent>_<City>.tar.gz` | One sequence | Used to hand in new sequences (§5.6) |

```bash
cd VQA-for-event-maps
tar -xzf VQA-dataset-<date>.tar.gz          # creates/fills ./data/
```

**Ask Antoni for the link to the latest dataset archive.** To keep the data elsewhere (external disk...), extract it there and set `VQA_DATA_DIR` in `.env`.

Only the maintainer publishes new dataset versions:

```bash
dataset/pack_data.sh dataset      # -> VQA-dataset-YYYY-MM-DD.tar.gz  (excludes the raw .svo2)
dataset/pack_data.sh svo          # -> VQA-raw-svo.tar.gz
```

### Folder structure

```
data/
├── <Continent>/<Sequence>/
│   ├── images/                     # all downloaded frames (0000_<mapillaryImageId>.jpg ...)
│   ├── images_selected/            # the 20 keyframes chosen for annotation (copies)
│   ├── metadata.json               # one entry per image: filename, latitude, longitude, compass, camera, sequence id, ...
│   ├── ground_truth_labels.jsonl   # manual labels (one line per image x question)
│   ├── ground_truth_labels.jsonl.bak   # automatic backup made by dataset/review_gt.py
│   ├── interactive_map_gt.html     # risk event map from GT (optional)
│   └── results/<model>/
│       ├── answers.jsonl           # model answers (same structure as GT)
│       ├── metrics.json            # full evaluation
│       ├── metrics_summary.csv     # overall / per level / per topic metrics
│       ├── answer_characteristics.txt  # yes/no balance of predictions vs GT
│       └── interactive_map.html    # risk event map from this model (optional)
├── Europe/Barcelona/               # IRI_sequences_GT.txt (planned routes) + svo/ (raw recordings)
├── all_GT/all_ground_truth.jsonl   # all GT merged (dataset/merge_gt.py)
└── model_comparison/               # paper figures (evaluation/compare_models.py)
```

`<Continent>` is one of `Africa`, `America`, `Asia`, `Europe`, `Oceania` (North and South America share `America`).

`metadata.json` is the **only** per-image metadata file (positions included): there are no separate GPS files.
Sequences downloaded before September 2026 use the **original** device GPS (`geometry`) and have no `position_source` field. Newer downloads use the computed geometry (§5.3).

A folder counts as an **annotated sequence** (and is picked up automatically by `vqa/run.py` and the evaluation) as soon as it contains `ground_truth_labels.jsonl`.

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

Model answers (`results/<model>/answers.jsonl`) have the same keys plus `model` and `confidence` (only ViLT gives a confidence). GT and predictions are matched on `(image_name, question_id)`. `image_path` is informative only.

### The question hierarchy

Defined in [`vqa/prompts.json`](vqa/prompts.json). Every question is prefixed with the base context:

> *"You are an expert at detecting pedestrian obstacles for people with low vision. Answer only with Yes or No."*

A question with a `dependency` is only asked if its parent was answered **Yes**. There are 8 Level-1 questions, 27 Level-2 and 9 Level-3 (44 in total).

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

Print it yourself with `python vqa/prompt_utils.py list-questions --preset full_hierarchical`.
</details>

> ⚠️ Do **not** change question IDs or texts in `vqa/prompts.json` while the dataset is being extended: all GT files and model answers are matched by `question_id`, and the paper numbers depend on the current wording.

---

## 4. Pipeline overview

```
 (1) Download sequence        dataset/download_mapillary.py  -> images/, metadata.json
            │                 (or dataset/export_svo.py for ZED recordings)
            ▼
 (2) Select 20 keyframes      dataset/label_gt.py            -> images_selected/
     + label ground truth                                    -> ground_truth_labels.jsonl
            ▼
 (3) Run VQA models           vqa/run.py                     -> results/<model>/answers.jsonl
            ▼
 (4) Evaluate                 evaluation/evaluate.py         -> results/<model>/metrics*.{json,csv}
            ▼
 (5) Review suspicious GT     dataset/review_gt.py           -> fixes ground_truth_labels.jsonl
            ▼
 (6) Aggregate                dataset/merge_gt.py, evaluation/compare_models.py, evaluation/event_map.py
```

Steps (1), (2) and (4)–(6) need no GPU. Only step (3) needs the models.

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
- **Is not 360°.** Panoramic sequences (`camera_type` `spherical`/`equirectangular`) are **not accepted**: the download script refuses them. Check `camera_type` as shown in §5.2 before choosing.

Before downloading, check with Antoni that the city or sequence is not already in the dataset (see §3).

### 5.2 Get the sequence ID

When you open an image in the Mapillary viewer, the URL contains `pKey=<IMAGE_ID>`. Ask the API which sequence that image belongs to:

```bash
set -a; source .env; set +a      # loads MAPILLARY_TOKEN into the shell
curl "https://graph.mapillary.com/<IMAGE_ID>?fields=id,sequence,captured_at,camera_type&access_token=$MAPILLARY_TOKEN"
# -> {"id": "...", "sequence": "5xBMc2sYv7nOLRUSoCrw8f", "captured_at": ..., "camera_type": "perspective"}
```

The `sequence` value (a ~22-character string) is the sequence ID. `camera_type` must be `perspective` (or `fisheye`). If it is `spherical` or `equirectangular`, the sequence is 360° and won't be accepted: pick another one.

### 5.3 Download it

```bash
python dataset/download_mapillary.py --continent Asia --city Hanoi --sequence AbCdEf1234567890xyz
```

Folder name (`--city`) convention: `CamelCase`, no spaces or accents, city name in English. If a city gets a second sequence, number them (`Tokio1`, `Tokio2`). The script refuses to overwrite a folder that already has images.

What it does:

- Lists all image IDs of the sequence. If the sequence is 360°, it stops without downloading anything.
- Keeps **every second image, up to 601 images**, and downloads the 2048 px version of each one to `data/<Continent>/<City>/images/0000_<imageId>.jpg`, `0001_...` (the prefix keeps the temporal order). Any isolated 360° image in the sequence is skipped.
- Uses Mapillary's **computed** position and heading (`computed_geometry`, `computed_compass_angle`: refined by Mapillary's 3D reconstruction). For images Mapillary hasn't processed yet, it falls back to the raw device GPS/compass (`geometry`, `compass_angle`). The `position_source` column says which one was used (`computed` / `original`).
- Writes `metadata.json`: one entry per image with its position (both computed and original are kept), heading, camera and size. The event maps read the positions from it.

Then open the `images/` folder and check the result. If the sequence turns out to be bad, delete the folder and pick another one.

### 5.4 Label the ground truth

Labeling uses an OpenCV window, so you need a desktop session (it won't work over plain SSH).

```bash
python dataset/label_gt.py --continent Asia --city Hanoi
```

#### Step A: keyframe selection

When `images_selected/` doesn't contain exactly 20 images (e.g. the first time), the **selection window** opens. The sequence is split into 20 equal segments, and you pick **one image per segment**:

| Key | Action |
|-----|--------|
| `K` | Keep this image (copied to `images_selected/`) and jump to the next segment |
| `S` | Skip, show the next image of the same segment |
| `Q` | Abort selection |

Keep the first image of each segment that is **sharp, facing forward, and representative** of what a pedestrian would see there. Don't keep only "interesting" images: the aim is a uniform sample of the route.

You must end with **exactly 20 images**. If a segment runs out of images without a `K`, or you abort, just run the command again. It will offer to delete the incomplete selection and start over. Add `--select` to force a new selection.

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
- To continue later, run the same command again: images that already have answers are skipped.
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

To re-ask specific Level-1 questions for all images of a sequence:

```bash
python dataset/label_gt.py --continent Asia --city Hanoi --patch q_stairs_visible q_surface_hazardous
```

This deletes the answers to those questions **and their follow-ups** for every image, then asks them again.

> ⚠️ `--override` without `--patch` **overwrites the whole GT file** (it asks for confirmation). Don't use it unless you really want to start from scratch.

### 5.5 Check it

```bash
python vqa/prompt_utils.py stats        # sanity check of the hierarchy
wc -l data/Asia/Hanoi/ground_truth_labels.jsonl   # ~300–550 lines for 20 images
ls data/Asia/Hanoi/images_selected | wc -l        # must be 20
```

### 5.6 Hand it in

Data never goes through git. Pack the sequence and send the archive to Antoni (shared drive):

```bash
dataset/pack_data.sh sequence Asia Hanoi      # -> Asia_Hanoi.tar.gz
```

Tell Antoni the Mapillary sequence ID, the city, and anything special about the sequence. Antoni extracts it at the repository root (`tar -xzf Asia_Hanoi.tar.gz`), runs the models (§7) and the GT review (§5.7), and publishes a new dataset version.

Code changes (bug fixes, new tools) go through git as usual, on a branch and via a pull request.

### 5.7 Review the GT against the models (after the models have run)

[`dataset/review_gt.py`](dataset/review_gt.py) is a small web app that shows only the GT answers where **at least 3 of the 4 models disagree** with the annotator. Most are model errors, but it catches annotation mistakes quickly.

```bash
python dataset/review_gt.py --continent Asia --city Hanoi     # open http://localhost:5000
```

| Key | Action |
|-----|--------|
| `↑` / `↓` | Set GT to Yes / No (saved immediately) |
| `←` / `→` | Previous / next flagged item |

The first change creates `ground_truth_labels.jsonl.bak` with the original file. Only correct the GT when the annotator was clearly wrong, not because the models say so. Afterwards, re-evaluate (`python evaluation/evaluate.py --continent Asia --city Hanoi`) and rebuild the merged GT (`python dataset/merge_gt.py`).

---

## 6. Self-recorded data (Biel Glasses / ZED)

The Barcelona sequences `data/Europe/00` … `22` were recorded around IRI with a ZED stereo camera mounted on Biel Glasses smart glasses (raw `.svo2` files in `data/Europe/Barcelona/svo/`, distributed separately as `VQA-raw-svo.tar.gz`).

- [`dataset/export_svo.py`](dataset/export_svo.py) extracts ~100 evenly spaced left-camera frames per recording to `data/Europe/<seq>/images/*.png`. It needs the ZED SDK and its `pyzed` Python wheel (not in `requirements.txt`).
- `data/Europe/Barcelona/IRI_sequences_GT.txt` contains the planned route of each recording (GPS track points).
- Most Barcelona sequences have **no per-frame GPS** (only `08` has positions in its `metadata.json`), so event maps can't be generated for them.

From step (2) on, the pipeline is the same as for Mapillary.

---

## 7. Running the VQA models

[`vqa/run.py`](vqa/run.py) processes every annotated sequence found in the data folder. For each model and sequence it:

1. Loads the 20 keyframes from `images_selected/`.
2. Asks the 8 Level-1 questions, then the follow-ups of every *yes* answer (Level 2 and then Level 3), with the base context prepended.
3. Writes `results/<model>/answers.jsonl` (**overwritten** on each run).

It only runs the models. Evaluation (§8) and event maps (§9) are separate steps that read `answers.jsonl`, so they can be re-run any time without the GPU.

```bash
python vqa/run.py                                        # all sequences, all 4 models
python vqa/run.py --models qwen-vl --continent Asia --city Hanoi
python evaluation/evaluate.py --continent Asia --city Hanoi --models qwen-vl   # then evaluate
```

Each model is loaded once and reused for all sequences. An error on one sequence is reported at the end and doesn't stop the others.

Answers of generative models are free text. Anything starting with "yes" counts as *yes* and everything else as *no*. ViLT accepts at most 40 text tokens: when the base context plus the question is longer, it is asked the question alone (the "Error processing question" lines in the log).

---

## 8. Evaluation and risk score

[`evaluation/evaluate.py`](evaluation/evaluate.py) matches predictions and GT on `(image_name, question_id)` and computes accuracy, precision, recall, specificity and F1 overall, per level, per topic (Level-1 hazard category) and per question. Levels and topics are derived from the hierarchy in `vqa/prompts.json`.

```bash
python evaluation/evaluate.py                                   # re-evaluate everything (no model is run)
python evaluation/evaluate.py --continent Asia --city Hanoi --models qwen-vl
```

- **Level-2/3 questions are evaluated only where the GT has an answer**, i.e. where the annotator said *yes* to the parent. Follow-ups triggered by a model false positive have no GT and are ignored (see paper §IV-B).

### Risk score (paper §III-C)

For each image, over the 8 Level-1 questions $Q$:

$$R_{img} = \frac{\max\left(0, \sum_{i \in Q} w_i x_i\right)}{\sum_{i \in Q} w_i}, \qquad x_i = \begin{cases} 1 & \text{Yes} \\ -1/|Q| & \text{No} \\ 0 & \text{unanswered} \end{cases}$$

with $w_i$ = 1.0 (Construction, Surface, Non-Sidewalk), 0.6 (Crossings, Stairs, Obstacles), 0.3 (Crowding, Vehicles). All of this is defined once in [`evaluation/risk.py`](evaluation/risk.py) (`HAZARD_CONFIG`, `SAFETY_REWARD_RATIO = 1/8`, thresholds), used by both the evaluation and the event maps.

- **Segment risk** = max of the image risks mapped to that OSM street segment.
- **Categories**: Safe ≤ 0.15 < Caution < 0.4 ≤ Danger < 0.7 ≤ High Risk.
- **MAE_R** = mean |R_gt − R_pred| over images (`Risk_MAE` in the outputs).

Per-sequence outputs: `metrics.json`, `metrics_summary.csv`, `answer_characteristics.txt` in `results/<model>/`.
Cross-sequence tables and figures (paper Tables I–III): `python evaluation/compare_models.py` → `data/model_comparison/`.

---

## 9. Event maps

[`evaluation/event_map.py`](evaluation/event_map.py) builds interactive HTML maps (folium, CartoDB tiles):

- The OSM *drive* network within 1 km of the sequence centre is downloaded with `osmnx` (internet needed).
- Each annotated keyframe is snapped to its nearest street segment, and segments are coloured by their max risk.
- Grey segments have no observations. The dashed blue line is the full GPS track, and markers show the keyframes with their answers.

```bash
python evaluation/event_map.py --continent America --city NewYork --model qwen-vl
# -> results/qwen-vl/interactive_map.html (model) and interactive_map_gt.html (GT)
```

Positions are read from the sequence `metadata.json` (its `filename` entries must match the names in `images_selected/`).

---

## 10. Other utilities

| Command | What it does |
|---|---|
| `python vqa/prompt_utils.py list-presets` | List question presets (`full_hierarchical`, `level_1_only`, `crossing`, ...) |
| `python vqa/prompt_utils.py list-questions --preset crossing` | List the questions of a preset |
| `python vqa/prompt_utils.py stats` | Question counts |
| `python dataset/merge_gt.py` | Merge all GT files into `data/all_GT/all_ground_truth.jsonl` |
| `python evaluation/compare_models.py` | Global / per-continent / per-topic comparison tables and plots |
| `dataset/pack_data.sh dataset \| sequence <Continent> <City> \| svo` | Create the data archives (§3) |

---

## 11. Gotchas

- **`images_selected/` must contain exactly 20 images**, otherwise `label_gt.py` opens the selection again.
- **Quitting (`Q`) during labeling loses the image in progress**, but finished images are saved.
- `vqa/run.py` **overwrites** `results/<model>/answers.jsonl` for the sequences it processes.
- Never commit `.env`, data, or archives. `.gitignore` blocks them, but check `git status` anyway.
- Mapillary images are licensed CC BY-SA 4.0. Keep `metadata.json` (image IDs) so the source can be attributed.

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
