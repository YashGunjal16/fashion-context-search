Read pasted-text.txtThought for 2sHere's the complete README.md content you can copy and paste directly:

```markdown
# Multimodal Fashion & Context Retrieval System  
### CLIP + FAISS + LLM + YOLO + SCHP + Places365

An end-to-end **ML-first**, **multimodal**, **region-aware fashion retrieval system** that goes beyond vanilla CLIP by explicitly modeling **compositional clothing attributes**, **scene context**, and **human body regions**.

---

## 🔥 Why This Project Exists

Most image–text retrieval systems using CLIP fail at **compositional queries**, for example:

- ❌ "red shirt with blue pants" vs "blue shirt with red pants"
- ❌ "white shirt with a red tie"
- ❌ "black jacket over a white inner shirt"
- ❌ "formal outfit inside an office"

This project **fixes that** by introducing **explicit spatial and semantic supervision** on top of CLIP embeddings.

---

## 🧠 Key Idea (In One Paragraph)

Instead of relying only on global CLIP embeddings, this system **decomposes a person image into semantic regions** (upper body, lower body, neck area), **extracts colors per region**, **classifies clothing layers**, **detects environment using Places365**, and **reranks results using structured logic driven by LLM-parsed queries**.

This makes the system **significantly better than vanilla CLIP** for fashion retrieval.

---

## 🏗️ High-Level Architecture

```

User Query (Natural Language)
↓
LLM + Rule-Based Query Parser
↓
Structured Query Attributes
↓
Semantic Retrieval (CLIP + FAISS)
↓
Candidate Images
↓
Attribute-Aware Reranking
↓
Final Results + Explanations

```plaintext

---

## 📦 Core Components

### 1️⃣ Query Understanding (Text → Structure)

**Hybrid Parsing Pipeline**
- Google Gemini (LLM)
- Rule-based NLP fallback
- Confidence-based switching

Extracted attributes:
- Upper garment type
- Lower garment type
- Neck/tie presence
- Colors per region
- Environment
- Style / vibe

---

### 2️⃣ Vision Processing (Image → Structure)

Each image goes through **five ML stages**:

| Stage | Model | Purpose |
|------|------|--------|
| Person Detection | YOLOv8 | Crop human region |
| Human Parsing | SCHP (LIP) | Pixel-level clothing regions |
| Region Segmentation | SCHP masks | Upper / Lower / Neck |
| Color Extraction | KMeans on crops | Region-specific colors |
| Scene Classification | Places365 | Indoor / Outdoor / Runway / Park |

---

### 3️⃣ Semantic Retrieval

- **CLIP ViT-B/32**
- **FAISS IVFFlat index**
- 512-dim normalized embeddings
- Over-fetch + rerank strategy

---

### 4️⃣ Reranking (Where the Magic Happens)

Final ranking is **not CLIP-only**.

We score based on:
- Upper garment color match
- Lower garment color match
- Neck/tie color match
- Garment type consistency
- Scene alignment
- Style / vibe compatibility

Each result also includes a **natural-language explanation**.

---

## 📂 Project Directory Structure

```

fashion-context-search/
│
├── backend/
│ ├── api/
│ │ ├── main.py
│ │ ├── routes.py
│ │ └── schemas.py
│ │
│ ├── indexer/
│ │ ├── build_index.py
│ │ ├── region_extractor.py
│ │ ├── color_extractor.py
│ │ ├── clothing_extractor.py
│ │ ├── vibe_extractor.py
│ │ ├── environment_extractor.py
│ │ ├── tie_extractor.py
│ │ └── clip_zeroshot.py
│ │
│ ├── models/
│ │ ├── clip_loader.py
│ │ ├── places365_loader.py
│ │ ├── scene_loader.py
│ │ └── attribute_head.py
│ │
│ ├── parsing/
│ │ └── schp_parser.py
│ │
│ ├── retrieval/
│ │ ├── search.py
│ │ ├── reranker.py
│ │ ├── query_parser.py
│ │ ├── llm_parser.py
│ │ ├── rule_parser.py
│ │ ├── confidence.py
│ │ └── test_retrieval.py
│ │
│ └── vector_store/
│ └── faiss_store.py
│
├── checkpoints/
│ ├── attribute_head.pt
│ ├── places365_resnet18.pth
│ └── categories_places365.txt
│
├── external/
│ └── SCHP/
│ ├── networks/
│ ├── modules/
│ ├── datasets/
│ ├── utils/
│ ├── simple_extractor.py
│ └── train.py
│
├── data/
│ ├── raw/
│ ├── processed/
│ │ └── faiss_index/
│ │ ├── index.faiss
│ │ └── metadata.json
│ └── metadata/
│
├── frontend/
│ └── app.py
│
├── notebooks/
│ ├── 01_dataset_preparation.ipynb
│ └── 02_attribute_analysis.ipynb
│
├── scripts/
│ └── reduce_dataset.py
│
├── model_cache/
│
├── .env
├── requirements.txt
├── README.md
└── WINDOWS_SETUP_GUIDE.md

```plaintext

---

## 🧪 Dataset

- Source: Fashionpedia + curated runway / street datasets
- Size: 1,000 images (configurable)
- Diversity:
  - Runway
  - Street
  - Park
  - Office
  - Casual / Formal / Editorial

---

## 🔬 Indexing Pipeline (Part A)

### What Happens During Indexing

For **each image**:

1. YOLO detects person
2. SCHP produces segmentation mask
3. Upper / lower / neck masks extracted
4. Region-wise color extraction
5. CLIP image embedding computed
6. Places365 predicts scene
7. Metadata stored in FAISS

### Command

```bash
python -m backend.indexer.build_index \
  --image_dir data/raw \
  --output_dir data/processed/faiss_index
```

---

## Retrieval Pipeline (Part B)

### Example Query

"A white shirt with a red tie in a formal office setting"

### Parsed Output

```json
{
  "upper_item": "shirt",
  "upper_colors": ["white"],
  "neck_item": "tie",
  "neck_colors": ["red"],
  "environment": "office",
  "vibe": "business_formal",
  "confidence": 0.92
}
```

### Why This Works Better Than CLIP Alone

| Vanilla CLIP | This System
|-----|-----
| Global embedding | Region-aware
| No compositionality | Explicit garment roles
| No scene logic | Places365
| No explanation | Human-readable reasoning


---

## ️ Frontend (Streamlit)

- Chat-style UI
- Attribute visualization
- Confidence indicators
- Explanations per result
- Designed for demo + evaluation


---

## ️ Installation (Windows)

### 1. Create Environment

```shellscript
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```shellscript
pip install -r requirements.txt
pip install ninja
```

### 3. Environment Variables

```plaintext
GOOGLE_API_KEY=your_gemini_key
```

---

## Running the System

### Backend

```shellscript
uvicorn backend.api.main:app --reload
```

### Frontend

```shellscript
streamlit run frontend/app.py
```

---

## Evaluation Queries (Assignment)

| Query | Supported
|-----|-----
| Yellow raincoat | ✅
| Business attire in office | ✅
| Blue shirt on park bench | ✅
| Casual city walk | ✅
| Red tie + white shirt | ✅
| Blue shirt + black pants | ✅


---

## Scalability

- FAISS IVFFlat scales to 1M+ images
- Index sharding supported
- Embeddings reusable
- Parsing models frozen


---

## ML-Centric Design Decisions

- Avoided overengineering infra
- Focused on attribute reasoning
- Used pretrained, proven models
- Explicit compositional handling


---

## Files You Can Safely Delete (Cleanup)

### Optional / Junk (After Final Submission)

- notebooks/
- scripts/
- training/
- model_cache/
- **pycache**/
- steps.txt
- QUICKSTART.txt
- package.json


### Do NOT Delete

- external/SCHP/
- checkpoints/
- backend/
- frontend/
- data/processed/


---

## Known Limitations

- SCHP is CPU-heavy
- No fine-grained fabric textures yet
- No multi-person disambiguation
- No temporal reasoning


---

## Future Work

- Lightweight parsing model
- Faster human parsing
- Fabric / pattern classification
- Multi-person queries
- Weather-aware outfits


---

## License

MIT License

---

## Final Note

This project is intentionally ML-heavy, not infra-heavy.

It demonstrates:

- Multimodal reasoning
- Compositional understanding
- Practical ML system design
- Clear extensibility


