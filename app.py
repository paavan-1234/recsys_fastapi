from fastapi import FastAPI, Query
import pickle
import numpy as np
from lightfm import LightFM
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LightFM Recommender API with Filtering")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and mappings
with open("model/lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/item_mapping.pkl", "rb") as f:
    item_mapping = pickle.load(f)

with open("model/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

with open("model/id_to_title.pkl", "rb") as f:
    id_to_title = pickle.load(f)

inv_item_mapping = {v: k for k, v in item_mapping.items()}

@app.get("/recommend")
def recommend(
    user_id: str = Query(..., description="User ID as string"),
    num_recs: int = Query(5, description="Number of recommendations"),
    exclude_watched: bool = Query(True, description="Exclude already watched movies")
):
    n_items = len(item_mapping)
    user_internal_id = dataset.mapping()[0].get(user_id)

    if user_internal_id is None:
        return {"error": f"User ID {user_id} not found in training data."}

    scores = model.predict(user_internal_id, np.arange(n_items))

    if exclude_watched:
        user_items = dataset.build_interactions([(user_id, str(item_id))
                                                 for item_id in inv_item_mapping.keys()])[0]
        watched_items = user_items.tocsr()[user_internal_id].indices
        scores[watched_items] = -np.inf  # Effectively exclude watched items

    top_items = np.argsort(-scores)[:num_recs]
    rec_item_ids = [inv_item_mapping[i] for i in top_items]
    rec_titles = [id_to_title.get(item_id, f"Unknown Title (ID {item_id})") for item_id in rec_item_ids]

    return {
        "user_id": user_id,
        "recommendations": rec_titles
    }
