from ..extensions import mongo

def get_frames_collection():
    return mongo.db.frames
