# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")
    
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://khoinguyen:khoinguyen5544@web86.m3ooc.mongodb.net/lesson4")
    MONGO_URI2 = os.getenv("MONGO_URI2", "mongodb+srv://EEIoT_newbie:ILOVEAIFOREVER@cluster0.xolr95j.mongodb.net/EEIoT_newbie" )
